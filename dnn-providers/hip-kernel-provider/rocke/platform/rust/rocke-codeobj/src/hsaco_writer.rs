// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Small ELF64/AMDHSA writer for the direct gfx1151 WMMA backend.
//!
//! The HIP loader needs an ET_DYN AMDGPU image with load segments, dynamic
//! symbols, hash tables, a kernel descriptor, and AMDHSA MessagePack metadata.
//! This module constructs those records directly; no object library or linker
//! participates at runtime.

const ELF_HEADER_SIZE: usize = 64;
const PROGRAM_HEADER_SIZE: usize = 56;
const PROGRAM_HEADER_COUNT: usize = 8;
const SECTION_HEADER_SIZE: usize = 64;
const NOTE_OFFSET: usize = 0x200;
const KERNEL_CODE_SIZE: usize = 0x2e8;
const DYNAMIC_SIZE: usize = 7 * 16;

const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_HASH: u32 = 5;
const SHT_DYNAMIC: u32 = 6;
const SHT_NOTE: u32 = 7;
const SHT_NOBITS: u32 = 8;
const SHT_DYNSYM: u32 = 11;
const SHT_STRTAB: u32 = 3;
const SHT_GNU_HASH: u32 = 0x6fff_fff6;

const SHF_WRITE: u64 = 1;
const SHF_ALLOC: u64 = 2;
const SHF_EXECINSTR: u64 = 4;
const SHF_MERGE: u64 = 0x10;
const SHF_STRINGS: u64 = 0x20;

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PT_NOTE: u32 = 4;
const PT_GNU_STACK: u32 = 0x6474_e551;
const PT_GNU_RELRO: u32 = 0x6474_e552;
const PT_PHDR: u32 = 6;

const PF_X: u32 = 1;
const PF_W: u32 = 2;
const PF_R: u32 = 4;

const DT_NULL: u64 = 0;
const DT_HASH: u64 = 4;
const DT_STRTAB: u64 = 5;
const DT_SYMTAB: u64 = 6;
const DT_STRSZ: u64 = 10;
const DT_SYMENT: u64 = 11;
const DT_GNU_HASH: u64 = 0x6fff_fef5;

#[derive(Clone, Copy, Debug)]
struct Layout {
    note: usize,
    dynsym: usize,
    gnu_hash: usize,
    hash: usize,
    dynstr: usize,
    rodata: usize,
    text: usize,
    text_vaddr: usize,
    dynamic: usize,
    dynamic_vaddr: usize,
    comment: usize,
    symtab: usize,
    shstrtab: usize,
    strtab: usize,
    section_headers: usize,
    relro_size: usize,
}

#[derive(Clone, Copy)]
struct SectionHeader {
    name: u32,
    kind: u32,
    flags: u64,
    address: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    alignment: u64,
    entry_size: u64,
}

#[derive(Clone, Copy)]
struct ProgramHeader {
    kind: u32,
    flags: u32,
    offset: u64,
    virtual_address: u64,
    file_size: u64,
    memory_size: u64,
    alignment: u64,
}

struct RegularStringTable {
    bytes: Vec<u8>,
    private_segment_size: u32,
    num_agpr: u32,
    num_vgpr: u32,
    has_dynamic_stack: u32,
    has_recursion: u32,
    kernel: u32,
    descriptor: u32,
    dynamic: u32,
}

fn align_up(value: usize, alignment: usize) -> usize {
    assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn append_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn append_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn msgpack_str(bytes: &mut Vec<u8>, value: &str) {
    let length = value.len();
    if length <= 31 {
        bytes.push(0xa0 | length as u8);
    } else {
        assert!(length <= u8::MAX as usize);
        bytes.extend([0xd9, length as u8]);
    }
    bytes.extend_from_slice(value.as_bytes());
}

fn msgpack_uint(bytes: &mut Vec<u8>, value: u32) {
    if value <= 127 {
        bytes.push(value as u8);
    } else if value <= 0xff {
        bytes.extend([0xcc, value as u8]);
    } else if value <= 0xffff {
        bytes.push(0xcd);
        bytes.extend_from_slice(&(value as u16).to_be_bytes());
    } else {
        bytes.push(0xce);
        bytes.extend_from_slice(&value.to_be_bytes());
    }
}

fn msgpack_array(bytes: &mut Vec<u8>, length: usize) {
    if length <= 15 {
        bytes.push(0x90 | length as u8);
    } else {
        bytes.push(0xdc);
        bytes.extend_from_slice(&(length as u16).to_be_bytes());
    }
}

fn msgpack_bool(bytes: &mut Vec<u8>, value: bool) {
    bytes.push(if value { 0xc3 } else { 0xc2 });
}

/// One kernel argument in the AMDHSA metadata.
#[derive(Clone, Debug)]
pub struct KernelArgMeta {
    pub name: String,
    pub offset: u32,
    pub size: u32,
    /// Global pointer argument (`global_buffer`); otherwise `by_value`.
    pub pointer: bool,
    /// Optional `.actual_access` for pointers.
    pub access: Option<&'static str>,
}

/// Everything that varies between single-kernel gfx1151 code objects.
#[derive(Clone, Debug)]
pub struct KernelObjectSpec<'a> {
    pub symbol: &'a str,
    pub code: &'a [u8],
    pub args: Vec<KernelArgMeta>,
    pub kernarg_size: u32,
    pub group_segment_size: u32,
    pub max_flat_workgroup_size: u32,
    pub sgpr_count: u32,
    pub vgpr_count: u32,
    pub rsrc1: u32,
    pub rsrc2: u32,
    pub rsrc3: u32,
}

fn metadata_argument(bytes: &mut Vec<u8>, arg: &KernelArgMeta) {
    if arg.pointer {
        if let Some(access) = arg.access {
            bytes.push(0x86);
            msgpack_str(bytes, ".actual_access");
            msgpack_str(bytes, access);
        } else {
            bytes.push(0x85);
        }
        msgpack_str(bytes, ".address_space");
        msgpack_str(bytes, "global");
    } else {
        bytes.push(0x84);
    }
    msgpack_str(bytes, ".name");
    msgpack_str(bytes, &arg.name);
    msgpack_str(bytes, ".offset");
    msgpack_uint(bytes, arg.offset);
    msgpack_str(bytes, ".size");
    msgpack_uint(bytes, arg.size);
    msgpack_str(bytes, ".value_kind");
    msgpack_str(bytes, if arg.pointer { "global_buffer" } else { "by_value" });
}

fn gemm_spec<'a>(symbol: &'a str, code: &'a [u8]) -> KernelObjectSpec<'a> {
    let arg = |name: &str, offset, size, access: Option<&'static str>| KernelArgMeta {
        name: name.to_string(),
        offset,
        size,
        pointer: access.is_some(),
        access,
    };
    KernelObjectSpec {
        symbol,
        code,
        args: vec![
            arg("A", 0, 8, Some("read_only")),
            arg("B", 8, 8, Some("read_only")),
            arg("C", 16, 8, Some("write_only")),
            arg("M", 24, 4, None),
            arg("N", 28, 4, None),
            arg("K", 32, 4, None),
        ],
        kernarg_size: 36,
        group_segment_size: 0,
        max_flat_workgroup_size: 32,
        sgpr_count: 14,
        vgpr_count: 30,
        rsrc1: 0xe0af_0003,
        rsrc2: 0x0000_0184,
        rsrc3: 0x60,
    }
}

#[cfg(test)]
fn metadata_note(symbol: &str) -> Vec<u8> {
    metadata_note_for(&gemm_spec(symbol, &[]))
}

fn metadata_note_for(spec: &KernelObjectSpec) -> Vec<u8> {
    let symbol = spec.symbol;
    let mut descriptor = Vec::new();
    descriptor.push(0x83);
    msgpack_str(&mut descriptor, "amdhsa.kernels");
    descriptor.push(0x91);
    descriptor.extend([0xde, 0x00, 0x10]);

    msgpack_str(&mut descriptor, ".args");
    msgpack_array(&mut descriptor, spec.args.len());
    for arg in &spec.args {
        metadata_argument(&mut descriptor, arg);
    }

    msgpack_str(&mut descriptor, ".group_segment_fixed_size");
    msgpack_uint(&mut descriptor, spec.group_segment_size);
    msgpack_str(&mut descriptor, ".kernarg_segment_align");
    msgpack_uint(&mut descriptor, 8);
    msgpack_str(&mut descriptor, ".kernarg_segment_size");
    msgpack_uint(&mut descriptor, spec.kernarg_size);
    msgpack_str(&mut descriptor, ".max_flat_workgroup_size");
    msgpack_uint(&mut descriptor, spec.max_flat_workgroup_size);
    msgpack_str(&mut descriptor, ".name");
    msgpack_str(&mut descriptor, symbol);
    msgpack_str(&mut descriptor, ".private_segment_fixed_size");
    msgpack_uint(&mut descriptor, 0);
    msgpack_str(&mut descriptor, ".sgpr_count");
    msgpack_uint(&mut descriptor, spec.sgpr_count);
    msgpack_str(&mut descriptor, ".sgpr_spill_count");
    msgpack_uint(&mut descriptor, 0);
    msgpack_str(&mut descriptor, ".symbol");
    msgpack_str(&mut descriptor, &format!("{symbol}.kd"));
    msgpack_str(&mut descriptor, ".uniform_work_group_size");
    msgpack_uint(&mut descriptor, 1);
    msgpack_str(&mut descriptor, ".uses_dynamic_stack");
    msgpack_bool(&mut descriptor, false);
    msgpack_str(&mut descriptor, ".vgpr_count");
    msgpack_uint(&mut descriptor, spec.vgpr_count);
    msgpack_str(&mut descriptor, ".vgpr_spill_count");
    msgpack_uint(&mut descriptor, 0);
    msgpack_str(&mut descriptor, ".wavefront_size");
    msgpack_uint(&mut descriptor, 32);
    msgpack_str(&mut descriptor, ".workgroup_processor_mode");
    msgpack_uint(&mut descriptor, 1);

    msgpack_str(&mut descriptor, "amdhsa.target");
    msgpack_str(&mut descriptor, "amdgcn-amd-amdhsa--gfx1151");
    msgpack_str(&mut descriptor, "amdhsa.version");
    descriptor.extend([0x92, 0x01, 0x02]);

    let mut note = Vec::with_capacity(20 + align_up(descriptor.len(), 4));
    append_u32(&mut note, 7);
    append_u32(&mut note, descriptor.len() as u32);
    append_u32(&mut note, 32);
    note.extend_from_slice(b"AMDGPU\0");
    note.resize(20, 0);
    note.extend_from_slice(&descriptor);
    note.resize(align_up(note.len(), 4), 0);
    note
}

fn dynamic_strings(symbol: &str) -> Vec<u8> {
    let mut strings = Vec::with_capacity(symbol.len() * 2 + 6);
    strings.push(0);
    strings.extend_from_slice(symbol.as_bytes());
    strings.push(0);
    strings.extend_from_slice(symbol.as_bytes());
    strings.extend_from_slice(b".kd\0");
    strings
}

fn dynamic_symbols(symbol: &str, text_vaddr: usize, rodata: usize, code_size: usize) -> Vec<u8> {
    let descriptor_name = symbol.len() + 2;
    let mut symbols = vec![0; 24];
    append_u32(&mut symbols, 1);
    symbols.extend([0x12, 0x03]);
    symbols.extend_from_slice(&7_u16.to_le_bytes());
    append_u64(&mut symbols, text_vaddr as u64);
    append_u64(&mut symbols, code_size as u64);
    append_u32(&mut symbols, descriptor_name as u32);
    symbols.extend([0x11, 0x00]);
    symbols.extend_from_slice(&6_u16.to_le_bytes());
    append_u64(&mut symbols, rodata as u64);
    append_u64(&mut symbols, 64);
    symbols
}

fn regular_strings(symbol: &str) -> RegularStringTable {
    fn add(bytes: &mut Vec<u8>, value: &str) -> u32 {
        let offset = bytes.len() as u32;
        bytes.extend_from_slice(value.as_bytes());
        bytes.push(0);
        offset
    }

    let mut bytes = vec![0];
    let private_segment_size = add(&mut bytes, &format!("{symbol}.private_seg_size"));
    let num_agpr = add(&mut bytes, &format!("{symbol}.num_agpr"));
    let num_vgpr = add(&mut bytes, &format!("{symbol}.num_vgpr"));
    let has_dynamic_stack = add(&mut bytes, &format!("{symbol}.has_dyn_sized_stack"));
    let has_recursion = add(&mut bytes, &format!("{symbol}.has_recursion"));
    let kernel = add(&mut bytes, symbol);
    let descriptor = add(&mut bytes, &format!("{symbol}.kd"));
    let dynamic = add(&mut bytes, "_DYNAMIC");
    RegularStringTable {
        bytes,
        private_segment_size,
        num_agpr,
        num_vgpr,
        has_dynamic_stack,
        has_recursion,
        kernel,
        descriptor,
        dynamic,
    }
}

fn append_symbol(
    symbols: &mut Vec<u8>,
    name: u32,
    info: u8,
    other: u8,
    section: u16,
    value: u64,
    size: u64,
) {
    append_u32(symbols, name);
    symbols.extend([info, other]);
    symbols.extend_from_slice(&section.to_le_bytes());
    append_u64(symbols, value);
    append_u64(symbols, size);
}

fn regular_symbols(strings: &RegularStringTable, layout: Layout, code_size: usize, num_vgpr: u32) -> Vec<u8> {
    const SHN_ABS: u16 = 0xfff1;
    let mut symbols = vec![0; 24];
    append_symbol(
        &mut symbols,
        strings.private_segment_size,
        0,
        0,
        SHN_ABS,
        0,
        0,
    );
    append_symbol(&mut symbols, strings.num_agpr, 0, 0, SHN_ABS, 0, 0);
    append_symbol(&mut symbols, strings.num_vgpr, 0, 0, SHN_ABS, num_vgpr as u64, 0);
    append_symbol(&mut symbols, strings.has_dynamic_stack, 0, 0, SHN_ABS, 0, 0);
    append_symbol(&mut symbols, strings.has_recursion, 0, 0, SHN_ABS, 0, 0);
    append_symbol(
        &mut symbols,
        strings.dynamic,
        0,
        2,
        8,
        layout.dynamic_vaddr as u64,
        0,
    );
    append_symbol(
        &mut symbols,
        strings.kernel,
        0x12,
        3,
        7,
        layout.text_vaddr as u64,
        code_size as u64,
    );
    append_symbol(
        &mut symbols,
        strings.descriptor,
        0x11,
        0,
        6,
        layout.rodata as u64,
        64,
    );
    symbols
}

fn gnu_hash(name: &[u8]) -> u32 {
    name.iter().fold(5381_u32, |hash, byte| {
        hash.wrapping_mul(33).wrapping_add(u32::from(*byte))
    })
}

fn gnu_hash_table(symbol: &str) -> Vec<u8> {
    let names = [symbol.to_string(), format!("{symbol}.kd")];
    let hashes = names.map(|name| gnu_hash(name.as_bytes()));
    let mut bloom = 0_u64;
    for hash in hashes {
        bloom |= 1_u64 << (hash % 64);
        bloom |= 1_u64 << ((hash >> 26) % 64);
    }
    let mut table = Vec::with_capacity(36);
    append_u32(&mut table, 1);
    append_u32(&mut table, 1);
    append_u32(&mut table, 1);
    append_u32(&mut table, 26);
    append_u64(&mut table, bloom);
    append_u32(&mut table, 1);
    append_u32(&mut table, hashes[0] & !1);
    append_u32(&mut table, hashes[1] | 1);
    table
}

fn sysv_hash(name: &[u8]) -> u32 {
    let mut hash = 0_u32;
    for byte in name {
        hash = (hash << 4).wrapping_add(u32::from(*byte));
        let high = hash & 0xf000_0000;
        if high != 0 {
            hash ^= high >> 24;
        }
        hash &= !high;
    }
    hash
}

fn sysv_hash_table(symbol: &str) -> Vec<u8> {
    const BUCKETS: usize = 3;
    let names = [symbol.to_string(), format!("{symbol}.kd")];
    let mut buckets = [0_u32; BUCKETS];
    let mut chains = [0_u32; 3];
    for (index, name) in names.iter().enumerate() {
        let symbol_index = index + 1;
        let bucket = sysv_hash(name.as_bytes()) as usize % BUCKETS;
        if buckets[bucket] == 0 {
            buckets[bucket] = symbol_index as u32;
        } else {
            let mut chain = buckets[bucket] as usize;
            while chains[chain] != 0 {
                chain = chains[chain] as usize;
            }
            chains[chain] = symbol_index as u32;
        }
    }
    let mut table = Vec::with_capacity(32);
    append_u32(&mut table, BUCKETS as u32);
    append_u32(&mut table, chains.len() as u32);
    for bucket in buckets {
        append_u32(&mut table, bucket);
    }
    for chain in chains {
        append_u32(&mut table, chain);
    }
    table
}

fn kernel_descriptor(text_vaddr: usize, rodata: usize, spec: &KernelObjectSpec) -> Vec<u8> {
    let mut descriptor = vec![0_u8; 64];
    put_u32(&mut descriptor, 0, spec.group_segment_size);
    put_u32(&mut descriptor, 8, spec.kernarg_size);
    put_u64(
        &mut descriptor,
        16,
        (text_vaddr as i64 - rodata as i64) as u64,
    );
    put_u32(&mut descriptor, 44, spec.rsrc3);
    put_u32(&mut descriptor, 48, spec.rsrc1);
    put_u32(&mut descriptor, 52, spec.rsrc2);
    put_u16(&mut descriptor, 56, 0x0408);
    descriptor
}

fn dynamic_table(layout: Layout, dynstr_size: usize) -> Vec<u8> {
    let entries = [
        (DT_SYMTAB, layout.dynsym as u64),
        (DT_SYMENT, 24),
        (DT_STRTAB, layout.dynstr as u64),
        (DT_STRSZ, dynstr_size as u64),
        (DT_GNU_HASH, layout.gnu_hash as u64),
        (DT_HASH, layout.hash as u64),
        (DT_NULL, 0),
    ];
    let mut dynamic = Vec::with_capacity(DYNAMIC_SIZE);
    for (tag, value) in entries {
        append_u64(&mut dynamic, tag);
        append_u64(&mut dynamic, value);
    }
    dynamic
}

fn write_program_header(bytes: &mut [u8], index: usize, program: ProgramHeader) {
    let base = ELF_HEADER_SIZE + index * PROGRAM_HEADER_SIZE;
    put_u32(bytes, base, program.kind);
    put_u32(bytes, base + 4, program.flags);
    put_u64(bytes, base + 8, program.offset);
    put_u64(bytes, base + 16, program.virtual_address);
    put_u64(bytes, base + 24, program.virtual_address);
    put_u64(bytes, base + 32, program.file_size);
    put_u64(bytes, base + 40, program.memory_size);
    put_u64(bytes, base + 48, program.alignment);
}

fn write_section_header(bytes: &mut [u8], table: usize, index: usize, section: SectionHeader) {
    let base = table + index * SECTION_HEADER_SIZE;
    put_u32(bytes, base, section.name);
    put_u32(bytes, base + 4, section.kind);
    put_u64(bytes, base + 8, section.flags);
    put_u64(bytes, base + 16, section.address);
    put_u64(bytes, base + 24, section.offset);
    put_u64(bytes, base + 32, section.size);
    put_u32(bytes, base + 40, section.link);
    put_u32(bytes, base + 44, section.info);
    put_u64(bytes, base + 48, section.alignment);
    put_u64(bytes, base + 56, section.entry_size);
}

fn append_section_name(table: &mut Vec<u8>, name: &str) -> u32 {
    let offset = table.len();
    table.extend_from_slice(name.as_bytes());
    table.push(0);
    offset as u32
}

/// Construct a loadable gfx1151 AMDHSA code object around encoded kernel code.
pub fn write_gfx1151_wmma_gemm(symbol: &str, code: &[u8]) -> Result<Vec<u8>, String> {
    if code.len() != KERNEL_CODE_SIZE {
        return Err(format!(
            "gfx1151 WMMA code is {} bytes; expected {KERNEL_CODE_SIZE}",
            code.len()
        ));
    }
    write_kernel_object(&gemm_spec(symbol, code))
}

/// Construct a loadable single-kernel gfx1151 AMDHSA code object.
pub fn write_kernel_object(spec: &KernelObjectSpec) -> Result<Vec<u8>, String> {
    let symbol = spec.symbol;
    let code = spec.code;
    if symbol.is_empty()
        || symbol.len() > 200
        || !symbol
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'$'))
    {
        return Err(format!("unsupported ELF kernel symbol {symbol:?}"));
    }
    if code.is_empty() || code.len() % 4 != 0 {
        return Err(format!("kernel code is {} bytes; expected whole dwords", code.len()));
    }
    let text_size = align_up(code.len(), 128) + 384;

    let note = metadata_note_for(spec);
    let dynstr = dynamic_strings(symbol);
    let gnu_hash = gnu_hash_table(symbol);
    let hash = sysv_hash_table(symbol);
    let regular_strings = regular_strings(symbol);

    let dynsym_offset = align_up(NOTE_OFFSET + note.len(), 8);
    let gnu_hash_offset = dynsym_offset + 72;
    let hash_offset = gnu_hash_offset + gnu_hash.len();
    let dynstr_offset = hash_offset + hash.len();
    let rodata_offset = align_up(dynstr_offset + dynstr.len(), 64);
    let text_offset = align_up(rodata_offset + 64, 256);
    let text_vaddr = text_offset + 0x1000;
    let dynamic_offset = text_offset + text_size;
    let dynamic_vaddr = align_up(text_vaddr + text_size, 0x1000) + (dynamic_offset & 0xfff);
    let relro_size = align_up(dynamic_vaddr + DYNAMIC_SIZE, 0x1000) - dynamic_vaddr;
    // Keep non-loadable tables at the qualified object's stable offsets while
    // recording honest producer provenance instead of LLVM linker text.
    let mut comment = [0_u8; 0x9c];
    let producer = b"rocKE Rust direct gfx1151 WMMA encoder + ELF writer\0";
    comment[..producer.len()].copy_from_slice(producer);
    let comment_offset = dynamic_offset + DYNAMIC_SIZE;

    let mut shstrtab = vec![0];
    let section_names = [
        append_section_name(&mut shstrtab, ".note"),
        append_section_name(&mut shstrtab, ".dynsym"),
        append_section_name(&mut shstrtab, ".gnu.hash"),
        append_section_name(&mut shstrtab, ".hash"),
        append_section_name(&mut shstrtab, ".dynstr"),
        append_section_name(&mut shstrtab, ".rodata"),
        append_section_name(&mut shstrtab, ".text"),
        append_section_name(&mut shstrtab, ".dynamic"),
        append_section_name(&mut shstrtab, ".relro_padding"),
        append_section_name(&mut shstrtab, ".AMDGPU.gpr_maximums"),
        append_section_name(&mut shstrtab, ".comment"),
        append_section_name(&mut shstrtab, ".symtab"),
        append_section_name(&mut shstrtab, ".shstrtab"),
        append_section_name(&mut shstrtab, ".strtab"),
    ];
    const SYMBOL_TABLE_SIZE: usize = 9 * 24;
    let symtab_offset = align_up(comment_offset + comment.len(), 8);
    let shstrtab_offset = symtab_offset + SYMBOL_TABLE_SIZE;
    let strtab_offset = shstrtab_offset + shstrtab.len();
    let section_headers = align_up(strtab_offset + regular_strings.bytes.len(), 8);
    const SECTION_COUNT: usize = 15;

    let layout = Layout {
        note: NOTE_OFFSET,
        dynsym: dynsym_offset,
        gnu_hash: gnu_hash_offset,
        hash: hash_offset,
        dynstr: dynstr_offset,
        rodata: rodata_offset,
        text: text_offset,
        text_vaddr,
        dynamic: dynamic_offset,
        dynamic_vaddr,
        comment: comment_offset,
        symtab: symtab_offset,
        shstrtab: shstrtab_offset,
        strtab: strtab_offset,
        section_headers,
        relro_size,
    };

    let dynsym = dynamic_symbols(symbol, layout.text_vaddr, layout.rodata, code.len());
    let symtab = regular_symbols(&regular_strings, layout, code.len(), spec.vgpr_count);
    assert_eq!(symtab.len(), SYMBOL_TABLE_SIZE);
    let rodata = kernel_descriptor(layout.text_vaddr, layout.rodata, spec);
    let dynamic = dynamic_table(layout, dynstr.len());
    let total_size = layout.section_headers + SECTION_COUNT * SECTION_HEADER_SIZE;
    let mut image = vec![0_u8; total_size];

    image[layout.note..layout.note + note.len()].copy_from_slice(&note);
    image[layout.dynsym..layout.dynsym + dynsym.len()].copy_from_slice(&dynsym);
    image[layout.gnu_hash..layout.gnu_hash + gnu_hash.len()].copy_from_slice(&gnu_hash);
    image[layout.hash..layout.hash + hash.len()].copy_from_slice(&hash);
    image[layout.dynstr..layout.dynstr + dynstr.len()].copy_from_slice(&dynstr);
    image[layout.rodata..layout.rodata + rodata.len()].copy_from_slice(&rodata);
    image[layout.text..layout.text + code.len()].copy_from_slice(code);
    for offset in (layout.text + code.len()..layout.text + text_size).step_by(4) {
        image[offset..offset + 4].copy_from_slice(&0xbf9f_0000_u32.to_le_bytes());
    }
    image[layout.dynamic..layout.dynamic + dynamic.len()].copy_from_slice(&dynamic);
    image[layout.comment..layout.comment + comment.len()].copy_from_slice(&comment);
    image[layout.symtab..layout.symtab + symtab.len()].copy_from_slice(&symtab);
    image[layout.shstrtab..layout.shstrtab + shstrtab.len()].copy_from_slice(&shstrtab);
    image[layout.strtab..layout.strtab + regular_strings.bytes.len()]
        .copy_from_slice(&regular_strings.bytes);

    image[..16].copy_from_slice(b"\x7fELF\x02\x01\x01\x40\x04\0\0\0\0\0\0\0");
    put_u16(&mut image, 16, 3);
    put_u16(&mut image, 18, 224);
    put_u32(&mut image, 20, 1);
    put_u64(&mut image, 24, 0);
    put_u64(&mut image, 32, ELF_HEADER_SIZE as u64);
    put_u64(&mut image, 40, layout.section_headers as u64);
    put_u32(&mut image, 48, 0x4a);
    put_u16(&mut image, 52, ELF_HEADER_SIZE as u16);
    put_u16(&mut image, 54, PROGRAM_HEADER_SIZE as u16);
    put_u16(&mut image, 56, PROGRAM_HEADER_COUNT as u16);
    put_u16(&mut image, 58, SECTION_HEADER_SIZE as u16);
    put_u16(&mut image, 60, SECTION_COUNT as u16);
    put_u16(&mut image, 62, 13);

    write_program_header(
        &mut image,
        0,
        ProgramHeader {
            kind: PT_PHDR,
            flags: PF_R,
            offset: ELF_HEADER_SIZE as u64,
            virtual_address: ELF_HEADER_SIZE as u64,
            file_size: (PROGRAM_HEADER_COUNT * PROGRAM_HEADER_SIZE) as u64,
            memory_size: (PROGRAM_HEADER_COUNT * PROGRAM_HEADER_SIZE) as u64,
            alignment: 8,
        },
    );
    write_program_header(
        &mut image,
        1,
        ProgramHeader {
            kind: PT_LOAD,
            flags: PF_R,
            offset: 0,
            virtual_address: 0,
            file_size: layout.text as u64,
            memory_size: layout.text as u64,
            alignment: 0x1000,
        },
    );
    write_program_header(
        &mut image,
        2,
        ProgramHeader {
            kind: PT_LOAD,
            flags: PF_R | PF_X,
            offset: layout.text as u64,
            virtual_address: layout.text_vaddr as u64,
            file_size: text_size as u64,
            memory_size: text_size as u64,
            alignment: 0x1000,
        },
    );
    write_program_header(
        &mut image,
        3,
        ProgramHeader {
            kind: PT_LOAD,
            flags: PF_R | PF_W,
            offset: layout.dynamic as u64,
            virtual_address: layout.dynamic_vaddr as u64,
            file_size: DYNAMIC_SIZE as u64,
            memory_size: layout.relro_size as u64,
            alignment: 0x1000,
        },
    );
    write_program_header(
        &mut image,
        4,
        ProgramHeader {
            kind: PT_DYNAMIC,
            flags: PF_R | PF_W,
            offset: layout.dynamic as u64,
            virtual_address: layout.dynamic_vaddr as u64,
            file_size: DYNAMIC_SIZE as u64,
            memory_size: DYNAMIC_SIZE as u64,
            alignment: 8,
        },
    );
    write_program_header(
        &mut image,
        5,
        ProgramHeader {
            kind: PT_GNU_RELRO,
            flags: PF_R,
            offset: layout.dynamic as u64,
            virtual_address: layout.dynamic_vaddr as u64,
            file_size: DYNAMIC_SIZE as u64,
            memory_size: layout.relro_size as u64,
            alignment: 1,
        },
    );
    write_program_header(
        &mut image,
        6,
        ProgramHeader {
            kind: PT_GNU_STACK,
            flags: PF_R | PF_W,
            offset: 0,
            virtual_address: 0,
            file_size: 0,
            memory_size: 0,
            alignment: 0,
        },
    );
    write_program_header(
        &mut image,
        7,
        ProgramHeader {
            kind: PT_NOTE,
            flags: PF_R,
            offset: layout.note as u64,
            virtual_address: layout.note as u64,
            file_size: note.len() as u64,
            memory_size: note.len() as u64,
            alignment: 4,
        },
    );

    let sections = [
        SectionHeader {
            name: 0,
            kind: SHT_NULL,
            flags: 0,
            address: 0,
            offset: 0,
            size: 0,
            link: 0,
            info: 0,
            alignment: 0,
            entry_size: 0,
        },
        SectionHeader {
            name: section_names[0],
            kind: SHT_NOTE,
            flags: SHF_ALLOC,
            address: layout.note as u64,
            offset: layout.note as u64,
            size: note.len() as u64,
            link: 0,
            info: 0,
            alignment: 4,
            entry_size: 0,
        },
        SectionHeader {
            name: section_names[1],
            kind: SHT_DYNSYM,
            flags: SHF_ALLOC,
            address: layout.dynsym as u64,
            offset: layout.dynsym as u64,
            size: dynsym.len() as u64,
            link: 5,
            info: 1,
            alignment: 8,
            entry_size: 24,
        },
        SectionHeader {
            name: section_names[2],
            kind: SHT_GNU_HASH,
            flags: SHF_ALLOC,
            address: layout.gnu_hash as u64,
            offset: layout.gnu_hash as u64,
            size: gnu_hash.len() as u64,
            link: 2,
            info: 0,
            alignment: 8,
            entry_size: 0,
        },
        SectionHeader {
            name: section_names[3],
            kind: SHT_HASH,
            flags: SHF_ALLOC,
            address: layout.hash as u64,
            offset: layout.hash as u64,
            size: hash.len() as u64,
            link: 2,
            info: 0,
            alignment: 4,
            entry_size: 4,
        },
        SectionHeader {
            name: section_names[4],
            kind: SHT_STRTAB,
            flags: SHF_ALLOC,
            address: layout.dynstr as u64,
            offset: layout.dynstr as u64,
            size: dynstr.len() as u64,
            link: 0,
            info: 0,
            alignment: 1,
            entry_size: 0,
        },
        SectionHeader {
            name: section_names[5],
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC,
            address: layout.rodata as u64,
            offset: layout.rodata as u64,
            size: rodata.len() as u64,
            link: 0,
            info: 0,
            alignment: 64,
            entry_size: 0,
        },
        SectionHeader {
            name: section_names[6],
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            address: layout.text_vaddr as u64,
            offset: layout.text as u64,
            size: text_size as u64,
            link: 0,
            info: 0,
            alignment: 256,
            entry_size: 0,
        },
        SectionHeader {
            name: section_names[7],
            kind: SHT_DYNAMIC,
            flags: SHF_ALLOC | SHF_WRITE,
            address: layout.dynamic_vaddr as u64,
            offset: layout.dynamic as u64,
            size: dynamic.len() as u64,
            link: 5,
            info: 0,
            alignment: 8,
            entry_size: 16,
        },
        SectionHeader {
            name: section_names[8],
            kind: SHT_NOBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            address: (layout.dynamic_vaddr + DYNAMIC_SIZE) as u64,
            offset: (layout.dynamic + DYNAMIC_SIZE) as u64,
            size: (layout.relro_size - DYNAMIC_SIZE) as u64,
            link: 0,
            info: 0,
            alignment: 1,
            entry_size: 0,
        },
        SectionHeader {
            name: section_names[9],
            kind: SHT_PROGBITS,
            flags: 0,
            address: 0,
            offset: layout.comment as u64,
            size: 0,
            link: 0,
            info: 0,
            alignment: 1,
            entry_size: 0,
        },
        SectionHeader {
            name: section_names[10],
            kind: SHT_PROGBITS,
            flags: SHF_MERGE | SHF_STRINGS,
            address: 0,
            offset: layout.comment as u64,
            size: comment.len() as u64,
            link: 0,
            info: 0,
            alignment: 1,
            entry_size: 1,
        },
        SectionHeader {
            name: section_names[11],
            kind: SHT_SYMTAB,
            flags: 0,
            address: 0,
            offset: layout.symtab as u64,
            size: symtab.len() as u64,
            link: 14,
            info: 7,
            alignment: 8,
            entry_size: 24,
        },
        SectionHeader {
            name: section_names[12],
            kind: SHT_STRTAB,
            flags: 0,
            address: 0,
            offset: layout.shstrtab as u64,
            size: shstrtab.len() as u64,
            link: 0,
            info: 0,
            alignment: 1,
            entry_size: 0,
        },
        SectionHeader {
            name: section_names[13],
            kind: SHT_STRTAB,
            flags: 0,
            address: 0,
            offset: layout.strtab as u64,
            size: regular_strings.bytes.len() as u64,
            link: 0,
            info: 0,
            alignment: 1,
            entry_size: 0,
        },
    ];
    for (index, section) in sections.into_iter().enumerate() {
        write_section_header(&mut image, layout.section_headers, index, section);
    }
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::validate_hsaco;
    use crate::gfx1151_isa::encode_wmma_gemm;

    const SYMBOL: &str = "rocke_direct_wmma_gemm_wmma16x16x16_fp16_rcr_xm";
    const ORACLE_HEX: &str = include_str!("gfx1151_wmma_gemm.hex");

    fn oracle() -> Vec<u8> {
        let digits: String = ORACLE_HEX.chars().filter(|c| !c.is_whitespace()).collect();
        (0..digits.len())
            .step_by(2)
            .map(|offset| u8::from_str_radix(&digits[offset..offset + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn metadata_and_loaded_sections_match_the_linker_oracle() {
        let oracle = oracle();
        let image = write_gfx1151_wmma_gemm(SYMBOL, &encode_wmma_gemm(true)).unwrap();

        assert_eq!(metadata_note(SYMBOL), oracle[0x200..0x5ac]);
        assert_eq!(&image[0x200..0xbf0], &oracle[0x200..0xbf0]);
        assert_eq!(image.len(), oracle.len());
        assert!(image
            .iter()
            .zip(&oracle)
            .enumerate()
            .filter(|(_, (generated, expected))| generated != expected)
            .all(|(offset, _)| (0xbf0..0xc8c).contains(&offset)));
    }

    #[test]
    fn emitted_elf_is_valid_and_contains_dynamic_symbol() {
        let image = write_gfx1151_wmma_gemm(SYMBOL, &encode_wmma_gemm(true)).unwrap();
        let facts = validate_hsaco(&image, &[SYMBOL.to_string()]).unwrap();
        assert_eq!(facts.flags, 0x4a);
        assert_eq!(facts.metadata_notes, 1);
        assert!(facts.exported_symbols.iter().any(|item| item == SYMBOL));
    }

    #[test]
    fn symbol_and_grid_order_are_generated_inputs() {
        let symbol = "different_wmma_gemm_wmma16x16x16_fp16_rcr_xn";
        let image = write_gfx1151_wmma_gemm(symbol, &encode_wmma_gemm(false)).unwrap();
        let facts = validate_hsaco(&image, &[symbol.to_string()]).unwrap();
        assert!(facts.exported_symbols.iter().any(|item| item == symbol));
        assert!(image
            .windows(symbol.len())
            .any(|window| window == symbol.as_bytes()));
    }
}
