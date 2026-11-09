// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Narrow ELF64 validator for HSA code objects.
//!
//! The validator does not rewrite the object. It reads only the fixed ELF
//! header, note sections, and symbol tables needed to reject a generated object
//! that the HIP loader cannot consume.

/// Loader-relevant facts read back from a generated code object.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ElfFacts {
    pub elf_class: u8,
    pub endian: u8,
    pub elf_type: u16,
    pub machine: u16,
    pub abi_version: u8,
    pub flags: u32,
    pub metadata_notes: usize,
    pub exported_symbols: Vec<String>,
}

const ELF64_HEADER_SIZE: usize = 64;
const ELF64_SECTION_HEADER_SIZE: usize = 64;
const ELF64_SYMBOL_SIZE: usize = 24;
const ET_DYN: u16 = 3;
const EM_AMDGPU: u16 = 224;
const SHT_SYMTAB: u32 = 2;
const SHT_NOTE: u32 = 7;
const SHT_DYNSYM: u32 = 11;
const SHN_UNDEF: u16 = 0;
const STB_LOCAL: u8 = 0;
const NT_AMDGPU_METADATA: u32 = 32;
const PT_LOAD: u32 = 1;

#[derive(Clone, Copy, Debug)]
struct Section {
    kind: u32,
    offset: usize,
    size: usize,
    link: usize,
    entry_size: usize,
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, String> {
    let raw = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| format!("ELF field at {offset:#x} is truncated"))?;
    Ok(u16::from_le_bytes(raw.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| format!("ELF field at {offset:#x} is truncated"))?;
    Ok(u32::from_le_bytes(raw.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let raw = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| format!("ELF field at {offset:#x} is truncated"))?;
    Ok(u64::from_le_bytes(raw.try_into().unwrap()))
}

fn as_usize(value: u64, field: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{field}={value} does not fit usize"))
}

fn section_at(bytes: &[u8], table: usize, stride: usize, index: usize) -> Result<Section, String> {
    let base = table
        .checked_add(
            stride
                .checked_mul(index)
                .ok_or_else(|| "ELF section-table index overflow".to_string())?,
        )
        .ok_or_else(|| "ELF section-table offset overflow".to_string())?;
    let header = bytes
        .get(base..base + ELF64_SECTION_HEADER_SIZE)
        .ok_or_else(|| format!("ELF section header {index} is truncated"))?;
    Ok(Section {
        kind: read_u32(header, 4)?,
        offset: as_usize(read_u64(header, 24)?, "section offset")?,
        size: as_usize(read_u64(header, 32)?, "section size")?,
        link: read_u32(header, 40)? as usize,
        entry_size: as_usize(read_u64(header, 56)?, "section entry size")?,
    })
}

fn section_bytes(bytes: &[u8], section: Section) -> Result<&[u8], String> {
    let end = section
        .offset
        .checked_add(section.size)
        .ok_or_else(|| "ELF section extent overflow".to_string())?;
    bytes
        .get(section.offset..end)
        .ok_or_else(|| "ELF section extends beyond the object".to_string())
}

fn align4(value: usize) -> Result<usize, String> {
    value
        .checked_add(3)
        .map(|rounded| rounded & !3)
        .ok_or_else(|| "ELF note alignment overflow".to_string())
}

fn metadata_notes(bytes: &[u8], sections: &[Section]) -> Result<usize, String> {
    let mut count = 0;
    for section in sections
        .iter()
        .copied()
        .filter(|section| section.kind == SHT_NOTE)
    {
        let notes = section_bytes(bytes, section)?;
        let mut cursor = 0;
        while cursor < notes.len() {
            if notes.len() - cursor < 12 {
                return Err("ELF note header is truncated".to_string());
            }
            let name_size = read_u32(notes, cursor)? as usize;
            let desc_size = read_u32(notes, cursor + 4)? as usize;
            let note_type = read_u32(notes, cursor + 8)?;
            cursor += 12;
            let name_end = cursor
                .checked_add(name_size)
                .ok_or_else(|| "ELF note name overflow".to_string())?;
            let name = notes
                .get(cursor..name_end)
                .ok_or_else(|| "ELF note name is truncated".to_string())?;
            cursor = align4(name_end)?;
            let desc_end = cursor
                .checked_add(desc_size)
                .ok_or_else(|| "ELF note descriptor overflow".to_string())?;
            notes
                .get(cursor..desc_end)
                .ok_or_else(|| "ELF note descriptor is truncated".to_string())?;
            cursor = align4(desc_end)?;
            if note_type == NT_AMDGPU_METADATA
                && name.strip_suffix(&[0]).unwrap_or(name) == b"AMDGPU"
            {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn string_at(table: &[u8], offset: usize) -> Result<&str, String> {
    let tail = table
        .get(offset..)
        .ok_or_else(|| format!("ELF symbol string offset {offset} is out of range"))?;
    let end = tail
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(tail.len());
    std::str::from_utf8(&tail[..end]).map_err(|error| format!("ELF symbol is not UTF-8: {error}"))
}

fn exported_symbols(bytes: &[u8], sections: &[Section]) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    for section in sections
        .iter()
        .copied()
        .filter(|section| matches!(section.kind, SHT_SYMTAB | SHT_DYNSYM))
    {
        let symbols = section_bytes(bytes, section)?;
        let strings = sections
            .get(section.link)
            .copied()
            .ok_or_else(|| format!("ELF symbol table links missing section {}", section.link))
            .and_then(|strings| section_bytes(bytes, strings))?;
        let stride = if section.entry_size == 0 {
            ELF64_SYMBOL_SIZE
        } else {
            section.entry_size
        };
        if stride < ELF64_SYMBOL_SIZE || symbols.len() % stride != 0 {
            return Err(format!(
                "ELF symbol table size {} is not a multiple of entry size {stride}",
                symbols.len()
            ));
        }
        for symbol in symbols.chunks_exact(stride) {
            let name_offset = read_u32(symbol, 0)? as usize;
            let binding = symbol[4] >> 4;
            let section_index = read_u16(symbol, 6)?;
            if name_offset == 0 || binding == STB_LOCAL || section_index == SHN_UNDEF {
                continue;
            }
            let name = string_at(strings, name_offset)?;
            if !name.is_empty() {
                names.push(name.to_string());
            }
        }
    }
    names.sort();
    names.dedup();
    Ok(names)
}

pub fn validate_hsaco(bytes: &[u8], expected_symbols: &[String]) -> Result<ElfFacts, String> {
    if bytes.len() < ELF64_HEADER_SIZE {
        return Err(format!(
            "HSACO is {} bytes; an ELF64 header needs {ELF64_HEADER_SIZE}",
            bytes.len()
        ));
    }
    if &bytes[..4] != b"\x7fELF" {
        return Err("HSACO has no ELF magic".to_string());
    }
    let elf_class = bytes[4];
    let endian = bytes[5];
    if elf_class != 2 {
        return Err(format!(
            "HSACO ELF class is {elf_class}; expected ELFCLASS64=2"
        ));
    }
    if endian != 1 {
        return Err(format!(
            "HSACO ELF data encoding is {endian}; expected little-endian=1"
        ));
    }
    if bytes[6] != 1 {
        return Err(format!(
            "HSACO ELF version is {}; expected EV_CURRENT=1",
            bytes[6]
        ));
    }
    if !matches!(bytes[7], 0 | 64) {
        return Err(format!(
            "HSACO OS ABI is {}; expected ELFOSABI_NONE=0 or ELFOSABI_AMDGPU_HSA=64",
            bytes[7]
        ));
    }

    let elf_type = read_u16(bytes, 16)?;
    let machine = read_u16(bytes, 18)?;
    if elf_type != ET_DYN {
        return Err(format!(
            "HSACO ELF type is {elf_type}; expected ET_DYN={ET_DYN}"
        ));
    }
    if machine != EM_AMDGPU {
        return Err(format!(
            "HSACO ELF machine is {machine}; expected EM_AMDGPU={EM_AMDGPU}"
        ));
    }
    if read_u64(bytes, 24)? != 0 {
        return Err("HSACO entry point is nonzero".to_string());
    }

    let program_table = as_usize(read_u64(bytes, 32)?, "program table offset")?;
    let program_stride = read_u16(bytes, 54)? as usize;
    let program_count = read_u16(bytes, 56)? as usize;
    if program_count == 0 || program_stride < 56 {
        return Err(format!(
            "HSACO has invalid program table count/stride ({program_count}, {program_stride})"
        ));
    }
    let program_bytes = program_stride
        .checked_mul(program_count)
        .and_then(|size| program_table.checked_add(size))
        .ok_or_else(|| "ELF program-table extent overflow".to_string())?;
    if program_bytes > bytes.len() {
        return Err("ELF program table extends beyond the object".to_string());
    }
    let has_load = (0..program_count).any(|index| {
        read_u32(bytes, program_table + index * program_stride)
            .map(|kind| kind == PT_LOAD)
            .unwrap_or(false)
    });
    if !has_load {
        return Err("HSACO has no PT_LOAD program segment".to_string());
    }

    let table = as_usize(read_u64(bytes, 40)?, "section table offset")?;
    let stride = read_u16(bytes, 58)? as usize;
    let mut count = read_u16(bytes, 60)? as usize;
    if stride < ELF64_SECTION_HEADER_SIZE {
        return Err(format!(
            "ELF section-header size is {stride}; expected at least {ELF64_SECTION_HEADER_SIZE}"
        ));
    }
    if count == 0 {
        let first = section_at(bytes, table, stride, 0)?;
        count = first.size;
    }
    let available = bytes
        .len()
        .checked_sub(table)
        .ok_or_else(|| "ELF section table starts beyond the object".to_string())?
        / stride;
    if count > available {
        return Err(format!(
            "ELF declares {count} sections but only {available} headers fit"
        ));
    }
    let mut sections = Vec::with_capacity(count);
    for index in 0..count {
        sections.push(section_at(bytes, table, stride, index)?);
    }

    let metadata_notes = metadata_notes(bytes, &sections)?;
    if metadata_notes != 1 {
        return Err(format!(
            "HSACO has {metadata_notes} AMDGPU metadata notes; expected exactly 1"
        ));
    }
    let exported_symbols = exported_symbols(bytes, &sections)?;
    for expected in expected_symbols {
        let descriptor = format!("{expected}.kd");
        if !exported_symbols
            .iter()
            .any(|actual| actual == expected || actual == &descriptor)
        {
            return Err(format!(
                "HSACO does not export expected kernel {expected:?}; exports={exported_symbols:?}"
            ));
        }
    }

    Ok(ElfFacts {
        elf_class,
        endian,
        elf_type,
        machine,
        abi_version: bytes[8],
        flags: read_u32(bytes, 48)?,
        metadata_notes,
        exported_symbols,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn section(
        bytes: &mut [u8],
        base: usize,
        kind: u32,
        offset: usize,
        size: usize,
        link: usize,
        entry_size: usize,
    ) {
        put_u32(bytes, base + 4, kind);
        put_u64(bytes, base + 24, offset as u64);
        put_u64(bytes, base + 32, size as u64);
        put_u32(bytes, base + 40, link as u32);
        put_u64(bytes, base + 56, entry_size as u64);
    }

    fn minimal_hsaco() -> Vec<u8> {
        let program_offset = 64;
        let program_size = 56;
        let note_offset = program_offset + program_size;
        let note_size = 24;
        let strings_offset = note_offset + note_size;
        let strings = b"\0kernel\0";
        let symbols_offset = strings_offset + strings.len();
        let symbols_size = ELF64_SYMBOL_SIZE * 2;
        let section_table = symbols_offset + symbols_size;
        let mut bytes = vec![0_u8; section_table + ELF64_SECTION_HEADER_SIZE * 4];

        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[6] = 1;
        bytes[7] = 64;
        bytes[8] = 5;
        put_u16(&mut bytes, 16, ET_DYN);
        put_u16(&mut bytes, 18, EM_AMDGPU);
        put_u64(&mut bytes, 32, program_offset as u64);
        put_u64(&mut bytes, 40, section_table as u64);
        put_u32(&mut bytes, 48, 0x42);
        put_u16(&mut bytes, 54, program_size as u16);
        put_u16(&mut bytes, 56, 1);
        put_u16(&mut bytes, 58, ELF64_SECTION_HEADER_SIZE as u16);
        put_u16(&mut bytes, 60, 4);

        put_u32(&mut bytes, program_offset, PT_LOAD);
        put_u64(&mut bytes, program_offset + 8, note_offset as u64);
        put_u64(
            &mut bytes,
            program_offset + 32,
            (section_table - note_offset) as u64,
        );
        put_u32(&mut bytes, note_offset, 7);
        put_u32(&mut bytes, note_offset + 4, 1);
        put_u32(&mut bytes, note_offset + 8, NT_AMDGPU_METADATA);
        bytes[note_offset + 12..note_offset + 19].copy_from_slice(b"AMDGPU\0");
        bytes[note_offset + 20] = 0xaa;
        bytes[strings_offset..strings_offset + strings.len()].copy_from_slice(strings);

        let symbol = symbols_offset + ELF64_SYMBOL_SIZE;
        put_u32(&mut bytes, symbol, 1);
        bytes[symbol + 4] = 0x12;
        put_u16(&mut bytes, symbol + 6, 1);

        section(
            &mut bytes,
            section_table + ELF64_SECTION_HEADER_SIZE,
            SHT_NOTE,
            note_offset,
            note_size,
            0,
            0,
        );
        section(
            &mut bytes,
            section_table + ELF64_SECTION_HEADER_SIZE * 2,
            3,
            strings_offset,
            strings.len(),
            0,
            0,
        );
        section(
            &mut bytes,
            section_table + ELF64_SECTION_HEADER_SIZE * 3,
            SHT_SYMTAB,
            symbols_offset,
            symbols_size,
            2,
            ELF64_SYMBOL_SIZE,
        );
        bytes
    }

    #[test]
    fn accepts_loadable_amdgpu_shape() {
        let facts = validate_hsaco(&minimal_hsaco(), &["kernel".to_string()]).unwrap();
        assert_eq!(facts.elf_type, ET_DYN);
        assert_eq!(facts.machine, EM_AMDGPU);
        assert_eq!(facts.metadata_notes, 1);
        assert_eq!(facts.exported_symbols, ["kernel"]);
    }

    #[test]
    fn rejects_wrong_machine() {
        let mut bytes = minimal_hsaco();
        put_u16(&mut bytes, 18, 62);
        assert!(validate_hsaco(&bytes, &[])
            .unwrap_err()
            .contains("EM_AMDGPU"));
    }

    #[test]
    fn rejects_missing_expected_symbol() {
        assert!(validate_hsaco(&minimal_hsaco(), &["other".to_string()])
            .unwrap_err()
            .contains("expected kernel"));
    }
}
