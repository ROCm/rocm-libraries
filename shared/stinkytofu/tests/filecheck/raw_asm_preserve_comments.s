# RUN: %stinkytofu-opt --arch gfx1250 %s --emit-asm --preserve-comments
#
# Verify that --preserve-comments captures a trailing source-level "//"
# comment from each parsed line and the pipeline re-emits it intact:
#
#   - Real instructions: the comment is attached as a CommentData modifier
#     and the emitter prints it (aligned to commentAlignColumn).
#   - .set directives: the comment is stored on AsmDirective::comment and
#     the SET emitter branch appends "// <comment>" after the assignment.
#   - Free-form ".<directive>" lines that round-trip via TEXTBLOCK have
#     their comment folded back into the verbatim text.
#
# Without the flag the parser would strip the "// ..." text outright; with
# it the original annotation survives the parse → IR → emit cycle.
#
# CHECK: .set vgprFoo, 5 // sets foo
# CHECK: v_mov_b32 v0, v1 {{.*}}// thread serial id
# CHECK: v_add_f32 v2, v0, v1 {{.*}}// add a and b
# CHECK: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.set vgprFoo, 5 // sets foo

v_mov_b32 v0, v1 // thread serial id
v_add_f32 v2, v0, v1 // add a and b
s_endpgm
