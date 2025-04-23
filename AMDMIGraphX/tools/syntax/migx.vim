" Vim syntax file
" Language: MIGraphX Intermediate Representation
" Current Maintainer: Charlie Lin (https://github.com/CharlieL7)
" Previous Maintainer:
" Last Change: 20 Feb 2024

" quit when a syntax file was already loaded
if exists("b:current_syntax")
  finish
endif

syn keyword migx_keyword param
syn keyword migx_keyword return
syn keyword migx_keyword target_id
syn keyword migx_keyword literal
syn match migx_ins_number "@\d\+"
syn match migx_instruction "\s=\s\zs.\+\ze\["
syn match migx_instruction_and_attributes "\s=\s\zs.\+\ze(" contains=migx_attributes 
syn region migx_attributes start="\[" end="\]" contains=migx_keyword
syn match migx_output_type "\s->\s\zs.\{-}\ze," nextgroup=migx_output_dims
syn match migx_output_dims "\s\zs{.\{-}}\ze,\starget_id"
syn match migx_exec_time ":\s\zs\d\{-}\.\d\{-}ms\ze"

let b:current_syntax = "migx"

hi def link migx_keyword NonText
hi def link migx_ins_number Number
hi def link migx_instruction Operator
hi def link migx_instruction_and_attributes Operator
hi def link migx_attributes Comment
hi def link migx_output_type Type
hi def link migx_output_dims Normal
hi def link migx_exec_time NonText
