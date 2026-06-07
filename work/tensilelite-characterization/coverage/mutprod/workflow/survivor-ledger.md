# Survivor Ledger

Triage and disposition of mutation survivors across all groups. One row per triaged survivor, grouped by function.

- **total survivors:** 131
- **triaged:** 131
- **untriaged (remaining):** 0

Verdict legend: KILLED = new/strengthened test fails the mutant and passes clean source; EQUIVALENT = behaviorally indistinguishable, unkillable; BAD = pragma applied (intentionally-unhelpful, no behavioral contract); — = not separately re-verified.

## `_validateWorkGroup` (Tensile.TensileLogic.ValidWorkGroup)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x__validateWorkGroup__mutmut_6 | _validateWorkGroup | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroup__validateWorkGroup_char.py::test_rejection_prints_formatted_diagnostic |

## `_chipIdDirFromPath` (Tensile.TensileLogic.ValidChipId)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x__chipIdDirFromPath__mutmut_12 | _chipIdDirFromPath | missing-assertion-strength | add-test | KILLED | test_mut_ValidChipId__chipIdDirFromPath_char.py::test_chip_id_dir_immediate_parent_is_inspected |

## `_report_xcc_failure` (Tensile.TensileLogic.ValidWorkGroupMappingXCC)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x__report_xcc_failure__mutmut_12 | _report_xcc_failure | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__report_xcc_failure_char.py::test_first_failure_prints_full_error_line |
| x__report_xcc_failure__mutmut_13 | _report_xcc_failure | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__report_xcc_failure_char.py::test_first_failure_prints_full_error_line |
| x__report_xcc_failure__mutmut_14 | _report_xcc_failure | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__report_xcc_failure_char.py::test_first_failure_message_is_not_literal_none |
| x__report_xcc_failure__mutmut_15 | _report_xcc_failure | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__report_xcc_failure_char.py::test_present_solution_index_is_emitted |
| x__report_xcc_failure__mutmut_16 | _report_xcc_failure | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__report_xcc_failure_char.py::test_absent_solution_index_falls_back_to_question_mark |
| x__report_xcc_failure__mutmut_17 | _report_xcc_failure | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__report_xcc_failure_char.py::test_absent_solution_index_falls_back_to_question_mark |
| x__report_xcc_failure__mutmut_18 | _report_xcc_failure | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__report_xcc_failure_char.py::test_absent_solution_index_falls_back_to_question_mark |
| x__report_xcc_failure__mutmut_19 | _report_xcc_failure | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__report_xcc_failure_char.py::test_present_solution_index_is_emitted |
| x__report_xcc_failure__mutmut_20 | _report_xcc_failure | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__report_xcc_failure_char.py::test_present_solution_index_is_emitted |
| x__report_xcc_failure__mutmut_21 | _report_xcc_failure | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__report_xcc_failure_char.py::test_present_solution_index_is_emitted |
| x__report_xcc_failure__mutmut_22 | _report_xcc_failure | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__report_xcc_failure_char.py::test_absent_solution_index_falls_back_to_question_mark |
| x__report_xcc_failure__mutmut_23 | _report_xcc_failure | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__report_xcc_failure_char.py::test_third_failure_is_silent |
| x__report_xcc_failure__mutmut_24 | _report_xcc_failure | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__report_xcc_failure_char.py::test_second_failure_prints_more_line |
| x__report_xcc_failure__mutmut_25 | _report_xcc_failure | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__report_xcc_failure_char.py::test_second_failure_more_line_is_not_literal_none |

## `_cu_count_from_path` (Tensile.TensileLogic.ValidWorkGroupMappingXCC)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x__cu_count_from_path__mutmut_9 | _cu_count_from_path | equivalent | none | EQUIVALENT | Regex literal `_(\d+)cu$` -> `_(\d+)CU$`; both compiled with re.IGNORECASE so the literal 'cu'/'CU' match the identical input set and the captured group is unaffected. Verified empirically in-container for cu/CU/Cu/cU and no-match inputs; no valid input distinguishes original from mutant. Unkillable. |

## `_validateWorkGroupMappingXCC` (Tensile.TensileLogic.ValidWorkGroupMappingXCC)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x__validateWorkGroupMappingXCC__mutmut_3 | _validateWorkGroupMappingXCC | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__validateWorkGroupMappingXCC_char.py::test_non_cu_dir_skips_invalid_xcc |
| x__validateWorkGroupMappingXCC__mutmut_4 | _validateWorkGroupMappingXCC | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__validateWorkGroupMappingXCC_char.py::test_one_cu_dir_validates_invalid_xcc |
| x__validateWorkGroupMappingXCC__mutmut_14 | _validateWorkGroupMappingXCC | equivalent | none | EQUIVALENT | Default for missing key -1 -> +1; only matters when WorkGroupMappingXCC absent. Original (xcc=-1) early-accepts True; mutant (xcc=1) is positive, a power of two (1&0==0), divides any cu_count>0, also returns True with no failure report. No state/print difference for any valid input. |
| x__validateWorkGroupMappingXCC__mutmut_21 | _validateWorkGroupMappingXCC | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__validateWorkGroupMappingXCC_char.py::test_xcc_one_is_accepted_on_cu_dir |
| x__validateWorkGroupMappingXCC__mutmut_23 | _validateWorkGroupMappingXCC | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__validateWorkGroupMappingXCC_char.py::test_nonpositive_message_includes_index_and_detail |
| x__validateWorkGroupMappingXCC__mutmut_24 | _validateWorkGroupMappingXCC | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__validateWorkGroupMappingXCC_char.py::test_nonpositive_message_includes_index_and_detail |
| x__validateWorkGroupMappingXCC__mutmut_31 | _validateWorkGroupMappingXCC | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__validateWorkGroupMappingXCC_char.py::test_xcc_one_is_accepted_on_cu_dir |
| x__validateWorkGroupMappingXCC__mutmut_35 | _validateWorkGroupMappingXCC | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__validateWorkGroupMappingXCC_char.py::test_not_power_of_two_message_includes_index_and_detail |
| x__validateWorkGroupMappingXCC__mutmut_36 | _validateWorkGroupMappingXCC | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__validateWorkGroupMappingXCC_char.py::test_not_power_of_two_message_includes_index_and_detail |
| x__validateWorkGroupMappingXCC__mutmut_45 | _validateWorkGroupMappingXCC | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__validateWorkGroupMappingXCC_char.py::test_does_not_divide_message_includes_index_and_detail |
| x__validateWorkGroupMappingXCC__mutmut_46 | _validateWorkGroupMappingXCC | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__validateWorkGroupMappingXCC_char.py::test_does_not_divide_message_includes_index_and_detail |
| x__validateWorkGroupMappingXCC__mutmut_52 | _validateWorkGroupMappingXCC | missing-assertion-strength | add-test | KILLED | test_mut_ValidWorkGroupMappingXCC__validateWorkGroupMappingXCC_char.py::test_exception_path_prints_error_message |

## `_validateMatrixInstruction` (Tensile.TensileLogic.ValidMatrixInstruction)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x__validateMatrixInstruction__mutmut_9 | _validateMatrixInstruction | missing-assertion-strength | add-test | KILLED | test_mut_ValidMatrixInstruction__validateMatrixInstruction_char.py::test_reject_branch_prints_error_diagnostic |

## `print1` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x_print1__mutmut_1 | print1 | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_print1_char.py::test_print1_prints_at_verbosity_one |
| x_print1__mutmut_2 | print1 | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_print1_char.py::test_print1_prints_at_verbosity_one |
| x_print1__mutmut_3 | print1 | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_print1_char.py::test_print1_prints_exact_message |

## `printWarning` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x_printWarning__mutmut_3 | printWarning | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_printWarning_char.py::test_print_warning_exact_format |

## `printExit` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x_printExit__mutmut_3 | printExit | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_printExit_char.py::test_print_exit_exact_message_prefix |
| x_printExit__mutmut_6 | printExit | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_printExit_char.py::test_print_exit_code_is_negative_one |
| x_printExit__mutmut_7 | printExit | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_printExit_char.py::test_print_exit_code_is_negative_one |
| x_printExit__mutmut_8 | printExit | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_printExit_char.py::test_print_exit_code_is_negative_one |

## `locateExe` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x_locateExe__mutmut_7 | locateExe | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_locateExe_char.py::test_locateExe_splits_path_on_pathsep |
| x_locateExe__mutmut_16 | locateExe | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_locateExe_char.py::test_locateExe_not_found_message_contains_exeName |

## `ensurePath` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x_ensurePath__mutmut_2 | ensurePath | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ensurePath_char.py::test_ensurepath_oserror_reraises_exact_message_and_type |
| x_ensurePath__mutmut_3 | ensurePath | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ensurePath_char.py::test_ensurepath_oserror_reraises_exact_message_and_type |
| x_ensurePath__mutmut_4 | ensurePath | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ensurePath_char.py::test_ensurepath_oserror_reraises_exact_message_and_type |
| x_ensurePath__mutmut_5 | ensurePath | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ensurePath_char.py::test_ensurepath_oserror_reraises_exact_message_and_type |
| x_ensurePath__mutmut_6 | ensurePath | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ensurePath_char.py::test_ensurepath_oserror_reraises_exact_message_and_type |

## `versionIsCompatible` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x_versionIsCompatible__mutmut_12 | versionIsCompatible | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_versionIsCompatible_char.py::test_higher_query_minor_is_incompatible |
| x_versionIsCompatible__mutmut_13 | versionIsCompatible | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_versionIsCompatible_char.py::test_equal_minor_branch_is_entered_for_step_check |
| x_versionIsCompatible__mutmut_17 | versionIsCompatible | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_versionIsCompatible_char.py::test_higher_step_when_minor_equal_is_incompatible |

## `ProgressBar.__init__` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| xǁProgressBarǁ__init____mutmut_1 | ProgressBar.__init__ | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar___init___char.py::test_init_default_width_and_maxticks |
| xǁProgressBarǁ__init____mutmut_3 | ProgressBar.__init__ | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar___init___char.py::test_init_char_is_pipe |
| xǁProgressBarǁ__init____mutmut_7 | ProgressBar.__init__ | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar___init___char.py::test_init_default_width_and_maxticks |
| xǁProgressBarǁ__init____mutmut_8 | ProgressBar.__init__ | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar___init___char.py::test_init_default_width_and_maxticks |
| xǁProgressBarǁ__init____mutmut_10 | ProgressBar.__init__ | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar___init___char.py::test_init_priorvalue_zero |
| xǁProgressBarǁ__init____mutmut_11 | ProgressBar.__init__ | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar___init___char.py::test_init_fraction_zero |
| xǁProgressBarǁ__init____mutmut_12 | ProgressBar.__init__ | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar___init___char.py::test_init_fraction_zero |
| xǁProgressBarǁ__init____mutmut_14 | ProgressBar.__init__ | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar___init___char.py::test_init_numticks_zero |

## `ProgressBar.increment` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| xǁProgressBarǁincrement__mutmut_1 | ProgressBar.increment | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar_increment_char.py::test_increment_default_step_is_one |

## `ProgressBar.update` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| xǁProgressBarǁupdate__mutmut_2 | ProgressBar.update | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar_update_char.py::test_update_fraction_is_value_div_maxvalue |
| xǁProgressBarǁupdate__mutmut_3 | ProgressBar.update | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar_update_char.py::test_update_fraction_is_value_div_maxvalue |
| xǁProgressBarǁupdate__mutmut_4 | ProgressBar.update | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar_update_char.py::test_update_fraction_is_value_div_maxvalue |
| xǁProgressBarǁupdate__mutmut_8 | ProgressBar.update | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar_update_char.py::test_update_no_advance_when_ticks_equal |

## `ProgressBar.printStatus` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| xǁProgressBarǁprintStatus__mutmut_2 | ProgressBar.printStatus | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar_printStatus_char.py::test_printstatus_incomplete_line_exact_output |
| xǁProgressBarǁprintStatus__mutmut_5 | ProgressBar.printStatus | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar_printStatus_char.py::test_printstatus_incomplete_line_exact_output |
| xǁProgressBarǁprintStatus__mutmut_8 | ProgressBar.printStatus | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar_printStatus_char.py::test_printstatus_incomplete_line_exact_output |
| xǁProgressBarǁprintStatus__mutmut_9 | ProgressBar.printStatus | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar_printStatus_char.py::test_printstatus_incomplete_line_exact_output |
| xǁProgressBarǁprintStatus__mutmut_10 | ProgressBar.printStatus | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar_printStatus_char.py::test_printstatus_incomplete_has_no_elapsed_suffix |
| xǁProgressBarǁprintStatus__mutmut_14 | ProgressBar.printStatus | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar_printStatus_char.py::test_printstatus_complete_line_exact_output |
| xǁProgressBarǁprintStatus__mutmut_15 | ProgressBar.printStatus | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar_printStatus_char.py::test_printstatus_complete_line_exact_output |
| xǁProgressBarǁprintStatus__mutmut_16 | ProgressBar.printStatus | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ProgressBar_printStatus_char.py::test_printstatus_elapsed_is_difference_not_sum |

## `SpinnyThing.__init__` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| xǁSpinnyThingǁ__init____mutmut_2 | SpinnyThing.__init__ | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_SpinnyThing___init___char.py::test_init_chars_list_exact |
| xǁSpinnyThingǁ__init____mutmut_3 | SpinnyThing.__init__ | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_SpinnyThing___init___char.py::test_init_chars_list_exact |
| xǁSpinnyThingǁ__init____mutmut_4 | SpinnyThing.__init__ | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_SpinnyThing___init___char.py::test_init_chars_list_exact |
| xǁSpinnyThingǁ__init____mutmut_5 | SpinnyThing.__init__ | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_SpinnyThing___init___char.py::test_init_chars_list_exact |
| xǁSpinnyThingǁ__init____mutmut_7 | SpinnyThing.__init__ | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_SpinnyThing___init___char.py::test_init_index_zero |

## `SpinnyThing.increment` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| xǁSpinnyThingǁincrement__mutmut_1 | SpinnyThing.increment | equivalent | none | EQUIVALENT | Flips default arg value=1 -> value=2, but `value` is never read in the body (only self.index/self.chars used). No valid input makes the change observable. |
| xǁSpinnyThingǁincrement__mutmut_4 | SpinnyThing.increment | intentionally-unhelpful | pragma | BAD | pragma: no mutate @ Utilities.py:219. Changes stdout write '\b'+char -> 'XX\bXX'+char; cosmetic terminal spinner display only, no behavioral contract. |
| xǁSpinnyThingǁincrement__mutmut_7 | SpinnyThing.increment | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_SpinnyThing_increment_char.py::test_increment_advances_index_forward_by_one |
| xǁSpinnyThingǁincrement__mutmut_8 | SpinnyThing.increment | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_SpinnyThing_increment_char.py::test_increment_advances_index_forward_by_one |

## `SpinnyThing.finish` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| xǁSpinnyThingǁfinish__mutmut_2 | SpinnyThing.finish | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_SpinnyThing_finish_char.py::test_finish_writes_exact_payload |

## `state` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x_state__mutmut_16 | state | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_state_char.py::test_state_tuple_statekey_unpacks_key_and_attr |

## `state_key_ordering` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x_state_key_ordering__mutmut_7 | state_key_ordering | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_state_key_ordering_char.py::test_lt_is_strict_for_equal_objects |

## `hash_combine` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x_hash_combine__mutmut_3 | hash_combine | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_hash_combine_char.py::test_shift_kwarg_is_honored |
| x_hash_combine__mutmut_4 | hash_combine | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_hash_combine_char.py::test_shift_kwarg_is_honored |
| x_hash_combine__mutmut_6 | hash_combine | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_hash_combine_char.py::test_shift_value_must_be_the_passed_integer |
| x_hash_combine__mutmut_7 | hash_combine | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_hash_combine_char.py::test_shift_kwarg_lookup_uses_correct_key |
| x_hash_combine__mutmut_8 | hash_combine | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_hash_combine_char.py::test_shift_kwarg_lookup_uses_correct_key |
| x_hash_combine__mutmut_13 | hash_combine | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_hash_combine_char.py::test_empty_iterable_returns_zero |
| x_hash_combine__mutmut_14 | hash_combine | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_hash_combine_char.py::test_empty_iterable_returns_zero |

## `hash_objs` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x_hash_objs__mutmut_1 | hash_objs | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_hash_objs_char.py::test_hash_objs_varies_with_input |

## `isRhel8` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x_isRhel8__mutmut_2 | isRhel8 | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_isRhel8_char.py::test_isRhel8_reads_canonical_etc_os_release_path |
| x_isRhel8__mutmut_3 | isRhel8 | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_isRhel8_char.py::test_isRhel8_reads_canonical_etc_os_release_path |
| x_isRhel8__mutmut_4 | isRhel8 | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_isRhel8_char.py::test_isRhel8_reads_canonical_etc_os_release_path |
| x_isRhel8__mutmut_14 | isRhel8 | equivalent | none | EQUIVALENT | open(file, "r") -> open(file, ) drops the explicit mode, but "r" is open()'s default. Observable behavior identical for any input. |
| x_isRhel8__mutmut_25 | isRhel8 | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_isRhel8_char.py::test_isRhel8_emits_exact_warning_text_on_match |
| x_isRhel8__mutmut_26 | isRhel8 | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_isRhel8_char.py::test_isRhel8_emits_exact_warning_text_on_match |
| x_isRhel8__mutmut_27 | isRhel8 | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_isRhel8_char.py::test_isRhel8_emits_exact_warning_text_on_match |
| x_isRhel8__mutmut_28 | isRhel8 | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_isRhel8_char.py::test_isRhel8_emits_exact_warning_text_on_match |

## `ceilDivide` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x_ceilDivide__mutmut_1 | ceilDivide | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ceilDivide_char.py::test_negative_numerator_returns_zero |
| x_ceilDivide__mutmut_2 | ceilDivide | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ceilDivide_char.py::test_zero_numerator_does_not_take_negative_guard |
| x_ceilDivide__mutmut_3 | ceilDivide | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ceilDivide_char.py::test_zero_numerator_does_not_take_negative_guard |
| x_ceilDivide__mutmut_4 | ceilDivide | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ceilDivide_char.py::test_zero_denominator_takes_divide_by_zero_branch |
| x_ceilDivide__mutmut_5 | ceilDivide | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ceilDivide_char.py::test_zero_denominator_takes_divide_by_zero_branch |
| x_ceilDivide__mutmut_6 | ceilDivide | intentionally-unhelpful | pragma | BAD | pragma: no mutate @ Utilities.py:362. print(...) -> print(None) on negative-register error path; pure logging noise, same return (0). |
| x_ceilDivide__mutmut_7 | ceilDivide | intentionally-unhelpful | pragma | BAD | pragma: no mutate @ Utilities.py:362. Negative-register error string -> 'XX...XX'; cosmetic logging only. |
| x_ceilDivide__mutmut_8 | ceilDivide | intentionally-unhelpful | pragma | BAD | pragma: no mutate @ Utilities.py:362. Negative-register error string lowercased; cosmetic logging only. |
| x_ceilDivide__mutmut_9 | ceilDivide | intentionally-unhelpful | pragma | BAD | pragma: no mutate @ Utilities.py:362. Negative-register error string uppercased; cosmetic logging only. |
| x_ceilDivide__mutmut_13 | ceilDivide | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ceilDivide_char.py::test_floor_division_not_float_division |
| x_ceilDivide__mutmut_14 | ceilDivide | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_ceilDivide_char.py::test_minus_one_offset_in_ceiling_formula |
| x_ceilDivide__mutmut_17 | ceilDivide | intentionally-unhelpful | pragma | BAD | pragma: no mutate @ Utilities.py:367. print(...) -> print(None) on divide-by-zero error path; pure logging noise, same return (0). |
| x_ceilDivide__mutmut_18 | ceilDivide | intentionally-unhelpful | pragma | BAD | pragma: no mutate @ Utilities.py:367. Divide-by-zero error string -> 'XX...XX'; cosmetic logging only. |
| x_ceilDivide__mutmut_19 | ceilDivide | intentionally-unhelpful | pragma | BAD | pragma: no mutate @ Utilities.py:367. Divide-by-zero error string lowercased; cosmetic logging only. |
| x_ceilDivide__mutmut_20 | ceilDivide | intentionally-unhelpful | pragma | BAD | pragma: no mutate @ Utilities.py:367. Divide-by-zero error string uppercased; cosmetic logging only. |

## `choose_multiplier` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x_choose_multiplier__mutmut_7 | choose_multiplier | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_choose_multiplier_char.py::test_l_uses_base_2_log |
| x_choose_multiplier__mutmut_11 | choose_multiplier | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_choose_multiplier_char.py::test_float_division_loses_precision_for_large_operands |
| x_choose_multiplier__mutmut_12 | choose_multiplier | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_choose_multiplier_char.py::test_mlow_initial_value_and_loop_condition |
| x_choose_multiplier__mutmut_14 | choose_multiplier | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_choose_multiplier_char.py::test_mlow_initial_value_and_loop_condition |
| x_choose_multiplier__mutmut_26 | choose_multiplier | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_choose_multiplier_char.py::test_float_division_loses_precision_for_large_operands |
| x_choose_multiplier__mutmut_27 | choose_multiplier | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_choose_multiplier_char.py::test_mlow_initial_value_and_loop_condition |
| x_choose_multiplier__mutmut_28 | choose_multiplier | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_choose_multiplier_char.py::test_mlow_initial_value_and_loop_condition |
| x_choose_multiplier__mutmut_29 | choose_multiplier | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_choose_multiplier_char.py::test_mlow_initial_value_and_loop_condition |
| x_choose_multiplier__mutmut_33 | choose_multiplier | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_choose_multiplier_char.py::test_mlow_halving_inside_loop |
| x_choose_multiplier__mutmut_34 | choose_multiplier | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_choose_multiplier_char.py::test_float_division_loses_precision_for_large_operands |
| x_choose_multiplier__mutmut_35 | choose_multiplier | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_choose_multiplier_char.py::test_mlow_halving_inside_loop |

## `wmmaV3InputVgprLayout` (Tensile.Common.Utilities)

| mutant_id | function | bucket | action | verdict | test_node-or-justification |
|---|---|---|---|---|---|
| x_wmmaV3InputVgprLayout__mutmut_36 | wmmaV3InputVgprLayout | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_wmmaV3InputVgprLayout_char.py::test_unhandled_shape_raises_not_treated_as_128_branch |
| x_wmmaV3InputVgprLayout__mutmut_37 | wmmaV3InputVgprLayout | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_wmmaV3InputVgprLayout_char.py::test_32_16_128_1_int8_enters_branch_and_returns_layout |
| x_wmmaV3InputVgprLayout__mutmut_38 | wmmaV3InputVgprLayout | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_wmmaV3InputVgprLayout_char.py::test_32_16_128_1_int8_enters_branch_and_returns_layout |
| x_wmmaV3InputVgprLayout__mutmut_39 | wmmaV3InputVgprLayout | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_wmmaV3InputVgprLayout_char.py::test_32_16_128_1_int8_enters_branch_and_returns_layout |
| x_wmmaV3InputVgprLayout__mutmut_40 | wmmaV3InputVgprLayout | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_wmmaV3InputVgprLayout_char.py::test_32_16_128_1_int8_enters_branch_and_returns_layout |
| x_wmmaV3InputVgprLayout__mutmut_50 | wmmaV3InputVgprLayout | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_wmmaV3InputVgprLayout_char.py::test_unsupported_bitwidth_raises |
| x_wmmaV3InputVgprLayout__mutmut_51 | wmmaV3InputVgprLayout | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_wmmaV3InputVgprLayout_char.py::test_bitwidth_6_returns_packed_layout |
| x_wmmaV3InputVgprLayout__mutmut_56 | wmmaV3InputVgprLayout | missing-assertion-strength | add-test | KILLED | test_mut_Utilities_wmmaV3InputVgprLayout_char.py::test_unsupported_bitwidth_raises |

## Per-bucket counts

| bucket | count |
|---|---|
| missing-assertion-strength | 118 |
| intentionally-unhelpful (pragma) | 9 |
| equivalent | 4 |
| **total triaged** | **131** |
| untriaged (remaining of 131) | 0 |

### Disposition by verdict

| verdict | count |
|---|---|
| KILLED | 118 |
| BAD (pragma applied) | 9 |
| EQUIVALENT | 4 |
| — | 0 |

Pragma note: the 9 intentionally-unhelpful items collapse to 3 unique source lines (Utilities.py 219, 362, 367); 3 `# pragma: no mutate` markers applied, slice suite green (184 passed, 70 snapshots).

### Certified against a fresh mutmut re-run (authoritative)

After adding the tests + pragmas, a clean `mutmut run` reports: **654 mutants —
566 killed / 4 survived / 84 no-tests** (was 665 — 450 / 131 / 84). Survivors
131 → 4. All 4 remaining were independently verified genuinely **equivalent**
(`_cu_count_from_path` regex under `re.IGNORECASE`; `_validateWorkGroupMappingXCC`
default `-1→+1` where `1` is a degenerate-valid value; `SpinnyThing.increment`
unused `value` param; `isRhel8` `open(file,"r")→open(file)` default mode). Covered
mutation score 77.5% → 99.3% (566/570); 100% of non-equivalent covered mutants
killed. The 84 no-tests are coverage gaps (out of scope).
