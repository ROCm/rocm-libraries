# Round-boundary track arms (ARM_FILES=/job/stages/rb-h-<n>/RB/arms_rb.sh): ATOM_Q38_RB_DEVMETA=1 (round t+1's
# accept-dependent metadata on the device; ATOM-truemono truemono/rb_devmeta.py) on the consolidated stack arms.
RB_REJECT='rb devmeta: off|rb devmeta check step [0-9]+: DIFF'
_rb_on() { ARM_ENV="$ARM_ENV ATOM_Q38_RB_DEVMETA=1${1:+ $1}"; ARM_REJECT="${ARM_REJECT:+$ARM_REJECT|}$RB_REJECT"; }
arm_full3_rb() { arm_full3; _rb_on; }
arm_full3_l_rb() { arm_full3_l; _rb_on; }
arm_full3_d_rb() { arm_full3_d; _rb_on; }
arm_hyb_tm_rb() { arm_hyb_tm; _rb_on; }
arm_finalHqd_rb() { arm_finalHqd; _rb_on; }
arm_full0_rb() { arm_full0; _rb_on; }
arm_full3_l_rb1() { arm_full3_l; _rb_on ATOM_Q38_RB_DEVMETA_FUSE=0; }     # v1: shift kernel + ATOM's re-derivations
arm_full3_rb1() { arm_full3; _rb_on ATOM_Q38_RB_DEVMETA_FUSE=0; }
# decision-15 re-tests on the rb base (levers that lost / tied while the boundary went through the host)
arm_full3_l_rb_hd8() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_DRAFT=lean ATOM_Q38_TM_TAIL_P9={\"hdelay\":8}'; _rb_on; }
arm_full3_l_rb_hd16() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_DRAFT=lean ATOM_Q38_TM_TAIL_P9={\"hdelay\":16}'; _rb_on; }
arm_full3_d_rb_hd8() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_DRAFT=p9 ATOM_Q38_TM_TAIL_P9D={\"hdelay\":8}'; _rb_on; }
arm_full3_l_rb_n8_64k() { _cs_full3 tunedHqd 65536 ATOM_Q38_TM_TAIL_DRAFT=lean; _rb_on; }
arm_full3_l_rb_dg3() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_DRAFT=lean ATOM_Q38_TM_TAIL_P9={\"head_dg\":3}'; _rb_on; }
# + ATOM_Q38_RB_DEVMETA_TRIM=1: skip the repeated graph-padding fills / copies and ATOM's own compressed-slot launch
arm_full3_l_rbt() { arm_full3_l; _rb_on ATOM_Q38_RB_DEVMETA_TRIM=1; }
arm_full3_rbt() { arm_full3; _rb_on ATOM_Q38_RB_DEVMETA_TRIM=1; }
arm_full3_d_rbt() { arm_full3_d; _rb_on ATOM_Q38_RB_DEVMETA_TRIM=1; }
# + ATOM_Q38_RB_DEVMETA_ADV=1 (with TRIM): steady-state steps skip the accept-dependent uploads, advanced on the device
arm_full3_l_rba() { arm_full3_l; _rb_on 'ATOM_Q38_RB_DEVMETA_TRIM=1 ATOM_Q38_RB_DEVMETA_ADV=1'; }
arm_full3_rba() { arm_full3; _rb_on 'ATOM_Q38_RB_DEVMETA_TRIM=1 ATOM_Q38_RB_DEVMETA_ADV=1'; }
arm_full3_l_rba2() { arm_full3_l; _rb_on 'ATOM_Q38_RB_DEVMETA_TRIM=1 ATOM_Q38_RB_DEVMETA_ADV=2'; }
arm_full3_rba2() { arm_full3; _rb_on 'ATOM_Q38_RB_DEVMETA_TRIM=1 ATOM_Q38_RB_DEVMETA_ADV=2'; }
# decision-15 re-tests on base full3_l_rba (after _TRIM and _ADV were accepted)
RBA_ENV='ATOM_Q38_RB_DEVMETA_TRIM=1 ATOM_Q38_RB_DEVMETA_ADV=1'
arm_full3_l_rba_hd8() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_DRAFT=lean ATOM_Q38_TM_TAIL_P9={\"hdelay\":8}'; _rb_on "$RBA_ENV"; }
arm_full3_l_rba_hd16() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_DRAFT=lean ATOM_Q38_TM_TAIL_P9={\"hdelay\":16}'; _rb_on "$RBA_ENV"; }
arm_full3_l_rba_dg3() { _cs_full3 tunedHqd 98304 'ATOM_Q38_TM_TAIL_DRAFT=lean ATOM_Q38_TM_TAIL_P9={\"head_dg\":3}'; _rb_on "$RBA_ENV"; }
arm_full3_l_rba_n8_64k() { _cs_full3 tunedHqd 65536 ATOM_Q38_TM_TAIL_DRAFT=lean; _rb_on "$RBA_ENV"; }
arm_full3_d_rba() { arm_full3_d; _rb_on "$RBA_ENV"; }
arm_full3_l_rba1() { arm_full3_l; _rb_on "$RBA_ENV ATOM_Q38_RB_DEVMETA_FUSE=0"; }
