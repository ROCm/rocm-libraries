# N8 (DESIGN 12.3): the drafter's argmax head over a frequency-ranked vocab subset (ATOM-truemono draft_vocab.py,
# ATOM_Q38_DRAFT_VOCAB). Ranking: n8/ranking.pt from qk-n8rank (hybrid replies on gsm8k-train / MATH-500 / MBPP /
# MT-Bench prompts, disjoint from the gsm8k test prompts the quick check times). On top of the candidate base hyb_tm.
N8_RANK='$H/truemono-quick/n8/ranking.pt'
arm_n8_32k() { arm_hyb_tm; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=32768"; ARM_EXPECT='draft vocab subset on: 32768 of'; }
arm_n8_48k() { arm_hyb_tm; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=49152"; ARM_EXPECT='draft vocab subset on: 49152 of'; }
arm_n8_64k() { arm_hyb_tm; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=65536"; ARM_EXPECT='draft vocab subset on: 65536 of'; }
arm_n8_16k() { arm_hyb_tm; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=16384"; ARM_EXPECT='draft vocab subset on: 16384 of'; }
arm_n8_96k() { arm_hyb_tm; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=98304"; ARM_EXPECT='draft vocab subset on: 98304 of'; }
arm_n8_128k() { arm_hyb_tm; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=131072"; ARM_EXPECT='draft vocab subset on: 131072 of'; }
# diagnostics only (an extra full-head launch per draft step: its timing is not a candidate): the share of draft
# rows whose full-head token is in the 64K subset, and of rows whose subset token equals it (server log)
arm_n8_audit64() { arm_n8_64k; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB_AUDIT=500"; ARM_EXPECT='draft vocab audit:'; }
# N8 96K + the subset-head pick on the candidate base, no lookback picks (RULER bisection of the final stack)
arm_n8hd_96k() { arm_hyb_tm; ARM_ENV="$ARM_ENV ATOM_Q38_DRAFT_VOCAB=$N8_RANK ATOM_Q38_DRAFT_VOCAB_N=98304 ATOM_Q38_DRAFT_VOCAB_SPEC=$N8_HEAD_SPEC"; ARM_EXPECT='draft vocab subset on: 98304 of .*n98304_k2560_r4_bf16_fused_kolane'; }
