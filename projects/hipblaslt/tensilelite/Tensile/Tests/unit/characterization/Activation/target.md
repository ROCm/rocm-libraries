# Activation.py — characterization target (PARTIAL by scope)

Pins the pure configuration/type/numeric layer of Activation.py; the rocisa
assembly codegen (the bulk) is out of scope per the codegen/asm exclusion and
mostly raises in this environment. See DECISIONS D13.

Coverage: 1037 stmts, 683 missed → 34.1% line (was 16.8%).

Pinned: ActivationAvailable, ActivationTypeRegister.typeAvailable, full
ActivationType API, actCacheInfo.isSame, getMagic/getMagicStr/HexToStr/addSpace,
ActivationModule defaults/setters/counters/vgprPrefix + working getModule paths
(abs/relu/none/clippedrelu/leakyrelu/clamp/drelu) + getAllGprUsage(single type).

Resistance (asm codegen, not pinned): getExp/getGelu/getSigmoid/getTanh/
getDGelu/getSilu/getSwish/getGeluScaling emitters (raise NameError 'SelectBit'/
'VMaxF16' or KeyError 'TransOpWait' without a full ISA/KernelWriter context),
CombineInstructions/FuseInstruction + iter helpers, replaceInst/removeOldInst,
ConvertCoeffToHex/HolderToGpr/createVgprIdxList, ActivationInline.
