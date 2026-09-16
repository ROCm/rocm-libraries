import type { OpCatalogEntry, ParamSpec, ParamValue } from "./model";

/**
 * The operator catalog: the palette of node kinds a user can place. Kept as
 * plain data so it is trivial to extend or eventually load from the hipDNN
 * op registry.
 */

/**
 * Stride orders offered for input/output tensors. Dimensions are always given
 * in hipDNN's canonical order (N,C,W / N,C,H,W / N,C,D,H,W); the layout only
 * decides the strides, i.e. which axis varies fastest. Each name lists axes
 * from slowest- to fastest-varying, so NHWC is channels-last over NCHW dims.
 * PACKED_ROW_MAJOR is rank-agnostic and CUSTOM takes explicit strides.
 */
export const STRIDE_LAYOUTS = [
  "PACKED_ROW_MAJOR",
  "NCW",
  "NWC",
  "NCHW",
  "NHWC",
  "CHWN",
  "NCDHW",
  "NDHWC",
  "CDHWN",
  "CUSTOM",
] as const;

const LAYOUT_PARAM: ParamSpec = {
  key: "layout",
  label: "Stride layout",
  type: "enum",
  default: "PACKED_ROW_MAJOR",
  options: STRIDE_LAYOUTS,
};

export const OP_CATALOG: readonly OpCatalogEntry[] = [
  {
    type: "Input",
    label: "Input Tensor",
    category: "IO",
    accent: "#3b82f6",
    inputs: [],
    outputs: [{ id: "out", label: "out" }],
    params: [
      { key: "shape", label: "Shape", type: "string", default: "1,3,224,224" },
      LAYOUT_PARAM,
      {
        key: "strides",
        label: "Strides",
        type: "string",
        default: "150528,50176,224,1",
        visibleWhen: [{ key: "layout", equals: "CUSTOM" }],
      },
      {
        key: "dtype",
        label: "dtype",
        type: "enum",
        default: "float32",
        options: ["float32", "float16", "bfloat16", "int8"],
      },
    ],
  },
  {
    type: "ConvolutionFprop",
    label: "Convolution Fprop",
    category: "Convolution",
    accent: "#8b5cf6",
    inputs: [
      { id: "x", label: "x" },
      { id: "w", label: "w" },
    ],
    outputs: [{ id: "y", label: "y" }],
    params: [
      { key: "padding", label: "Padding", type: "int", default: 1 },
      { key: "stride", label: "Stride", type: "int", default: 1 },
      { key: "dilation", label: "Dilation", type: "int", default: 1 },
      {
        key: "conv_mode",
        label: "Mode",
        type: "enum",
        default: "CROSS_CORRELATION",
        options: ["CROSS_CORRELATION", "CONVOLUTION"],
      },
    ],
  },
  {
    type: "ConvolutionDgrad",
    label: "Convolution Dgrad",
    category: "Convolution",
    accent: "#7c3aed",
    inputs: [
      { id: "dy", label: "dy" },
      { id: "w", label: "w" },
    ],
    outputs: [{ id: "dx", label: "dx" }],
    params: [
      // dx shape can't be deduced from dy+w alone; give the forward input shape.
      { key: "dx_shape", label: "dx shape", type: "string", default: "1,3,224,224" },
      { key: "padding", label: "Padding", type: "int", default: 1 },
      { key: "stride", label: "Stride", type: "int", default: 1 },
      { key: "dilation", label: "Dilation", type: "int", default: 1 },
      {
        key: "conv_mode",
        label: "Mode",
        type: "enum",
        default: "CROSS_CORRELATION",
        options: ["CROSS_CORRELATION", "CONVOLUTION"],
      },
    ],
  },
  {
    type: "ConvolutionWgrad",
    label: "Convolution Wgrad",
    category: "Convolution",
    accent: "#6d28d9",
    inputs: [
      { id: "dy", label: "dy" },
      { id: "x", label: "x" },
    ],
    outputs: [{ id: "dw", label: "dw" }],
    params: [
      // dw (filter) shape can't be deduced; give [outC,inC,kh,kw].
      { key: "dw_shape", label: "dw shape", type: "string", default: "64,3,3,3" },
      { key: "padding", label: "Padding", type: "int", default: 1 },
      { key: "stride", label: "Stride", type: "int", default: 1 },
      { key: "dilation", label: "Dilation", type: "int", default: 1 },
      {
        key: "conv_mode",
        label: "Mode",
        type: "enum",
        default: "CROSS_CORRELATION",
        options: ["CROSS_CORRELATION", "CONVOLUTION"],
      },
    ],
  },
  {
    type: "MatMul",
    label: "MatMul",
    category: "Linear",
    accent: "#a855f7",
    inputs: [
      { id: "a", label: "a" },
      { id: "b", label: "b" },
    ],
    outputs: [{ id: "y", label: "y" }],
    params: [],
  },
  {
    type: "BatchNormInference",
    label: "BatchNorm (Inference)",
    category: "Normalization",
    accent: "#06b6d4",
    inputs: [
      { id: "x", label: "x" },
      { id: "mean", label: "mean" },
      { id: "inv_variance", label: "inv_var" },
      { id: "scale", label: "scale" },
      { id: "bias", label: "bias" },
    ],
    outputs: [{ id: "y", label: "y" }],
    params: [],
  },
  {
    type: "BatchNormBackward",
    label: "BatchNorm (Backward)",
    category: "Normalization",
    accent: "#0891b2",
    inputs: [
      { id: "dy", label: "dy" },
      { id: "x", label: "x" },
      { id: "scale", label: "scale" },
    ],
    outputs: [
      { id: "dx", label: "dx" },
      { id: "dscale", label: "dscale" },
      { id: "dbias", label: "dbias" },
    ],
    params: [],
  },
  {
    type: "Layernorm",
    label: "LayerNorm (Forward)",
    category: "Normalization",
    accent: "#14b8a6",
    inputs: [
      { id: "x", label: "x" },
      { id: "scale", label: "scale" },
      { id: "bias", label: "bias" },
    ],
    outputs: [
      { id: "y", label: "y" },
      { id: "mean", label: "mean" },
      { id: "inv_variance", label: "inv_var" },
    ],
    params: [
      { key: "epsilon", label: "Epsilon", type: "float", default: 1e-5 },
      {
        key: "forward_phase",
        label: "Phase",
        type: "enum",
        default: "training",
        options: ["training", "inference"],
      },
    ],
  },
  {
    type: "LayernormBackward",
    label: "LayerNorm (Backward)",
    category: "Normalization",
    accent: "#0d9488",
    inputs: [
      { id: "dy", label: "dy" },
      { id: "x", label: "x" },
      { id: "scale", label: "scale" },
      { id: "mean", label: "mean" },
      { id: "inv_variance", label: "inv_var" },
    ],
    outputs: [
      { id: "dx", label: "dx" },
      { id: "dscale", label: "dscale" },
      { id: "dbias", label: "dbias" },
    ],
    params: [],
  },
  {
    type: "RMSNorm",
    label: "RMSNorm (Forward)",
    category: "Normalization",
    accent: "#2dd4bf",
    inputs: [
      { id: "x", label: "x" },
      { id: "scale", label: "scale" },
      { id: "bias", label: "bias" },
    ],
    outputs: [
      { id: "y", label: "y" },
      { id: "inv_rms", label: "inv_rms" },
    ],
    params: [
      { key: "epsilon", label: "Epsilon", type: "float", default: 1e-5 },
      {
        key: "forward_phase",
        label: "Phase",
        type: "enum",
        default: "training",
        options: ["training", "inference"],
      },
    ],
  },
  {
    type: "RMSNormBackward",
    label: "RMSNorm (Backward)",
    category: "Normalization",
    accent: "#0f766e",
    inputs: [
      { id: "dy", label: "dy" },
      { id: "x", label: "x" },
      { id: "scale", label: "scale" },
      { id: "inv_rms", label: "inv_rms" },
    ],
    outputs: [
      { id: "dx", label: "dx" },
      { id: "dscale", label: "dscale" },
      { id: "dbias", label: "dbias" },
    ],
    params: [{ key: "compute_dbias", label: "Compute dbias", type: "bool", default: true }],
  },
  {
    type: "Pointwise",
    label: "Pointwise (Unary)",
    category: "Activation",
    accent: "#22c55e",
    inputs: [{ id: "x", label: "x" }],
    outputs: [{ id: "y", label: "y" }],
    params: [
      {
        key: "mode",
        label: "Mode",
        type: "enum",
        default: "relu_fwd",
        options: [
          "relu_fwd",
          "sigmoid_fwd",
          "tanh_fwd",
          "gelu_fwd",
          "exp",
          "log",
          "abs",
          "neg",
          "sqrt",
          "rsqrt",
          "reciprocal",
        ],
      },
    ],
  },
  {
    type: "PointwiseBinary",
    label: "Pointwise (Binary)",
    category: "Elementwise",
    accent: "#f97316",
    inputs: [
      { id: "a", label: "a" },
      { id: "b", label: "b" },
    ],
    outputs: [{ id: "y", label: "y" }],
    params: [
      {
        key: "mode",
        label: "Mode",
        type: "enum",
        default: "add",
        options: ["add", "mul", "sub", "div", "max_op", "min_op"],
      },
    ],
  },
  {
    type: "Reduction",
    label: "Reduction",
    category: "Elementwise",
    accent: "#d946ef",
    inputs: [{ id: "x", label: "x" }],
    outputs: [{ id: "y", label: "y" }],
    params: [
      {
        key: "mode",
        label: "Mode",
        type: "enum",
        default: "add",
        options: ["add", "mul", "min", "max", "amax", "avg", "norm1", "norm2"],
      },
      // Output shape can't be deduced (axes are reduced); give the reduced dims.
      { key: "out_shape", label: "Output shape", type: "string", default: "2,3,4,1" },
    ],
  },
  {
    type: "ResampleFwd",
    label: "Resample (Forward)",
    category: "Pooling",
    accent: "#eab308",
    inputs: [{ id: "x", label: "x" }],
    outputs: [{ id: "y", label: "y" }],
    params: [
      { key: "window", label: "Window", type: "int", default: 2 },
      { key: "stride", label: "Stride", type: "int", default: 2 },
      { key: "padding", label: "Padding", type: "int", default: 0 },
      {
        key: "resample_mode",
        label: "Mode",
        type: "enum",
        default: "MAXPOOL",
        options: ["MAXPOOL", "AVGPOOL_EXCLUDE_PADDING", "AVGPOOL_INCLUDE_PADDING", "BILINEAR"],
      },
      {
        key: "padding_mode",
        label: "Padding mode",
        type: "enum",
        default: "NEG_INF_PAD",
        options: ["NEG_INF_PAD", "ZERO_PAD"],
      },
    ],
  },
  {
    type: "BlockScaleQuantize",
    label: "Block Scale Quantize",
    category: "Quantization",
    accent: "#ec4899",
    inputs: [{ id: "x", label: "x" }],
    outputs: [
      { id: "y", label: "y" },
      { id: "scale", label: "scale" },
    ],
    params: [
      { key: "block_size", label: "Block size", type: "int", default: 32 },
      { key: "axis", label: "Axis", type: "int", default: 1 },
      { key: "transpose", label: "Transpose", type: "bool", default: false },
    ],
  },
  {
    type: "BlockScaleDequantize",
    label: "Block Scale Dequantize",
    category: "Quantization",
    accent: "#db2777",
    inputs: [
      { id: "x", label: "x" },
      { id: "scale", label: "scale" },
    ],
    outputs: [{ id: "y", label: "y" }],
    params: [
      { key: "block_size", label: "Block size", type: "int", default: 32 },
      { key: "is_negative_scale", label: "Negative scale", type: "bool", default: false },
    ],
  },
  {
    type: "Output",
    label: "Output Tensor",
    category: "IO",
    accent: "#ef4444",
    inputs: [{ id: "in", label: "in" }],
    outputs: [],
    params: [
      { key: "use_defaults", label: "Use defaults (inferred shape)", type: "bool", default: true },
      {
        key: "shape",
        label: "Shape",
        type: "string",
        default: "1,3,224,224",
        visibleWhen: [{ key: "use_defaults", equals: false }],
      },
      { ...LAYOUT_PARAM, visibleWhen: [{ key: "use_defaults", equals: false }] },
      {
        key: "strides",
        label: "Strides",
        type: "string",
        default: "150528,50176,224,1",
        visibleWhen: [
          { key: "use_defaults", equals: false },
          { key: "layout", equals: "CUSTOM" },
        ],
      },
    ],
  },
];

const CATALOG_BY_TYPE: Record<string, OpCatalogEntry> = Object.fromEntries(
  OP_CATALOG.map((entry) => [entry.type, entry]),
);

export function catalogEntry(type: string): OpCatalogEntry | undefined {
  return CATALOG_BY_TYPE[type];
}

export function defaultParams(type: string): Record<string, ParamValue> {
  const entry = CATALOG_BY_TYPE[type];
  if (!entry) return {};
  const params: Record<string, ParamValue> = {};
  for (const spec of entry.params) params[spec.key] = spec.default;
  return params;
}

/**
 * The params of `entry` that apply to a node currently holding `params`, i.e.
 * those whose `visibleWhen` conditions (if any) all hold. A param the node has
 * never been given falls back to its catalog default, so graphs saved before a
 * param existed still resolve.
 */
export function visibleParams(
  entry: OpCatalogEntry,
  params: Record<string, ParamValue>,
): readonly ParamSpec[] {
  if (!entry.params.some((spec) => spec.visibleWhen)) return entry.params;
  return entry.params.filter(
    (spec) =>
      spec.visibleWhen?.every((cond) => {
        const other = entry.params.find((s) => s.key === cond.key);
        return (params[cond.key] ?? other?.default) === cond.equals;
      }) ?? true,
  );
}
