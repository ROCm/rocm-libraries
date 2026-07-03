Name	op_type	dtype	input_shape	output_shape	kh×kw	ops
encoder_0	Concat	int8 → int8	2 × (1,4,2160,3840)	(1,8,2160,3840)
encoder_0	Conv	int8 → int32	(1,8,2160,3840)	(1,32,2160,3840)	3 × 3	38.2 GOP
encoder_0	QuantizeLinear	int32 → int8	(1,32,2160,3840)	(1,32,2160,3840)
encoder_0	Relu	int8 → int8	(1,32,2160,3840)	(1,32,2160,3840)
encoder_0	QuantizeLinear	int8 → int4	(1,32,2160,3840)	(1,32,2160,3840)
encoder_0	Conv	int4 → int32	(1,32,2160,3840)	(1,24,2160,3840)	1 × 1	12.7 GOP
encoder_0	QuantizeLinear	int32 → int4	(1,24,2160,3840)	(1,24,2160,3840)
encoder_0	Relu	int4 → int4	(1,24,2160,3840)	(1,24,2160,3840)
encoder_0	MaxPool	int4 → int4	(1,24,2160,3840)	(1,24,1080,1920)	2 × 2
encoder_0	QuantizeLinear	int4 → int4	(1,24,1080,1920)	(1,24,1080,1920)
	Fusion	int8 → int4	2 × (1,4,2160,3840)	(1,24,1080,1920)		51.0 GOP
