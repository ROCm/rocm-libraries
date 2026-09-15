# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _mx_generation_types import *  # noqa: F403


def expected_mx(case):
    rows, columns = case.shape
    leading_dimension = case.leading_dimension or rows
    format_spec = DATA_FORMATS[case.data_type]
    blocked_extent = case.shape[case.block_axis]
    free_extent = case.shape[1 - case.block_axis]
    block_count = (blocked_extent + case.block_size - 1) // case.block_size
    scale_count = block_count * free_extent
    fixed_scale_raw = explicit_scale_raw(case.scale_type, case.scale)
    scale_candidates = (
        []
        if fixed_scale_raw is not None
        else finite_nonzero_scale_candidates(case.scale_type)
    )
    decoded_data = [
        decode_binary(format_spec, raw) for raw in range(1 << format_spec.storage_bits)
    ]
    finite_data_candidates = finite_data_raw_candidates(format_spec)
    maximum_data_value = finite_binary_value(
        format_spec, format_spec.maximum_finite_raw
    )

    physical_raw = [0] * (leading_dimension * columns)
    scale_raw = [0] * scale_count
    scale_indices = np.empty((rows, columns), dtype=np.uint32)
    reference = np.empty((rows, columns), dtype=np.float32)

    for scale_index in range(scale_count):
        if case.block_axis == 0:
            block = scale_index % block_count
            free_coordinate = scale_index // block_count
        else:
            block = scale_index // free_extent
            free_coordinate = scale_index % free_extent
        coordinates = block_coordinates(case, block, free_coordinate)

        if case.data.kind is RecipeKind.UNBOUNDED:
            if fixed_scale_raw is None:
                candidate_index = counter_random(
                    case.seed, MX_UNBOUNDED_SCALE_RANDOM_DOMAIN, scale_index
                ) % len(scale_candidates)
                selected_scale_raw = scale_candidates[candidate_index][1]
            else:
                selected_scale_raw = fixed_scale_raw
            scale_raw[scale_index] = selected_scale_raw
            scale_value = decode_scale(case.scale_type, selected_scale_raw)

            for row, column in coordinates:
                logical_index = row + column * rows
                physical_index = row + column * leading_dimension
                candidate_index = counter_random(
                    case.seed, GENERATION_REAL_RANDOM_DOMAIN, logical_index
                ) % len(finite_data_candidates)
                data_raw = finite_data_candidates[candidate_index]
                physical_raw[physical_index] = data_raw
                scale_indices[row, column] = scale_index
                with np.errstate(over="ignore", invalid="ignore"):
                    reference[row, column] = np.float32(
                        decoded_data[data_raw] * scale_value
                    )
            continue

        block_values = [
            generated_value(case, row, column, row + column * rows)
            for row, column in coordinates
        ]
        if fixed_scale_raw is not None:
            selected_scale_raw = fixed_scale_raw
        elif any(math.isnan(value) for value in block_values):
            selected_scale_raw = encode_scale(case.scale_type, math.nan)
        else:
            maximum_magnitude = max(
                (abs(value) for value in block_values if math.isfinite(value)),
                default=0.0,
            )
            if maximum_magnitude == 0.0:
                selected_scale_raw = encode_scale(case.scale_type, 1.0)
            else:
                selected_scale_raw = scale_at_least(
                    maximum_magnitude / maximum_data_value,
                    scale_candidates,
                    case.data.kind
                    in (RecipeKind.BOUNDED, RecipeKind.BOUNDED_ALTERNATING_SIGN),
                    case.seed,
                    scale_index,
                )

        scale_raw[scale_index] = selected_scale_raw
        scale_value = decode_scale(case.scale_type, selected_scale_raw)
        for (row, column), source_value in zip(coordinates, block_values):
            logical_index = row + column * rows
            physical_index = row + column * leading_dimension
            scaled_value = (
                source_value if source_value == 0.0 else source_value / scale_value
            )
            data_raw = encode_binary(format_spec, scaled_value)
            if case.data.kind is RecipeKind.BOUNDED:
                data_raw = constrain_raw_to_interval(
                    format_spec,
                    data_raw,
                    scale_value,
                    case.data.lower,
                    case.data.upper,
                    decoded_data,
                )
            elif case.data.kind is RecipeKind.BOUNDED_ALTERNATING_SIGN:
                maximum = case.data.maximum_magnitude
                data_raw = constrain_raw_to_interval(
                    format_spec,
                    data_raw,
                    scale_value,
                    -maximum,
                    maximum,
                    decoded_data,
                )

            physical_raw[physical_index] = data_raw
            scale_indices[row, column] = scale_index
            with np.errstate(over="ignore", invalid="ignore"):
                reference[row, column] = np.float32(
                    decoded_data[data_raw] * scale_value
                )

    return ExpectedMx(
        data_storage=pack_little_endian(physical_raw, format_spec.storage_bits),
        scale_storage=bytes(scale_raw),
        scale_indices=scale_indices,
        reference=reference,
    )


def make_problem(case):
    data, quantization, represented_range, scale_override = case.data.native(
        case.shape, case.seed
    )
    if quantization == hv.MxDataQuantization.PreserveRange:
        data_generation = hv.MxDataGeneration.preserve_range(data, represented_range)
    elif quantization == hv.MxDataQuantization.PreserveGeneratedEncoding:
        data_generation = hv.MxDataGeneration.preserve_generated_encoding(data)
    else:
        data_generation = hv.MxDataGeneration.quantize(data)
    options = hv.MxGenerationOptions()
    options.data_type = case.data_type
    options.scale_type = case.scale_type
    options.leading_dimension = case.leading_dimension
    options.block_axis = case.block_axis
    options.block_size = case.block_size
    options.scale = case.scale if scale_override is None else scale_override
    return hv.Shape(list(case.shape)), data_generation, options


__all__ = [name for name in globals() if not name.startswith("__")]
