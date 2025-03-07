function run_test()

length = load("-ascii", "in_len.txt");

batch = load("-ascii", "in_batch.txt");

pp_dim = load("-ascii", "in_pp_dim.txt");

pp_radices = load("-ascii", "in_pp_radices.txt");

fid = fopen("in_test_mode.txt", 'r');
test_mode = textscan(fid, '%s', 'delimiter', '\n');
test_mode = cellstr(test_mode);
fclose(fid);

in_batched = rocfft_input_data();

out_batched = rocfft_output_data();

partial_pass_3d(length, batch, pp_dim, pp_radices, in_batched, out_batched, test_mode);

delete('rocfft_input_data.m');
delete('rocfft_output_data.m');