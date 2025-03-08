function partial_pass_3d(in_length, nbatch, pp_dim, pp_radices, in_batched, out_batched, test_mode)

  test_mode_1 = 'full-3d';
  test_mode_2 = 'direction_1';
  test_mode_3 = 'direction_1_step_1_2';
  test_mode_4 = 'direction_1_step_1_2_3_4';

  if ~(strcmp(test_mode,test_mode_1) || strcmp(test_mode,test_mode_2) || ...
       strcmp(test_mode,test_mode_3) || strcmp(test_mode,test_mode_4))
      display(test_mode);
    error('Invalid test mode');
  endif

  format longG;
  ordering='column-major';
  data_empty_value = -123456789;

  N = prod(in_length);

  pp_mode = 'four-step';

  in_batched = convert_1d_to_3d(in_batched, in_length(1), in_length(2), in_length(3), nbatch, ordering);
  out_batched = convert_1d_to_3d(out_batched, in_length(1), in_length(2), in_length(3), nbatch, ordering);

  for ibatch=1:nbatch
    in = in_batched(:,:,:,ibatch);
    out_ = out_batched(:,:,:,ibatch);

    % Validate output
    idx_data=find(real(out_)~=data_empty_value);
    if ( length(idx_data) != N )
      error('Error: incomplete data');
    endif

    % 3D-FFT (MATLAB built-in)
    out = fftn(in);
    out = convert_3d_to_1d(out, ordering);
    out_ = convert_3d_to_1d(out_, ordering);

    if (strcmp(test_mode,test_mode_1))
      linf_rocfft_vs_octave_built_in = norm(out-out_,'inf');
      disp(['l-inf norm: '  num2str(linf_rocfft_vs_octave_built_in)]);
    else
      % CS_3D_RC from rocFFT (with partial pass)
      [out_3d_rc, out_3d_rc_1, out_3d_rc_pp_1, out_3d_rc_pp_2] = run_CS_3D_RC(in_length, in, pp_dim, pp_radices, pp_mode);

      out_3d_rc = convert_3d_to_1d(out_3d_rc, ordering);
      linf_test = norm(out_3d_rc-out,'inf');
      if (linf_test > 1E-8)
        error("Error: partial-pass 3D-RC failed accuracy test");
      endif

      if (strcmp(test_mode,test_mode_2))
        out_3d_rc_1 = convert_3d_to_1d(out_3d_rc_1, ordering);
        linf_test = norm(out_3d_rc_1-out_,'inf');
        disp(['l-inf norm: '  num2str(linf_test)]);
      endif

      if (strcmp(test_mode,test_mode_3))
        out_3d_rc_pp_1 = convert_3d_to_1d(out_3d_rc_pp_1, ordering);
        linf_test = norm(out_3d_rc_pp_1-out_,'inf');
        disp(['l-inf norm: '  num2str(linf_test)]);
      endif

      if (strcmp(test_mode,test_mode_4))
        out_3d_rc_pp_2 = convert_3d_to_1d(out_3d_rc_pp_2, ordering);
        linf_test = norm(out_3d_rc_pp_2-out_,'inf');
        disp(['l-inf norm: '  num2str(linf_test)]);
      endif
    endif
  endfor

  function [out, out_1, out_pp_1, out_pp_2] = run_CS_3D_RC(in_length, in, pp_dim, pp_radices, pp_mode)
    n = in_length(pp_dim);

    % Flip radices, as the radix order is reversed in steps 1-2 and 3-4
    pp_radices = flip(pp_radices);
    
    n1 = pp_radices(1);
    n2 = pp_radices(2);

    F_n = dft_matrix(n);
    F_n1 = dft_matrix(n1);
    F_n2 = dft_matrix(n2);

    out = in;

    if (pp_dim == 1)
      % 1st kernel (2nd dimension)
      out = fft(out,[], 2);
      out_1 = out;
      out = partial_pass_step_1_2(out, 1, n1, n2, F_n1, F_n2, F_n, pp_mode);
      out_pp_1 = out;

      % 2nd kernel (3rd dimension)
      out = partial_pass_step_3_4(out, 1, n1, n2, F_n1, F_n2, F_n, pp_mode);
      out_pp_2 = out;
      out = fft(out,[], 3);
    endif

    if (pp_dim == 2)
      % Correct ordering for intermediate results comparison
      transp_order_comp = [3 1 2];

      % 1st kernel (1st dimension)
      out = fft(out,[], 1);
      out_1 = permute(out, transp_order_comp);
      out = partial_pass_step_1_2(out, 2, n1, n2, F_n1, F_n2, F_n, pp_mode);
      out_pp_1 = permute(out, transp_order_comp); 

      % 2nd kernel (3rd dimension)
      transp_order = [3 2 1];
      out = permute(out, transp_order);

      out = partial_pass_step_3_4(out, 2, n1, n2, F_n1, F_n2, F_n, pp_mode);
      out_pp_2 = out;

      out = fft(out,[], 1);

      out = permute(out, transp_order);
    endif

    if (pp_dim == 3)
      % 1st kernel (1st dimension)
      out = fft(out,[], 1);
      out_1 = out;
      out = partial_pass_step_1_2(out, 3, n1, n2, F_n1, F_n2, F_n, pp_mode);
      out_pp_1 = out;

      % 2nd kernel (2nd dimension)
      out = partial_pass_step_3_4(out, 3, n1, n2, F_n1, F_n2, F_n, pp_mode);
      out_pp_2 = out;
      out = fft(out,[], 2);
    endif
  endfunction

  function [dim1, dim2] = get_data_dim_partial_pass(input, pp_dim)
    if (pp_dim==1)
      dim1 = size(input,2);
      dim2 = size(input,3);
    elseif (pp_dim==2)
      dim1 = size(input,1);
      dim2 = size(input,3);
    elseif (pp_dim==3)
      dim1 = size(input,1);
      dim2 = size(input,2);
    endif
  endfunction

  function input_data_decomp = get_pp_decomposed_data(input_data, pp_dim, idx1, idx2, n1, n2)
    if (pp_dim==1)
      input_data_decomp = reshape(input_data(:,idx1,idx2), n1, n2);
    elseif (pp_dim==2)
      input_data_decomp = reshape(input_data(idx1,:,idx2), n1, n2);
    elseif (pp_dim==3)
      input_data_decomp = reshape(input_data(idx1,idx2,:), n1, n2);
    endif
  endfunction

  function output = set_pp_data(input, input_decomp, pp_dim, idx1, idx2)
    output = input;

    if (pp_dim==1)
      output(:,idx1,idx2) = reshape(input_decomp, [], 1);
    elseif (pp_dim==2)
      output(idx1,:,idx2) = reshape(input_decomp, [], 1);
    elseif (pp_dim==3)
      output(idx1,idx2,:) = reshape(input_decomp, [], 1);
    endif
  endfunction

  function output = partial_pass_step_1_2(input, pp_dim, n1, n2, F_n1, F_n2, F_n, mode)
    output = input;

    [dim1, dim2] = get_data_dim_partial_pass(input, pp_dim);

    for idx2=1:dim2
      for idx1=1:dim1
        in_decomp = get_pp_decomposed_data(output, pp_dim, idx1, idx2, n1, n2);

        if strcmp(mode, 'four-step')
          % Length-n2 FFT along rows of in_decomp
          out_decomp = fft(in_decomp, n2, 2);
          % Twiddle multiply
          out_decomp = F_n(1:n1, 1:n2).*out_decomp;
        elseif strcmp(mode, 'six-step')
          % Local transpose
          out_decomp = in_decomp.';
          % Length-n1 FFT along columns of out_decomp
          out_decomp = fft(out_decomp, n1, 1);
          % Twiddle multiply
          out_decomp = F_n(1:n1, 1:n2).*out_decomp;
        else
          error('invalid partial-pass mode');
        endif

        output = set_pp_data(output, out_decomp, pp_dim, idx1, idx2);
      endfor
    endfor
  endfunction

  function output = partial_pass_step_3_4(input, pp_dim, n1, n2, F_n1, F_n2, F_n, mode)
    output = input;

    [dim1, dim2] = get_data_dim_partial_pass(input, pp_dim);

    for idx1=1:dim1
      for idx2=1:dim2
        in_decomp = get_pp_decomposed_data(output, pp_dim, idx1, idx2, n1, n2);

        if strcmp(mode, 'four-step')
          % Local transpose
          out_decomp = in_decomp.';
          % Length-n1 FFT along rows of out_decomp
          out_decomp = fft(out_decomp, n1, 2);
        elseif strcmp(mode, 'six-step')
          % Local transpose
          out_decomp = in_decomp.';
          % Length-n2 FFT along columns of out_decomp
          out_decomp = fft(out_decomp, n2, 1);
          % Local transpose
          out_decomp = out_decomp.';
        else
          error('invalid partial-pass mode');
        endif

        output = set_pp_data(output, out_decomp, pp_dim, idx1, idx2);
      endfor
    endfor
  endfunction

endfunction
