.. meta::
  :description: rocAL RNNT dataloading reference
  :keywords: rocAL, ROCm, API, RNNT, dataloading, python, pytorch, speech recognition

*********************************
rocAL RNNT dataloading in Python
*********************************

rocAL supports the RNNT speech recognition model through audio readers and other functions that can be used with PyTorch.
 
All the functions used for RNNT dataloading are available in the ``amd.rocal.fn`` module. See :doc:`Using rocAL with Python API <../how-to/using-with-python>` for more details about this module. 

All the augmentations used in the RNNT dataloader pipeline are available as part of rocAL. These augmentations need to be plugged into the rocAL PyTorch dataloader to run the training. PyTorch samples can be found `in the rocAL GitHub repository <https://github.com/ROCm/rocAL/tree/develop/docs/examples/pytorch>`_.

.. Note::

    The rocAL GitHub repository does not host the entire RNNT dataloader source.

.. list-table:: Supported augmentations
    :header-rows: 1

    *   - Function
        - Description
        - Details
    
    *   - ``fn.resample`` 
        - Resamples an audio signal.
        - Resampling is achieved by applying a sinc filter with a Hann window. The extent is controlled by the function's ``quality`` argument.

    *   - ``fn.nonsilent_region``
        - Detects leading and trailing silences.
        - Returns the beginning and length of the non-silent region. Compares the short-term power calculated for the window length of the signal with a silence cut-off threshold. The signal is considered to be silent when the short term power in decibels is less than the cut-off threshold in decibels.

    *   - ``fn.slice`` 
        - Slices the input.
        - The slice is specified by an anchor and a shape for the slice.

    *   - ``fn.preemphasis_filter``
        - Applies a preemphasis filter to the input.
        - The filter used is ``Output[t] = Input[t] - coeff * Input[t-1]`` if ``t > 1`` and ``Output[t] = Input[t] - coeff * Input_border``  if ``t == 0``

        
    *   - ``fn.spectrogram`` 
        - Produces a spectrogram from a 1D audio signal.
        - 

    *   - ``fn.mel_filter_bank`` 
        - Converts a spectrogram to a mel spectrogram. 
        - Conversion is done by applying a bank of triangular filters where the frequency dimension is selected from the input layout.

    *   - ``fn.to_decibels`` 
        - Converts a magnitude to decibels.
        - The conversion is done using ``out[i] = multiplier * log10( max(min_ratio, input[i]/reference) )`` where ``min_ratio = pow(10, cutoff_db/multiplier)``.

    *   - ``fn.normalize`` 
        - Normalizes an input.
        - Normalization is done by removing the mean and dividing by the standard deviation.

