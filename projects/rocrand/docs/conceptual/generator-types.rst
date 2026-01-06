.. meta::
  :description: Background on random number generation and generator types
  :keywords: rocRAND, ROCm, API, documentation, generator types, random number generation
  
.. _generator-types:

*******************************************************************
Random number generators
*******************************************************************

A random number generator (RNG) can be implemented by using a hardware-based
randomization source or strictly in software. Hardware devices can sample environmental
details that are statistically random. Methods that are strictly based on software are much faster, but do
not have access to true sources of randomization. This means that they can only provide
various approximations of true randomness.

Any decision regarding what randomization technique to use must account for tradeoffs in statistical performance
against computing performance. rocRAND implements several different software-based random number generators
that attempt to balance performance and viable randomization. This topic discusses the various
classes of random number generators and provides some background on the various generator types 
supported by rocRAND.

Hardware and software random number generators
==============================================

Random number generators can be classified as hardware RNGs or software RNGs. The software-based variants
can be further subdivided into pseudo-random number generators (PRNGs) or quasi-random number generators (QRNGs).

Hardware random number generators
---------------------------------

Hardware random number generators (HRNGs) produce random numbers from physical sources characterized by entropy.
They sample low-level signals that contain random noise, such as thermal noise, Brownian motion, and
electronic circuit jitter and instability. They can output nearly perfect random numbers, which makes
them valuable for data encryption. However, the sensors often need to be calibrated, and
their output must be sampled, converted to a digital format, and conditioned, which takes more time.
For many applications, the benefits don't outweigh the additional complexity and slower speed.

Pseudo-random number generators
---------------------------------

Pseudo-random number generators (PRNGs) are implemented in software. They generate a sequence of random-seeming
numbers algorithmically. This sequence is deterministically dependent on the initial seed value,
settings, and the algorithm in use, so it's not truly random.
Therefore, it's important to choose good values for these parameters so the algorithm can produce a sequence
that provides a close approximation of true randomness.

Two important advantages of SRNGs are that they are very fast and that they generate reproducible results.
Many of the fundamental operations of SRNGs, such as XOR operations and shifts, are very fast to
calculate. They are reproducible in that they generate the same results each time, given the same
initial seed, parameters, and algorithm. This is useful for analysis and testing.

The downside of their lack of true randomness is that the generated sequences can include artifacts that fail
statistical pattern-detection tests. For instance, they might contain a series of alternating sequences
that are all ascending or descending, or fail to completely fill the potential output space in all
dimensions. These artifacts might pose problems if randomness is required for cryptographic security, for
example, because new outputs cannot be able to be predicted from earlier results. The statistical artifacts
could reveal information about the parameters or the algorithm. However, this might not matter if
randomization is required for video game inputs or statistical modeling for research.

rocRAND provides a choice of several random number generators that show good performance and create
sequences that perform well on statistical tests for randomness. Here's a list and
short description of the supported PRNGs.

*  **XORWOW**: This is an example of an xorshift generator. It's based on linear recurrence
   and implemented using XOR and shift operations, which makes it very fast.
   Xorshift generators typically need further refinement to pass the more challenging statistical tests for randomness, which
   XORWOW achieves by scrambling the output with an additive counter.

*  **MT19937**: (Mersenne Twister) This uses linear recurrence and addresses flaws in other generators.
   It is based on the prime number :math:`2^{19937}-1` and avoids major problems but still runs quickly.
   MT19937 is widely used in programming and graphics.

*  **MTGP32**: (Mersenne Twister for Graphics Programs) This is a variant of the Mersenne Twister for graphics programs.
   It generates sequences with high-dimensional uniformity for thorough coverage.

*  **Philox 4x32-10**: This is a counter-based PRNG. It uses an integer counter for its initial state along
   with multiplication-based mixing. It can support independent streams, so it's a good choice 
   for parallel computation of pseudo-random numbers in GPU clusters.

*  **MRG32K3A**/**MRG31K3P**: This is a combined multiple recursive generator (CMRG). It's suitable for
   use with GPUs, where it demonstrates good performance and provides a sufficient period length for most applications.

*  **LFSR113**: (Linear-feedback shift register) This is a shift register that uses an input bit that is a linear function
   of its previous state.

*  **ThreeFry 2x32-20**, **4x32-30**, **2x64-20**, and **4x64-20**: These are variants of a counter-based PRNG that's
   based on the Threefish block cipher.

Quasi-random number generators
---------------------------------

Quasi-random number generators (QRNGs) are a class of software RNGs that produce highly uniform
samples that contain a very even distribution of points. They do so by filling in gaps in the initial segment
of a software-generated random sequence. They might fail tests for true randomness, but their main purpose is to
balance thoroughness and randomness. They are useful for optimization, experimental design, and predictive
models by ensuring the domain of interest is meaningfully covered.

rocRAND supports several versions of the Sobol class of QRNGs, including Sobol32, Sobol64, Scrambled Sobol32, and
Scrambled Sobol64. These generators are based on the Sobol sequence, which uses a base of two to partition
the initial interval and then reorders the results to create a low-discrepancy sequence.

