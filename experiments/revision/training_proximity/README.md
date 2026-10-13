# Training-proximity diagnostic

The original 40,000-pair pools were generated in memory and not saved. The
diagnostic script exactly reconstructs their initial-condition draws from the
recorded PRNG seed, split pattern, Fourier coefficient distribution, and (for
Burgers) chunk/fold-in scheme. It computes the relative nearest-neighbor
distance in the 100-point grid L2 geometry, then reports Pearson and rank
associations with prediction and spectral error. These associations are
descriptive and are not, by themselves, evidence for or against memorization.
