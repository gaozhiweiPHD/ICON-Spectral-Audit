# Third-party notices

The attention-mask and prompt-assembly utilities in
`src/common/models_utils.py` derive from the public ICON implementation:
https://github.com/LiuYangMage/in-context-operator-networks
(`icon-lm/models_utils.py`, upstream commit
`b994edd0c31aaeb41cf481d80ac48fd4e84a1560`).
The upstream MIT license and Copyright (c) 2023 Liu Yang are retained in
`third_party/ICON-MIT-LICENSE`. This attribution also applies to ICON architecture
and prompt-construction portions of the preserved experiment code.
Numerical PDE generators and manuscript diagnostics are supplied from the
authors' local ICON project. Exact origins of every historical line have not
been certified; author review remains required before public release.

JAX, Flax, Optax, einshape, NumPy and Matplotlib are dependencies, not vendored
distributions. Their respective licenses remain applicable. No pretrained
third-party model or third-party experimental dataset is bundled.
