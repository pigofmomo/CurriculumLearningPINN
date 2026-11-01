# Third-party code

- DeepXDE 1.12.2: https://github.com/lululxvi/deepxde (LGPL-2.1). Installed as a dependency. Its Python computation code matches the local dependency used here after excluding comments and an additional top-level optimizer import; the local training extensions are in `pinn_pro`.
- FBPINNs: https://github.com/benmoseley/FBPINNs, commit `8b3b2334a8752511c8d6ada1302daa5a343c86ed` (MIT). The used library snapshot is in `third_party/fbpinns`; copyright/license text is in `third_party/FBPINNs_LICENSE`. Local additions provide equation adapters, ring windows, and a SciPy L-BFGS refinement hook in `trainers.py`. JAX's removed `tree_map` name is mapped to `tree_util.tree_map` in the runner.
- XPINNs: https://github.com/AmeyaJagtap/XPINNs (MIT; `third_party/XPINNs_LICENSE`). The PyTorch adapters implement the independent subnetworks, adaptive activations, and solution/residual interface formulation. Their geometry, normalization, and boundary conditions are specified in `methods.md`.
- The PB problem follows the CL-PINN case used in the companion study; this package contains the fixed-problem spatial-order diagnostic, not the parameter curriculum.

The root MIT license covers this project's code. Third-party files retain their own notices.
