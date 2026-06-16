# Curriculum Learning PINN

This repository provides the initial implementation of **Curriculum Learning of Physics-Informed Neural Networks Based on Spatial Correlation**.

Physics-Informed Neural Networks (PINNs) have shown promise for solving partial differential equations (PDEs), but their training can be unstable due to non-convex optimization landscapes, imbalanced physical constraints, and insufficient spatial information propagation. This project explores a spatially correlated curriculum learning framework for improving PINN training on boundary value problems and spatially coupled PDE systems.

> **Note:** This is the first public version of the code. The repository is still under active development, and some scripts, configurations, and documentation may be updated in future versions.
>
> For a quick start, review the experiment scripts under `experiments/` and the reusable PINN components under `pinn_pro/`.

---

## Overview

The proposed framework is designed to improve PINN training by introducing spatially structured learning strategies. The current implementation includes the following main components:

1. **Spatial Curriculum Learning**

   The computational domain is divided into multiple subregions. Training weights are assigned according to spatial relationships, so that near-boundary regions are emphasized first and information is gradually guided toward the interior of the domain.

2. **Low-Frequency Information Bridge**

   Sparse anchor points are used to construct a low-frequency approximation of the solution, which provides additional consistency constraints across spatially separated regions and helps suppress global low-frequency drift.

3. **Region-Adaptive Reweighting**

   Subregion-wise PDE losses are dynamically adjusted according to the local optimization status, allowing the model to focus more on difficult regions with large residuals or insufficient gradient contribution.

---

## Notes

- The code is provided mainly for reference and research communication.
- Some experimental scripts or configuration files may be incomplete in the current version.
- Users interested in the implementation details may inspect and adapt the source code at their own discretion.
