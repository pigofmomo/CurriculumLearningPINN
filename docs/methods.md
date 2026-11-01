# Implementation notes

The main model uses one network with boundary-distance layers. C1 adjusts regional residual multipliers and optionally fits a low-degree polynomial to network predictions at anchors. The bridge is refreshed at scheduled updates and removed after its active window. C2 adjusts regional scores using residuals and gradient norms; setting `grad_norms_scale` to zero removes the gradient contribution.

The ODE forcing is unchanged from the initial implementation. Its evaluation reference is corrected using

```text
z = alpha + i*k
H(x) = 2*k*Re(exp(z*x)*(x*(1-x)/z^2 - 2*(1-2*x)/z^3 - 6/z^4))
y(x) = x*(1-x)*exp(alpha*x)*sin(k*x) + H(x) - (1-x)*H(0) - x*H(1)
```

The low/high parameters are `(2,10*pi)` and `(3,20*pi)`. `check.py` verifies the boundary values and independently integrates the unchanged source.

The hard-gate baseline cumulatively activates three spatial layers during Adam; active outer layers remain included. The ARS-derived comparison starts after full C1 and uses regional terms `exp(-q_j)*R_j + q_j/2`, followed by L-BFGS with fixed weights. This is a second-stage adaptation of the ARSPINN weighting objective.

FBPINN uses overlapping subdomains and a SciPy L-BFGS refinement step. XPINN uses independent networks with solution and residual interface penalties. Geometry and interface coefficients are explicit in the implementation.

The PB diagnostic uses two equal-area regions with a 10:1 priority ratio for 1000 Adam steps, followed by 500 equal-weight steps. Both regions remain active. It uses a hard boundary transform and a finite-difference reference generated at run time.

Saved C1 states include optimizer, random, sampling, and bridge state. Load only trusted checkpoints produced by this package.
