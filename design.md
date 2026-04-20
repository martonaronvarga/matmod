# 1. Core principle
Split everything into:

## (A) Statistical objects and low-level capabilities
- `LogDensity`
- `GradLogDensity`
- `Metric`
- `TransitionModel` / `ObservationModel`

## (B) Inference algorithms
- HMC / NUTS
- particle filters / SMC
- Gibbs / MH
- BP / variational

## (C) Orchestration (rust)
- scheduling
- composition of kernels + interface
- memory ownership + FFI

- prefer generics in hot paths
bad:
```rust
  Box<dyn TransitionKernel>
```
good:
```rust
  struct Chain<K: TransitionKernel> { kernel: K }
```

- explicit ownership and mutability
  - pass `&mut state` for kernels
  - avoid cloning large vectors/arrays
  - reuse buffers aggressively

- feature-gated backends:
```Cargo.toml
  [features]
  simd = []
  openblas = []
  enzyme = []
  futhark = []
```

## (D) Analysis layer (python/R)
- posterior analysis
- diagnostics
- visualization
---
# 2. Language choices

## rust (orchestration):
  - composition
  - FFI
  - parallelism: `rayon`, `tokio`
- crates:
  - `faer` linalg
  - `rand`, `rand_distr` randomness
  - `rayon`, `tokio` parallelism
  - `pyo3` / `maturin` python bridge
  - `cxx` / `ffi-support`, `bindgen` c++ interop
  - `futhark-bindgen` futhark interop

## C++ (performance kernels + existing libs)
- eigen (linalg)
- Stan Math (autodiff + probability functions)
- enzyme (autodiff at LLVM IR level)
- libtorch optional (neural likelihoods)

## Futhark (experimental)
- batched likelihood evaluation
- particle filtering
- large-scale Monte-Carlo
- useful for: data-parallel array transforms, gpu performance

# 3. Model specific implementations

## Drift Diffusion models
- C++ implementation (HDDM or Stan Math)
- wrap into rust ffi
- cache / approx. using:
  - spline interp
  - neural surrogate (optionally)

## NHMC / HMM
- rust / futhark: forward-backward
- log-space semiring
- parallelize over sequences
- nhmc: represent transition matrix as closure or tensor

## MRF / NHMRF
- multiple inference backends
  1. Gibbs / MH
  2. Loopy belief propagation
  3. Graph cuts
- core graph structure: rust
- heavy ops:
  - message updates: optionally futhark
  - energy eval: C++ SIMD

## Dynamic / state-space
- Linear Gaussian
  - kalman filter in rust
- nonlinear
  - particle filter:
    - rust + rayon
    - optional futhark for resampling / propagation
- diffusions
  - euler-maruyama / milstein in C++/rust
  - Stan Math gradients

## Monte Carlo / MCMC
Minimal core:
- RWMH (fallback)
- HMC (leapfrog + autodiff via Stan Math? or self leapfrog in rust AD in enzyme)
- NUTS
- Gibbs (for discrete blocks)
- trait based kernel interface
- RNG strategy: seeded per chain, deterministic splitting (`rand_chacha` or `PCG`)

# 4. Interop
- rust <-> c++
  - `cxx`
  - expose: likelihoods, gradients
- rust <-> python
  - `pyo3` + `maturin`
  - expose: sampling results, diagnostics
- rust <-> futhark
  - `futhark-bindgen`

# 5. Optimizers

(1) Mixed inference
- continuous: HMC, NUTS, RMHMC, ChEES-HMC, SMC
- discrete: Gibbs
- time-series: particle MCMC

(2) Likelihood amortization?
- precompute grids
- train neural surrogates (optional)

(3) Exploit structure
- HMM: vectorize over seqs
- MRF: sparse adjacency
- SSM: block structure

(4) Memory layout
- SoA > AoS for SIMD
- col major for blas and linalg
- contiguous arays for futhark/gpu
