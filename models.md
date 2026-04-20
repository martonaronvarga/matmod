## 1. Dynamic DDM / LCA

"a diffusion model for the congruency sequence effect" paper

### Structure

Latent state modulates parameters:
$\Theta_t = f(c_t, m_t)$
$y_t \sim DDM(\Theta_t)$

- Wiener likelihood
- kernels based on paper (time-varying drift)

- rust layer:
  - wrap likelihood
  - batch eval for HMC

- inference:
  - HMC / MCMC
  - precompute likelihood grids or splines 

## 2. Bayesian latent-state model with volatility

"control demand is inferred from recent evidence"

### Structure

Directed state-space model:
$(c_t, v_t) -> (c_{t+1}, v_{t+1}), \quad y_t \sim p(y_t \bar c_t, v_t)$

- state struct:

```rust
  struct State {
    position,
    gradient,
    momentum,
    log_prob,
  }
```

- transition:
  - nonlinear, stochastic
  - volatility-dependent variance

- inference:
  - particle MCMC
  - HMC if fully continuous

## 3. HMM / NHMC

### Structure

Discrete latent states:
$s_t \sim P_t(s_t \bar s_{t-1}, s_{t-2},\ldots)$

State augmentation, include memory:
$s_t = (c_t, m_t, v_t)$

- impl
  - forward-backward in log-space
  - vectorized
  - nhmc: store P_t as function:

```rust
fn transition(t: usize, context: &Ctx) -> Matrix
```

- inference:
  - exact (forward-backward)
  - EM / gradient-based

- optim:
  - simd over state dim
  - parallel seqs

## 4. MRF / NHMRF
- as multiple interacting latent processes:
  - control - memory - expectation (bidirectional)
  - soft constraints instead of transitions

> _prior over trajectories, not full model_

### Structure

Undirected:

$p(s) \propto exp(\sum_i \phi_i(s_i) + \sum_{i,j} \psi_{ij}(s_i, s_j))$

Temporal chain becomes chain MRF.

- impl
  - adjacency list
  - factor repr
  ```rust
  enum Factor {
    Unary,
    Pairwise,
  }
  ```
  - energy eval: simd

  - inference
    - Gibbs
    - loopy BP
    - HMC

#### NHMRF

Time varying potentials:
$\phi_t(s_t,s_{t-1}) = f(context_t)$
```rust
fn pairwise(t:usize, s_t: &S, s_prev: &S) -> f64
```
## 3. Hybrid control-plus-memory model
$c_{t+1} = f_c(c_t​,conflict_t, volatility_t) + \epsilon_t^c$
$m_{t+1} = f_m(m_t, feature repetitions_t, contingencies_t) + \epison_t^m$
$v_{t+1} = f_v(v_t, surprise_t) + \epsilon_t^v$
$y_t \sim DDM/LCA(c_t, m_t, v_t, stimulus features)$

## 4. Experimental quantum drift model (within-trial)

### Structure

State:
$\phi_t \in \mathcal{C}^2$

Evolution:
$\phi_{t+\Delta} = U(\Theta_t)\phi_t$

Decision:
- hitting probability or projection

- parameters:
  - phase
  - interference strength
  - decoherence

## 5. Experimental quantum + non-Markov lattice prior

### Structure

$\phi_t \sim U(\phi{t-1}, \Theta_t)$
$\Theta_t \sim exp(\sum \phi_ij(\Theta_i, \Theta_j))$

Graph:
- edges:
  - t-1
  - t-k
  - similarity based

- impl: rust factor graph
- inference: hmc, or variational

## 6. Neural surrogate hybrid

- train NN:
$\Theta_t = f_{NN}(history)$
- plug into ddm or quantum etc.

