# Modeling the Congruency Sequence Effect

## Project layout

```
matmod/
├── crates/
│   ├── kernels/             # traits + algebra
│   ├── ffi/                 # FFI bindings
│   ├── runtime/             # scheduler + execution
│   ├── app/                 # binary (experiments)
│   ├── data_prep/           # data loading and preparation
│   ├── Cargo.toml           # workspace
```

### Structure

```
kernels/
├── lib.rs
├── state.rs          # chain state and working buffers
├── density.rs        # LogDensity, GradLogDensity, HessianLogDensity
├── kernel.rs         # TransitionKernel, GradientKernel, state-space traits
├── buffer.rs         # owned buffers + views
├── metric.rs         # Identity, diagonal, dense Cholesky geometry
```

- prefer generics over trait objects
- keep the core trait layer small

```
ffi/
├── lib.rs
├── gaussian.rs      # Gaussian target backend
├── stan.rs          # C++ / Stan-style backend example
└── ddm.rs           # future DDM backend
```

- raw FFI stays in the wrapper layer
- wrapper types implement `LogDensity` / `GradLogDensity`

```
runtime/
├── lib.rs
├── scheduler/
│   ├── mod.rs
│   ├── chain.rs        # single-chain state machine
│   ├── multi.rs        # multi-chain orchestration
│   └── stage.rs        # pipeline stages
├── execution/
│   ├── mod.rs
│   ├── rayon.rs        # parallel backend
│   └── sequential.rs
├── adapt/
│   ├── mod.rs
│   └── step_size.rs
├── io/
│   ├── mod.rs
│   └── writer.rs
└── config.rs
```

- no ffi here
- algorithms depend only on the trait layer
- uses `rayon` for parallelism and sched
- add HMC / NUTS / SMC here as separate modules

```
app/
├── main.rs
└── ...
```

- wire kernels together
- define models
- run experiments
---
## Guides and links

1. [polars](https://docs.rs/polars/latest/polars/)
2. [Rust FFI](https://jakegoulding.com/rust-ffi-omnibus/)
3. [Stan Math Wiki & Quickstart](https://github.com/stan-dev/math/wiki)
4. [SMTC (Particle Filter in C++)](https://github.com/awllee/smctc)
5. [Eigen C++](https://eigen.tuxfamily.org/dox/GettingStarted.html)

6. [Futhark Scan](https://futhark-book.readthedocs.io/en/latest/functional-parallel-programming.html#scan)
7. [Futhark C/Rust Backend](https://futhark.readthedocs.io/en/latest/c-api.html)

8. [Zig Guide](https://zig.guide/)
9. [Zig C Interop](https://ziglang.org/documentation/master/#C)

## Profiling

- Perf: `perf record -g ./your_binary && perf report` or open `perf.data` in `hotspot`
- Valgrind: `valgrind --tool=massif ./binary` to profile memory, view with `massif-visualizer`
- [hyperfine](https://github.com/sharkdp/hyperfine)
