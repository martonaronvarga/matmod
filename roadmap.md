# Development plan

## Phase 1: foundation & data pipeline

- goal: read processed datafiles and set up the simulation harness
- tasks:
  - initialize orchestrator, read data with `polars`
  - define rust structs to hold data, setup FFI bindings/subprocess calls to C++, Zig,CmdStan, Futhark

  ## Phase 2: baseline & conflict DDM

  - goal: replicate CSE Diffusion model
