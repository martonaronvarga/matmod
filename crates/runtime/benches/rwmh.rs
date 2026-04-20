use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;

use ffi::gaussian::Gaussian;
use kernels::density::LogDensity;
use kernels::kernel::TransitionKernel;
use kernels::metric::CholeskyFactor;
use kernels::state::State;

use runtime::mcmc::proposal::{DenseCholeskyProposal, IsotropicProposal};
use runtime::mcmc::rwmh::{Rwmh, RwmhConfig};
use runtime::mcmc::Proposal;

fn bench_rwmh_isotropic(c: &mut Criterion) {
    let dim = 256;
    let density = Gaussian;

    let step_size = 2.38 / (dim as f64).sqrt();

    let config = RwmhConfig::default()
        .with_warmup(0)
        .with_draws(0)
        .with_adapt_step_size(false)
        .with_step_size(step_size);

    let mut kernel = Rwmh::isotropic(config, dim);

    c.bench_function("rwmh_isotropic_step", |b| {
        b.iter_batched(
            || {
                let mut state = State::new(dim);
                state.position.fill(0.1);
                let rng = Xoshiro256PlusPlus::seed_from_u64(42);
                (state, rng)
            },
            |(mut state, mut rng)| {
                kernel.step(
                    black_box(&mut state),
                    black_box(&density),
                    black_box(&mut rng),
                )
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_rwmh_dense(c: &mut Criterion) {
    let dim = 256;
    let density = Gaussian;

    let step_size = 2.38 / (dim as f64).sqrt();

    let config = RwmhConfig::default()
        .with_warmup(0)
        .with_draws(0)
        .with_adapt_step_size(false)
        .with_step_size(step_size);

    let mut chol = kernels::buffer::OwnedBuffer::new(dim * dim);
    chol.fill(0.0);
    for i in 0..dim {
        chol.as_mut_slice()[i * dim + i] = 1.0;
    }

    let factor = CholeskyFactor::new_lower(dim, chol);
    let mut kernel = Rwmh::dense_cholesky(config, factor);

    c.bench_function("rwmh_dense_step", |b| {
        b.iter_batched(
            || {
                let rng = Xoshiro256PlusPlus::seed_from_u64(42);
                let mut state = State::new(dim);
                state.position.fill(0.1);
                (state, rng)
            },
            |(mut state, mut rng)| {
                kernel.step(
                    black_box(&mut state),
                    black_box(&density),
                    black_box(&mut rng),
                )
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_isotropic_proposal(c: &mut Criterion) {
    let dim = 256;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut proposal = IsotropicProposal::new(dim);

    let x = vec![0.0; dim];
    let mut out = vec![0.0; dim];

    c.bench_function("proposal_isotropic", |b| {
        b.iter(|| {
            proposal.propose_into(
                black_box(&x),
                black_box(&mut out),
                black_box(1.0),
                black_box(&mut rng),
            )
        })
    });
}

fn bench_dense_proposal(c: &mut Criterion) {
    let dim = 256;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

    let mut chol = kernels::buffer::OwnedBuffer::new(dim * dim);
    chol.fill(0.0);
    for i in 0..dim {
        chol.as_mut_slice()[i * dim + i] = 1.0;
    }

    let factor = CholeskyFactor::new_lower(dim, chol);
    let mut proposal = DenseCholeskyProposal::new(factor);

    let x = vec![0.0; dim];
    let mut out = vec![0.0; dim];

    c.bench_function("proposal_dense", |b| {
        b.iter(|| {
            proposal.propose_into(
                black_box(&x),
                black_box(&mut out),
                black_box(1.0),
                black_box(&mut rng),
            )
        })
    });
}

fn bench_density(c: &mut Criterion) {
    let dim = 256;
    let density = Gaussian;

    let x = vec![0.1; dim];

    c.bench_function("density_gaussian", |b| {
        b.iter(|| black_box(density.log_prob(black_box(&x))))
    });
}

fn bench_step_no_density(c: &mut Criterion) {
    let dim = 256;

    struct Dummy;
    impl LogDensity for Dummy {
        #[inline(always)]
        fn log_prob(&self, _: &[f64]) -> f64 {
            0.0
        }
    }

    let density = Dummy;

    let config = RwmhConfig::default()
        .with_warmup(0)
        .with_draws(0)
        .with_adapt_step_size(false);

    let mut kernel = Rwmh::isotropic(config, dim);

    c.bench_function("step_no_density", |b| {
        b.iter_batched(
            || {
                let state = State::new(dim);
                let rng = Xoshiro256PlusPlus::seed_from_u64(42);
                (state, rng)
            },
            |(mut state, mut rng)| kernel.step(&mut state, &density, &mut rng),
            BatchSize::SmallInput,
        )
    });
}

fn bench_fill_normals(c: &mut Criterion) {
    let dim = 256;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut buf = vec![0.0; dim];

    c.bench_function("fill_normals", |b| {
        b.iter(|| runtime::mcmc::fill_normals(black_box(&mut buf), black_box(&mut rng)))
    });
}

fn bench_copy(c: &mut Criterion) {
    let dim = 256;
    let x = vec![0.1; dim];
    let mut out = vec![0.0; dim];

    c.bench_function("copy_from_slice", |b| {
        b.iter(|| out.copy_from_slice(black_box(&x)))
    });
}

criterion_group!(
    benches,
    bench_rwmh_isotropic,
    bench_rwmh_dense,
    bench_isotropic_proposal,
    bench_dense_proposal,
    bench_density,
    bench_step_no_density,
    bench_fill_normals,
    bench_copy,
);

criterion_main!(benches);
