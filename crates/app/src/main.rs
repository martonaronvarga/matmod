use ffi::gaussian::Gaussian;
use kernels::{
    metric::IdentityMetric,
    state::{LogProbState, State},
};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use runtime::{run_chain, Chain, Rwmh, RwmhConfig};

fn main() {
    let dim = 50;
    let target = Gaussian;
    let metric = IdentityMetric::new(dim);

    let config = RwmhConfig::default()
        .with_warmup(100)
        .with_draws(100)
        .with_adapt_step_size(true)
        .with_step_size(2.38 / (dim as f64).sqrt());

    let kernel = Rwmh::<_, State>::new(config, metric);
    let state = State::new(dim);
    let mut chain = Chain::new(kernel, target, state);

    let mut rng = SmallRng::seed_from_u64(42);
    run_chain(&mut chain, 100, &mut rng);

    println!(
        "log_prob={} dim={}",
        chain.state.log_prob(),
        chain.state.dim()
    );
}
