use rand::rngs::SmallRng;
use rand::SeedableRng;

use ffi::gaussian::Gaussian;
use kernels::state::State;
use kernels::{density::LogDensity, metric::IdentityMetric};
use runtime::{
    chain::Chain,
    mcmc::rwmh::{Rwmh, RwmhConfig},
};

fn main() {
    // let dim = 1000;
    // let target = Gaussian;
    // let metric = IdentityMetric::new(dim);
    // let step_size = 2.32 / (dim as f64).sqrt();
    // let config = RwmhConfig::default()
    //     .with_warmup(1000)
    //     .with_step_size(step_size)
    //     .with_draws(50);
    // let kernel = Rwmh::new(config, metric);
    // let state = State::new(dim);
    // let mut chain = Chain::new(kernel, target, state);
    let density = Gaussian;
    density.dlog_prob(vec![3.0; 5].as_slice());
    let x = 3.0;
    let (value, grad) = dlog_prob(x, 1.0);
    println!("value = {}", value);
    println!("grad  = {}", grad);

    // println!("acceptance = {}", kernel.acceptance_rate());
}
