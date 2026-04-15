use kernels::kernel::TransitionKernel;
use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;

use ffi::gaussian::Gaussian;
use kernels::state::State;
use runtime::models::rwmh::Rwmh;
use runtime::models::rwmh::RwmhConfig;

fn main() {
    let dim = 6000;

    let mut state = State::new(dim);

    state.position.as_mut_slice().fill(0.1);
    let density = Gaussian;

    // 2.38 / sqrt(d)
    let step_size = 2.32 / (dim as f64).sqrt();

    let config = RwmhConfig::default()
        .with_warmup(1000)
        .with_step_size(step_size)
        .with_draws(50);

    // let mut kernel = Rwmh::isotropic(config, dim);
    let mut chol = vec![0.0; dim * dim];
    let mut chol_buf = kernels::buffer::OwnedBuffer::new(dim * dim);
    for i in 0..dim {
        chol[i * dim + i] = 1.0;
    }
    chol_buf.as_mut_slice().copy_from_slice(&chol);
    let mut kernel = Rwmh::dense_cholesky(config, dim, chol_buf);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

    // let _draws = kernel.sample(&density, &mut state, &mut rng);
    kernel.sample(&density, &mut state, &mut rng);
    // for _ in 0..50 {
    //     kernel.step(&mut state, &density, &mut rng);
    // }
    // println!("{:?}", draws.data.as_slice());
    println!("acceptance = {}", kernel.acceptance_rate());
}
