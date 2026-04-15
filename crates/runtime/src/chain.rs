use kernels::density::Gradient;
use kernels::kernel::TransitionKernel;
use kernels::state::State;
use rand::Rng;

pub fn run_chain<K: TransitionKernel, D: Gradient>(
    kernel: &mut K,
    density: &D,
    state: &mut State,
    n_steps: usize,
    rng: &mut impl Rng,
) {
    kernel.initialize(state, density);
    for _ in 0..n_steps {
        kernel.step(state, density, rng);
    }
}
