use kernels::kernel::Kernel;
use rand::Rng;

pub struct Chain<K, D>
where
    K: Kernel<D>,
{
    pub kernel: K,
    pub target: D,
    pub state: K::State,
}
impl<K, D> Chain<K, D>
where
    K: Kernel<D>,
{
    pub fn new(kernel: K, target: D, state: K::State) -> Self {
        Self {
            kernel,
            target,
            state,
        }
    }

    pub fn initialize(&mut self) {
        self.kernel.initialize(&mut self.state, &self.target);
    }

    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        self.kernel.step(&mut self.state, &self.target, rng)
    }
}
pub fn run_chain<K, D, R>(chain: &mut Chain<K, D>, n_steps: usize, rng: &mut R)
where
    K: Kernel<D>,
    R: Rng + ?Sized,
{
    chain.initialize();
    for _ in 0..n_steps {
        chain.step(rng);
    }
}
