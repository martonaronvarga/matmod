use super::proposal::{DenseCholeskyProposal, IsotropicProposal, Proposal};
use kernels::{buffer::OwnedBuffer, density::LogDensity, kernel::TransitionKernel, state::State};
use rand::{Rng, RngExt};

// ── Dual-averaging constants (Nesterov 2009, used by Stan) ────────────────────
const DELTA_TARGET: f64 = 0.234; // optimal acceptance rate for RWMH in high dim
const GAMMA: f64 = 0.05;
const T0: f64 = 10.0;
const KAPPA: f64 = 0.75;

#[inline(always)]
fn finite_or_neg_inf(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        f64::NEG_INFINITY
    }
}

#[inline(always)]
fn positive_finite(x: f64) -> f64 {
    x.max(f64::MIN_POSITIVE)
}

#[inline(always)]
fn log_accept_ratio(current_lp: f64, proposal_lp: f64) -> f64 {
    if proposal_lp.is_finite() {
        if current_lp.is_finite() {
            proposal_lp - current_lp
        } else {
            f64::INFINITY
        }
    } else {
        f64::NEG_INFINITY
    }
}

#[derive(Debug, Clone)]
pub struct RwmhConfig {
    /// Initial proposal scale. If using a Cholesky factor, this multiplies it.
    /// Gelman's rule: 2.38 / sqrt(d) for isotropic proposals.
    pub step_size: f64,
    /// Number of warmup (burn-in) iterations.
    pub n_warmup: usize,
    /// Number of sampling iterations after warmup.
    pub n_draws: usize,
    /// Whether to adapt step size during warmup via dual averaging.
    pub adapt_step_size: bool,
    /// Target acceptance rate for adaptation. Default 0.234 (optimal RWMH).
    pub target_accept_rate: f64,
}

impl Default for RwmhConfig {
    fn default() -> Self {
        Self {
            step_size: 1.0,
            n_warmup: 1_000,
            n_draws: 1_000,
            adapt_step_size: true,
            target_accept_rate: DELTA_TARGET,
        }
    }
}

impl RwmhConfig {
    #[inline]
    pub fn with_step_size(mut self, s: f64) -> Self {
        self.step_size = s;
        self
    }
    #[inline]
    pub fn with_warmup(mut self, n: usize) -> Self {
        self.n_warmup = n;
        self
    }
    #[inline]
    pub fn with_draws(mut self, n: usize) -> Self {
        self.n_draws = n;
        self
    }
    #[inline]
    pub fn with_adapt_step_size(mut self, adapt: bool) -> Self {
        self.adapt_step_size = adapt;
        self
    }

    #[inline]
    pub fn with_target_accept_rate(mut self, delta: f64) -> Self {
        self.target_accept_rate = delta;
        self
    }
}

struct DualAvg {
    mu: f64,          // log(10 * initial_step_size)
    h_bar: f64,       // running mean of acceptance stats
    log_eps_bar: f64, // smoothed log step size
    m: usize,         // iteration counter
    delta: f64,       // target acceptance rate
}

impl DualAvg {
    #[inline]
    fn new(initial_step_size: f64, target_accept_rate: f64) -> Self {
        assert!(initial_step_size.is_finite() && initial_step_size > 0.0);
        assert!(
            target_accept_rate.is_finite() && target_accept_rate > 0.0 && target_accept_rate < 1.0
        );

        Self {
            mu: (10.0 * initial_step_size).ln(),
            h_bar: 0.0,
            log_eps_bar: initial_step_size.ln(),
            m: 0,
            delta: target_accept_rate,
        }
    }

    /// Update and return the new step size for the next iteration.
    #[inline]
    fn update(&mut self, accept_prob: f64) -> f64 {
        self.m += 1;
        let m = self.m as f64;

        let w = 1.0 / (m + T0);
        self.h_bar = (1.0 - w) * self.h_bar + w * (self.delta - accept_prob);

        let log_eps = self.mu - (m.sqrt() / GAMMA) * self.h_bar;
        let decay = m.powf(-KAPPA);

        self.log_eps_bar = decay * log_eps + (1.0 - decay) * self.log_eps_bar;
        positive_finite(log_eps.exp())
    }

    /// After warmup, fix step size at the smoothed estimate.
    #[inline]
    fn final_step_size(&self) -> f64 {
        positive_finite(self.log_eps_bar.exp())
    }
}

/// Flat row-major draw matrix: [n_draws × dim].
pub struct Draws {
    pub data: OwnedBuffer,
    pub n_draws: usize,
    pub dim: usize,
}

impl Draws {
    #[inline]
    fn new(n_draws: usize, dim: usize) -> Self {
        let len = n_draws.checked_mul(dim).expect("draw buffer overflow");
        Self {
            data: OwnedBuffer::new(len),
            n_draws,
            dim,
        }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        let start = i * self.dim;
        &self.data[start..start + self.dim]
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        let start = i * self.dim;
        &mut self.data[start..start + self.dim]
    }
}

pub struct Rwmh<P> {
    pub config: RwmhConfig,
    proposal: P,
    dim: usize,
    /// Proposal scratch buffer. Reused every step to avoid allocation.
    scratch: OwnedBuffer,
    // statistics
    accepted_warmup: usize,
    total_warmup: usize,
    accepted_main: usize,
    total_main: usize,
    dual_avg: Option<DualAvg>,
}

pub type IsotropicRwmh = Rwmh<IsotropicProposal>;
pub type DenseCholeskyRwmh = Rwmh<DenseCholeskyProposal>;

impl<P: Proposal> Rwmh<P> {
    #[inline]
    pub fn new(config: RwmhConfig, proposal: P) -> Self {
        assert!(config.step_size.is_finite() && config.step_size > 0.0);
        let dim = proposal.dim();

        let dual_avg = if config.adapt_step_size {
            Some(DualAvg::new(config.step_size, config.target_accept_rate))
        } else {
            None
        };

        Self {
            config,
            proposal,
            dim,
            scratch: OwnedBuffer::new(dim),
            accepted_warmup: 0,
            total_warmup: 0,
            accepted_main: 0,
            total_main: 0,
            dual_avg,
        }
    }

    #[inline]
    pub fn warmup_acceptance_rate(&self) -> f64 {
        if self.total_warmup == 0 {
            0.0
        } else {
            self.accepted_warmup as f64 / self.total_warmup as f64
        }
    }

    #[inline]
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_main == 0 {
            0.0
        } else {
            self.accepted_main as f64 / self.total_main as f64
        }
    }

    #[inline]
    pub fn current_step_size(&self) -> f64 {
        self.config.step_size
    }

    #[inline(always)]
    fn finish_warmup(&mut self) {
        if let Some(step_size) = self.dual_avg.as_ref().map(|da| da.final_step_size()) {
            self.config.step_size = step_size;
        }
    }

    #[inline(always)]
    fn step_impl<D: LogDensity, R: Rng + ?Sized>(
        &mut self,
        state: &mut State,
        density: &D,
        rng: &mut R,
    ) -> (bool, f64) {
        debug_assert_eq!(state.dim(), self.dim);

        if !state.log_prob.is_finite() {
            state.log_prob = finite_or_neg_inf(density.log_prob(state.position.as_slice()));
        }

        let current_lp = finite_or_neg_inf(state.log_prob);

        {
            let x = state.position.as_slice();
            let out = self.scratch.as_mut_slice();
            self.proposal
                .propose_into(x, out, self.config.step_size, rng);
        }

        let proposal_lp = finite_or_neg_inf(density.log_prob(self.scratch.as_slice()));
        let log_alpha = log_accept_ratio(current_lp, proposal_lp);
        let accept_prob = if log_alpha.is_sign_positive() {
            1.0
        } else {
            log_alpha.exp()
        };

        let accepted = rng.random::<f64>() < accept_prob;

        if accepted {
            std::mem::swap(&mut state.position, &mut self.scratch);
            state.log_prob = proposal_lp;
        }

        (accepted, accept_prob)
    }

    pub fn sample<D: LogDensity, R: Rng + ?Sized>(
        &mut self,
        density: &D,
        state: &mut State,
        rng: &mut R,
    ) -> Draws {
        debug_assert_eq!(state.dim(), self.dim);

        if !state.log_prob.is_finite() {
            state.log_prob = finite_or_neg_inf(density.log_prob(state.position.as_slice()));
        }

        self.total_warmup = 0;
        self.total_main = 0;
        self.accepted_warmup = 0;
        self.accepted_main = 0;

        if self.config.adapt_step_size {
            self.dual_avg = Some(DualAvg::new(
                self.config.step_size,
                self.config.target_accept_rate,
            ));
        } else {
            self.dual_avg = None;
        }

        for _ in 0..self.config.n_warmup {
            let (accepted, accept_prob) = self.step_impl(state, density, rng);

            self.total_warmup += 1;
            self.accepted_warmup += accepted as usize;

            if let Some(step_size) = self.dual_avg.as_mut().map(|da| da.update(accept_prob)) {
                self.config.step_size = step_size;
            }
        }

        if self.config.n_warmup > 0 {
            self.finish_warmup();
        }

        let mut draws = Draws::new(self.config.n_draws, self.dim);

        for i in 0..self.config.n_draws {
            let (accepted, _) = self.step_impl(state, density, rng);

            self.total_main += 1;
            self.accepted_main += accepted as usize;

            draws.row_mut(i).copy_from_slice(state.position.as_slice());
        }

        draws
    }
}

impl Rwmh<IsotropicProposal> {
    #[inline]
    pub fn isotropic(config: RwmhConfig, dim: usize) -> Self {
        Self::new(config, IsotropicProposal::new(dim))
    }
}

impl Rwmh<DenseCholeskyProposal> {
    #[inline]
    pub fn dense_cholesky(config: RwmhConfig, dim: usize, chol: OwnedBuffer) -> Self {
        Self::new(config, DenseCholeskyProposal::new(dim, chol))
    }

    #[inline]
    pub fn with_cholesky(config: RwmhConfig, dim: usize, chol: OwnedBuffer) -> Self {
        Self::dense_cholesky(config, dim, chol)
    }
}

impl TransitionKernel for Rwmh<IsotropicProposal> {
    #[inline]
    fn step<D: LogDensity, R: Rng + ?Sized>(
        &mut self,
        state: &mut State,
        density: &D,
        rng: &mut R,
    ) -> bool {
        self.step_impl(state, density, rng).0
    }
}

impl TransitionKernel for Rwmh<DenseCholeskyProposal> {
    #[inline]
    fn step<D: LogDensity, R: Rng + ?Sized>(
        &mut self,
        state: &mut State,
        density: &D,
        rng: &mut R,
    ) -> bool {
        self.step_impl(state, density, rng).0
    }
}

#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use ffi::gaussian::Gaussian;
    use rand::rngs::{StdRng, Xoshiro256PlusPlus};
    use rand::SeedableRng;

    fn run_isotropic(dim: usize, draws: usize) {
        let mut state = State::new(dim);
        state.position.fill(0.1);

        let density = Gaussian;
        let config = RwmhConfig::default()
            .with_warmup(500)
            .with_draws(draws)
            .with_step_size(2.38 / (dim as f64).sqrt())
            .with_adapt_step_size(false);

        let mut kernel = Rwmh::isotropic(config, dim);
        let mut rng = StdRng::seed_from_u64(42);

        let start = std::time::Instant::now();
        let _draws = kernel.sample(&density, &mut state, &mut rng);
        let elapsed = start.elapsed();

        println!("dim={} draws={} dense=false time={:?}", dim, draws, elapsed);
    }

    fn run_dense(dim: usize, draws: usize) {
        let mut state = State::new(dim);
        state.position.fill(0.1);

        let density = Gaussian;
        let config = RwmhConfig::default()
            .with_warmup(500)
            .with_draws(draws)
            .with_step_size(2.38 / (dim as f64).sqrt())
            .with_adapt_step_size(false);

        let mut chol = OwnedBuffer::new(dim * dim);
        chol.fill(0.0);
        for i in 0..dim {
            chol.as_mut_slice()[i * dim + i] = 1.0;
        }

        let mut kernel = Rwmh::dense_cholesky(config, dim, chol);
        let mut rng = StdRng::seed_from_u64(42);

        let start = std::time::Instant::now();
        let _draws = kernel.sample(&density, &mut state, &mut rng);
        let elapsed = start.elapsed();

        println!("dim={} draws={} dense=true time={:?}", dim, draws, elapsed);
    }

    #[test]
    fn benchmark_matrix() {
        let dims = [128, 512, 1000];
        let draws = [10_000, 50_000];

        for &d in &dims {
            for &n in &draws {
                run_isotropic(d, n);
                run_dense(d, n);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernels::density::LogDensity;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    struct StandardNormalDensity;

    impl LogDensity for StandardNormalDensity {
        #[inline]
        fn log_prob(&self, x: &[f64]) -> f64 {
            -0.5 * x.iter().map(|v| v * v).sum::<f64>()
        }
    }

    #[inline]
    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        assert!(
            (a - b).abs() <= tol,
            "{msg}: got {a}, expected {b}, tol {tol}"
        );
    }

    #[test]
    fn isotropic_proposal_moments_match_theory() {
        let dim = 4;
        let n = 50_000;
        let step = 0.7;

        let mut rng = StdRng::seed_from_u64(11);
        let mut proposal = IsotropicProposal::new(dim);

        let x = vec![0.0; dim];
        let mut out = vec![0.0; dim];
        let mut sum = vec![0.0; dim];
        let mut sumsq = vec![0.0; dim];

        for _ in 0..n {
            proposal.propose_into(&x, &mut out, step, &mut rng);
            for d in 0..dim {
                sum[d] += out[d];
                sumsq[d] += out[d] * out[d];
            }
        }

        let n = n as f64;
        for d in 0..dim {
            let mean = sum[d] / n;
            let var = sumsq[d] / n - mean * mean;
            assert_close(mean, 0.0, 0.02, "mean");
            assert_close(var, step * step, 0.03, "variance");
        }
    }

    #[test]
    fn dense_cholesky_identity_matches_isotropic_moments() {
        let dim = 4;
        let n = 50_000;
        let step = 0.7;

        let mut chol = OwnedBuffer::new(dim * dim);
        chol.fill(0.0);
        for i in 0..dim {
            chol.as_mut_slice()[i * dim + i] = 1.0;
        }

        let mut rng = StdRng::seed_from_u64(13);
        let mut proposal = DenseCholeskyProposal::new(dim, chol);

        let x = vec![0.0; dim];
        let mut out = vec![0.0; dim];
        let mut sum = vec![0.0; dim];
        let mut sumsq = vec![0.0; dim];

        for _ in 0..n {
            proposal.propose_into(&x, &mut out, step, &mut rng);
            for d in 0..dim {
                sum[d] += out[d];
                sumsq[d] += out[d] * out[d];
            }
        }

        let n = n as f64;
        for d in 0..dim {
            let mean = sum[d] / n;
            let var = sumsq[d] / n - mean * mean;
            assert_close(mean, 0.0, 0.02, "mean");
            assert_close(var, step * step, 0.03, "variance");
        }
    }

    #[test]
    fn rwmh_recovers_standard_normal_moments() {
        let dim = 4;
        let mut rng = StdRng::seed_from_u64(17);
        let density = StandardNormalDensity;

        let config = RwmhConfig::default()
            .with_step_size(0.8)
            .with_warmup(2_000)
            .with_draws(10_000)
            .with_adapt_step_size(false);

        let mut kernel = Rwmh::isotropic(config, dim);
        let mut state = State::new(dim);
        state.position.fill(0.0);
        state.initialize_log_prob(&density);

        let draws = kernel.sample(&density, &mut state, &mut rng);

        let mut mean = vec![0.0; dim];
        let mut second = vec![0.0; dim];

        for i in 0..draws.n_draws {
            let row = draws.row(i);
            for d in 0..dim {
                mean[d] += row[d];
                second[d] += row[d] * row[d];
            }
        }

        let n = draws.n_draws as f64;
        for d in 0..dim {
            let m = mean[d] / n;
            let v = second[d] / n - m * m;
            assert!(m.abs() < 0.08, "mean too far from 0: {m}");
            assert!((v - 1.0).abs() < 0.12, "variance too far from 1: {v}");
        }

        let ar = kernel.acceptance_rate();
        assert!(ar > 0.15 && ar < 0.9, "unexpected acceptance rate: {ar}");
    }
}
