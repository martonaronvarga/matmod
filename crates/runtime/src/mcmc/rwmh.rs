use super::proposal::fill_normals;
use super::proposal::{DenseCholeskyProposal, IsotropicProposal, Proposal};
use core::marker::PhantomData;
use kernels::{
    buffer::OwnedBuffer,
    density::LogDensity,
    kernel::Kernel,
    metric::Metric,
    numeric::{finite_or_neg_inf, log_accept_ration, positive_finite},
    state::{Draws, LogProbState, State},
};
use rand::Rng;

const DELTA_TARGET: f64 = 0.234;
const GAMMA: f64 = 0.05;
const T0: f64 = 10.0;
const KAPPA: f64 = 0.75;

#[derive(Debug, Clone)]
pub struct RwmhConfig {
    pub step_size: f64,
    pub n_warmup: usize,
    pub n_draws: usize,
    pub adapt_step_size: bool,
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
    mu: f64,
    h_bar: f64,
    log_eps_bar: f64,
    m: usize,
    delta: f64,
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

    #[inline]
    fn final_step_size(&self) -> f64 {
        positive_finite(self.log_eps_bar.exp())
    }
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

pub struct Rwmh<M, S> {
    pub config: RwmhConfig,
    metric: M,
    noise: OwnedBuffer,
    proposal: OwnedBuffer,
    accepted_warmup: usize,
    total_warmup: usize,
    accepted_main: usize,
    total_main: usize,
    dual_avg: Option<DualAvg>,
    _marker: PhantomData<S>,
}

impl<M, S> Rwmh<M, S>
where
    M: Metric,
    S: LogProbState,
{
    #[inline]
    pub fn new(config: RwmhConfig, metric: M) -> Self {
        assert!(positive_finite(config.step_size));

        let dim = metric.dim();

        Self {
            config,
            metric,
            noise: OwnedBuffer::new(dim),
            proposal: OwnedBuffer::new(dim),
            accepted_warmup: 0,
            total_warmup: 0,
            accepted_main: 0,
            total_main: 0,
            dual_avg: config
                .adapt_step_size
                .then(|| DualAvg::new(config.step_size, config.target_accept_rate)),
            _marker: PhantomData,
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

    fn propose<R: Rng + ?Sized>(&mut self, state: &S, rng: &mut R) {
        fill_normals(self.noise.as_mut_slice(), rng);
        self.metric
            .apply_sqrt_into(self.noise.as_slice(), self.proposal.as_mut_slice());

        let x = state.position();
        let y = self.proposal.as_mut_slice();
        for i in 0..self.dim {
            y[i] = x[i] + self.config.step_size * y[i];
        }
    }

    #[inline(always)]
    fn step_impl<D, R>(&mut self, state: &mut S, target: &D, rng: &mut R) -> (bool, f64)
    where
        D: LogDensity,
        R: Rng + ?Sized,
    {
        debug_assert_eq!(state.dim(), self.dim);

        if !state.log_prob.is_finite() {
            state.initialize_log_prob(target);
        }

        let current_lp = finite_or_neg_inf(state.log_prob());
        self.propose(state, rng);

        let proposal_lp = finite_or_neg_inf(density.log_prob(self.proposal.as_slice()));
        let log_alpha = log_accept_ratio(current_lp, proposal_lp);
        let accept_prob = if log_alpha.is_sign_positive() {
            1.0
        } else {
            log_alpha.exp()
        };

        let accepted = rng.random::<f64>() < accept_prob;

        if accepted {
            state
                .position_mut()
                .copy_from_slice(self.proposal.as_slice());
            state.set_log_prob(proposal_lp);
        }

        (accepted, accept_prob)
    }

    pub fn sample<D, R>(&mut self, target: &D, state: &mut S, rng: &mut R) -> Draws
    where
        D: LogDensity,
        R: Rng + ?Sized,
    {
        debug_assert_eq!(state.dim(), self.dim);

        if !state.log_prob.is_finite() {
            state.initialize_log_prob(target);
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

impl<M, S, D> Kernel<D> for Rwmh<M, S>
where
    M: Metric,
    S: LogProbState,
    D: LogDensity,
{
    type State = S;

    fn initialize(&mut self, state: &mut Self::State, target: &D) {
        if !state.log_prob().is_finite() {
            state.initialize_log_prob(target);
        }
    }

    fn step<R: Rng + ?Sized>(&mut self, state: &mut Self::State, target: &D, rng: &mut R) -> bool {
        let (accepted, _) = self.step_impl(state, target, rng);
        accepted
    }
}
