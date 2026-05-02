use crate::ddm::wiener::{WienerFpt, WienerObservation, WienerParams};
use crate::state_space::ObservationModel;

#[derive(Debug, Clone, Copy)]
pub struct LatentState<Context, Latent> {
    pub context: Context,
    pub latent: Latent,
}

impl<Context, Latent> LatentState<Context, Latent> {
    #[inline]
    pub fn new(context: Context, latent: Latent) -> Self {
        Self { context, latent }
    }

    #[inline]
    pub fn into_parts(self) -> (Context, Latent) {
        (self.context, self.latent)
    }
}

pub trait LatentStateView {
    type Context;
    type Latent;

    fn context(&self) -> &Self::Context;
    fn latent(&self) -> &Self::Latent;
}

impl<Context, Latent> LatentStateView for LatentState<Context, Latent> {
    #[inline]
    fn context(&self) -> &Self::Context {
        &self.context
    }

    #[inline]
    fn latent(&self) -> &Self::Latent {
        &self.latent
    }
}

pub trait LatentParameterMap<Context, Latent> {
    fn write_params(&self, context: &Context, latent: &Latent, out: &mut WienerParams);
}

#[derive(Debug, Clone, Copy)]
pub struct ObservationLayer<M> {
    pub map: M,
    pub density: WienerFpt,
}

impl<M> ObservationLayer<M> {
    #[inline]
    pub fn new(map: M, density: WienerFpt) -> Self {
        Self { map, density }
    }
}

impl<Context, Latent, M> ObservationModel<LatentState<Context, Latent>, WienerObservation>
    for ObservationLayer<M>
where
    M: LatentParameterMap<Context, Latent>,
{
    #[inline]
    fn log_likelihood(
        &self,
        state: &DdmLatentState<Context, Latent>,
        obs: &WienerObservation,
    ) -> f64 {
        let mut params = WienerParams {
            alpha: 1.0,
            tau: 0.0,
            beta: 0.5,
            delta: 0.0,
        };
        self.map
            .write_params(&state.context, &state.latent, &mut params);
        self.density.log_pdf(obs, params)
    }
}

use crate::state_space::TransitionModel;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct TransitionLayer<T> {
    pub transition: T,
}

impl<T> TransitionLayer<T> {
    #[inline]
    pub fn new(transition: T) -> Self {
        Self { transition }
    }
}

impl<Context, Latent, T> TransitionModel<LatentState<Context, Latent>> for TransitionLayer<T>
where
    T: TransitionModel<LatentState<Context, Latent>>,
{
    #[inline]
    fn log_transition(
        &self,
        prev: &LatentState<Context, Latent>,
        next: &LatentState<Context, Latent>,
        t: usize,
    ) -> f64 {
        self.transition.log_transition(prev, next, t)
    }

    #[inline]
    fn sample_next<R: Rng + ?Sized>(
        &self,
        prev: &LatentState<Context, Latent>,
        rng: &mut R,
    ) -> LatentState<Context, Latent> {
        self.transition.sample_next(prev, rng)
    }
}

/// Alternative direction ->
///

/// Single-time latent state used by transition and observation models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatentFrame {
    pub c: f64,
    pub m: f64,
}

/// Canonical affine map from latent factors to unconstrained Wiener parameters.
///
/// The affine form is on the transformed scale:
///   η_alpha = b0 + b1 c_t + b2 m_t
///   η_tau   = b0 + b1 c_t + b2 m_t
///   η_beta  = b0 + b1 c_t + b2 m_t
///   δ       = b0 + b1 c_t + b2 m_t
///
/// Then:
///   alpha = exp(η_alpha), tau = exp(η_tau), beta = sigmoid(η_beta)
///   delta = η_delta
///
/// This keeps the semantics layer separate from the distribution layer.
#[derive(Debug, Clone, Copy)]
pub struct AffineLatentWienerMap {
    pub alpha: [f64; 3],
    pub tau: [f64; 3],
    pub beta: [f64; 3],
    pub delta: [f64; 3],
}

impl AffineLatentWienerMap {
    #[inline]
    pub fn new(alpha: [f64; 3], tau: [f64; 3], beta: [f64; 3], delta: [f64; 3]) -> Self {
        Self {
            alpha,
            tau,
            beta,
            delta,
        }
    }

    #[inline]
    fn dot(coeffs: &[f64; 3], c: f64, m: f64) -> f64 {
        coeffs[0] + coeffs[1] * c + coeffs[2] * m
    }

    #[inline]
    fn sigmoid(x: f64) -> f64 {
        if x >= 0.0 {
            let e = (-x).exp();
            1.0 / (1.0 + e)
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    /// Map one latent frame to one Wiener parameter tuple.
    #[inline]
    pub fn map_frame(&self, frame: LatentFrame) -> WienerParams {
        WienerParams {
            alpha: Self::dot(&self.alpha, frame.c, frame.m).exp(),
            tau: Self::dot(&self.tau, frame.c, frame.m).exp(),
            beta: Self::sigmoid(Self::dot(&self.beta, frame.c, frame.m)),
            delta: Self::dot(&self.delta, frame.c, frame.m),
        }
    }

    /// Fill a batch of parameters from SoA latent factors.
    #[inline]
    pub fn map_into(
        &self,
        latent: LatentFrameSoA<'_>,
        out_alpha: &mut [f64],
        out_tau: &mut [f64],
        out_beta: &mut [f64],
        out_delta: &mut [f64],
    ) -> bool {
        if !latent.validate() {
            return false;
        }
        let n = latent.len();
        if out_alpha.len() < n || out_tau.len() < n || out_beta.len() < n || out_delta.len() < n {
            return false;
        }

        for i in 0..n {
            let c = latent.c[i];
            let m = latent.m[i];
            out_alpha[i] = Self::dot(&self.alpha, c, m).exp();
            out_tau[i] = Self::dot(&self.tau, c, m).exp();
            out_beta[i] = Self::sigmoid(Self::dot(&self.beta, c, m));
            out_delta[i] = Self::dot(&self.delta, c, m);
        }
        true
    }

    /// Parallel path preserves the same SoA layout and can be sharded by the
    /// caller or by the orchestration layer.
    #[inline]
    pub fn map_into_parallel(
        &self,
        latent: LatentFrameSoA<'_>,
        out_alpha: &mut [f64],
        out_tau: &mut [f64],
        out_beta: &mut [f64],
        out_delta: &mut [f64],
    ) -> bool {
        self.map_into(latent, out_alpha, out_tau, out_beta, out_delta)
    }
}

/// A latent dynamics model that evolves `c_t` and `m_t`.
///
/// This is the structural layer that plugs into `TransitionModel`, while the
/// DDM/Wiener layer remains separate.
#[derive(Debug, Clone, Copy)]
pub struct LatentRandomWalk {
    pub sigma_c: f64,
    pub sigma_m: f64,
}

impl LatentRandomWalk {
    #[inline]
    pub fn new(sigma_c: f64, sigma_m: f64) -> Option<Self> {
        if !(sigma_c.is_finite() && sigma_c >= 0.0 && sigma_m.is_finite() && sigma_m >= 0.0) {
            return None;
        }
        Some(Self { sigma_c, sigma_m })
    }

    #[inline]
    fn gaussian_log_density(x: f64, mean: f64, sigma: f64) -> f64 {
        if !(sigma > 0.0) {
            return if (x - mean).abs() <= f64::EPSILON {
                0.0
            } else {
                f64::NEG_INFINITY
            };
        }
        let z = (x - mean) / sigma;
        -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }
}

impl TransitionModel<LatentFrame> for LatentRandomWalk {
    #[inline]
    fn log_transition(&self, prev: &LatentFrame, next: &LatentFrame, _t: usize) -> f64 {
        Self::gaussian_log_density(next.c, prev.c, self.sigma_c)
            + Self::gaussian_log_density(next.m, prev.m, self.sigma_m)
    }

    #[inline]
    fn sample_next<R: rand::Rng + ?Sized>(&self, prev: &LatentFrame, rng: &mut R) -> LatentFrame {
        use rand::Rng;
        let dc = if self.sigma_c > 0.0 {
            rng.random::<f64>() * self.sigma_c
        } else {
            0.0
        };
        let dm = if self.sigma_m > 0.0 {
            rng.random::<f64>() * self.sigma_m
        } else {
            0.0
        };
        LatentFrame {
            c: prev.c + dc,
            m: prev.m + dm,
        }
    }
}

/// Observation model that turns a latent frame into Wiener parameters and then
/// evaluates the exact first-passage density.
#[derive(Debug, Clone, Copy)]
pub struct LatentObservationModel {
    pub projection: AffineLatentWienerMap,
    pub distribution: WienerFpt,
}

impl LatentObservationModel {
    #[inline]
    pub fn new(projection: AffineLatentWienerMap, distribution: WienerFpt) -> Self {
        Self {
            projection,
            distribution,
        }
    }
}

impl ObservationModel<LatentFrame, WienerTrial> for LatentObservationModel {
    #[inline]
    fn log_likelihood(&self, state: &LatentFrame, obs: &WienerTrial) -> f64 {
        let params = self.projection.map_frame(*state);
        self.distribution.log_prob_trial(*obs, &params)
    }
}
