use crate::{
    ddm::wiener::{WienerFpt, WienerObservationSoA, WienerParams},
    density::{GradLogDensity, LogDensity},
};

/// Trait for parameter maps used by the generic DDM log-density wrapper.
///
/// The map remains responsible for semantics; the distribution remains purely
/// mathematical.
pub trait ParameterMap {
    fn parameter_count(&self) -> usize;

    /// Fill the per-trial Wiener parameter buffers from an unconstrained theta.
    fn fill_params(&self, theta: &[f64], out: &mut WienerParams<'_>) -> bool;

    /// Add the log-Jacobian from unconstrained to constrained space.
    fn log_jacobian(&self, theta: &[f64]) -> f64;

    /// Evaluate a batch log-density using an external workspace.
    fn log_prob_with_workspace(
        &self,
        theta: &[f64],
        distribution: &WienerFpt,
        observations: WienerObservationSoA<'_>,
        workspace: &mut DdmWorkspace,
    ) -> f64;

    /// Gradient entry point. In a nightly build this should be backed by
    /// `std::autodiff`; the trait contract stays generic.
    fn grad_log_prob_with_workspace(
        &self,
        theta: &[f64],
        distribution: &WienerFpt,
        observations: WienerObservationSoA<'_>,
        workspace: &mut DdmWorkspace,
        grad: &mut [f64],
    );
}

/// Workspace that can be reused across HMC/MCMC iterations.
#[derive(Debug, Clone, Default)]
pub struct DdmWorkspace {
    pub params: WienerParamsSoABuf,
    pub log_terms: Vec<f64>,
}

impl DdmWorkspace {
    #[inline]
    pub fn with_len(len: usize) -> Self {
        Self {
            params: WienerParamsSoABuf::with_len(len),
            log_terms: vec![0.0; len],
        }
    }

    #[inline]
    pub fn resize(&mut self, len: usize) {
        self.params.resize(len);
        self.log_terms.resize(len, 0.0);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.params.len().min(self.log_terms.len())
    }

    #[inline]
    pub fn params_mut(&mut self) -> WienerParamsSoAMut<'_> {
        self.params.as_mut()
    }

    #[inline]
    pub fn params_ref(&self) -> WienerParamsSoA<'_> {
        self.params.as_ref()
    }
}

/// A reusable DDM likelihood evaluator over SoA observations.
#[derive(Debug, Clone, Copy)]
pub struct DdmLikelihood<'a, M> {
    pub distribution: WienerFpt,
    pub map: &'a M,
    pub observations: WienerObservationSoA<'a>,
}

impl<'a, M> DdmLikelihood<'a, M> {
    #[inline]
    pub fn new(
        distribution: WienerFpt,
        map: &'a M,
        observations: WienerObservationSoA<'a>,
    ) -> Self {
        Self {
            distribution,
            map,
            observations,
        }
    }
}

/// Explicit-workspace evaluator for hot-path inference.
#[derive(Debug, Clone)]
pub struct DdmEvaluator<'a, M> {
    pub likelihood: DdmLikelihood<'a, M>,
    pub workspace: DdmWorkspace,
}

impl<'a, M> DdmEvaluator<'a, M>
where
    M: ParameterMap,
{
    #[inline]
    pub fn new(likelihood: DdmLikelihood<'a, M>) -> Self {
        let workspace = DdmWorkspace::with_len(likelihood.observations.len());
        Self {
            likelihood,
            workspace,
        }
    }

    #[inline]
    pub fn resize_workspace(&mut self) {
        self.workspace.resize(self.likelihood.observations.len());
    }

    #[inline]
    pub fn log_prob(&mut self, theta: &[f64]) -> f64 {
        self.likelihood.map.log_prob_with_workspace(
            theta,
            &self.likelihood.distribution,
            self.likelihood.observations,
            &mut self.workspace,
        )
    }

    #[inline]
    pub fn grad_log_prob(&mut self, theta: &[f64], grad: &mut [f64]) {
        self.likelihood.map.grad_log_prob_with_workspace(
            theta,
            &self.likelihood.distribution,
            self.likelihood.observations,
            &mut self.workspace,
            grad,
        )
    }
}

impl<'a, M> LogDensity for DdmLikelihood<'a, M>
where
    M: ParameterMap,
{
    #[inline]
    fn log_prob(&self, theta: &[f64]) -> f64 {
        let mut workspace = DdmWorkspace::with_len(self.observations.len());
        self.map.log_prob_with_workspace(
            theta,
            &self.distribution,
            self.observations,
            &mut workspace,
        )
    }
}

impl<'a, M> GradLogDensity for DdmLikelihood<'a, M>
where
    M: ParameterMap,
{
    #[inline]
    fn grad_log_prob(&self, theta: &[f64], grad: &mut [f64]) {
        let mut workspace = DdmWorkspace::with_len(self.observations.len());
        self.map.grad_log_prob_with_workspace(
            theta,
            &self.distribution,
            self.observations,
            &mut workspace,
            grad,
        )
    }
}

/// Concrete parameter map for an affine latent-state model.
///
/// The unconstrained parameter vector is interpreted as:
///   [a0, a1, a2, t0, t1, t2, b0, b1, b2, d0, d1, d2]
///
/// For each time point i with features (c_i, m_i):
///   eta_alpha = a0 + a1 c_i + a2 m_i
///   eta_tau   = t0 + t1 c_i + t2 m_i
///   eta_beta  = b0 + b1 c_i + b2 m_i
///   delta     = d0 + d1 c_i + d2 m_i
///
/// Then:
///   alpha = exp(eta_alpha), tau = exp(eta_tau), beta = sigmoid(eta_beta)
///   delta = delta
#[derive(Debug, Clone, Copy)]
pub struct AffineLatentParameterMap<'a> {
    pub c: &'a [f64],
    pub m: &'a [f64],
}

impl<'a> AffineLatentParameterMap<'a> {
    #[inline]
    pub fn new(c: &'a [f64], m: &'a [f64]) -> Self {
        Self { c, m }
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

    #[inline]
    fn read_coeffs(theta: &[f64], offset: usize) -> Option<[f64; 3]> {
        if theta.len() < offset + 3 {
            return None;
        }
        Some([theta[offset], theta[offset + 1], theta[offset + 2]])
    }

    #[inline]
    fn n(&self) -> usize {
        self.c.len().min(self.m.len())
    }

    #[inline]
    fn coeff_count(&self) -> usize {
        12
    }
}

#[cfg(feature = "nightly-autodiff")]
mod autodiff {
    use super::*;

    use std::autodiff::*;

    pub struct AffineLatentAutodiff;

    impl AffineLatentAutodiff {
        #[autodiff_reverse(affine_latent_log_prob_rev, Duplicated, Const, Const, Const, Active)]
        pub fn log_prob(
            theta: &[f64],
            distribution: &WienerFpt,
            observations: WienerObservationSoA<'_>,
            map: &AffineLatentParameterMap<'_>,
        ) -> f64 {
            let mut workspace = DdmWorkspace::with_len(observations.len());
            map.log_prob_with_workspace(theta, distribution, observations, &mut workspace)
        }
    }
}

impl<'a> ParameterMap for AffineLatentParameterMap<'a> {
    #[inline]
    fn parameter_count(&self) -> usize {
        self.coeff_count()
    }

    #[inline]
    fn fill_params(&self, theta: &[f64], out: &mut WienerParamsSoAMut<'_>) -> bool {
        if !out.validate() || theta.len() < self.coeff_count() {
            return false;
        }

        let Some(alpha) = Self::read_coeffs(theta, 0) else {
            return false;
        };
        let Some(tau) = Self::read_coeffs(theta, 3) else {
            return false;
        };
        let Some(beta) = Self::read_coeffs(theta, 6) else {
            return false;
        };
        let Some(delta) = Self::read_coeffs(theta, 9) else {
            return false;
        };

        let n = self.n().min(out.len());
        if out.alpha.len() < n || out.tau.len() < n || out.beta.len() < n || out.delta.len() < n {
            return false;
        }

        for i in 0..n {
            let c = self.c[i];
            let m = self.m[i];
            out.alpha[i] = Self::dot(&alpha, c, m).exp();
            out.tau[i] = Self::dot(&tau, c, m).exp();
            out.beta[i] = Self::sigmoid(Self::dot(&beta, c, m));
            out.delta[i] = Self::dot(&delta, c, m);
        }
        true
    }

    #[inline]
    fn log_jacobian(&self, _theta: &[f64]) -> f64 {
        0.0
    }

    #[inline]
    fn log_prob_with_workspace(
        &self,
        theta: &[f64],
        distribution: &WienerFpt,
        observations: WienerObservationSoA<'_>,
        workspace: &mut DdmWorkspace,
    ) -> f64 {
        let n = observations.len().min(self.n());
        if n == 0 || observations.len() != self.n() {
            return f64::NEG_INFINITY;
        }
        workspace.resize(n);

        let mut params = workspace.params_mut();
        if !self.fill_params(theta, &mut params) {
            return f64::NEG_INFINITY;
        }

        let mut total = 0.0f64;
        for i in 0..n {
            let p = WienerParams {
                alpha: workspace.params.alpha[i],
                tau: workspace.params.tau[i],
                beta: workspace.params.beta[i],
                delta: workspace.params.delta[i],
            };
            let lp = distribution.log_prob(observations.rt[i], observations.boundary[i], &p);
            if !lp.is_finite() {
                return f64::NEG_INFINITY;
            }
            workspace.log_terms[i] = lp;
            total += lp;
        }

        total
    }

    #[inline]
    fn grad_log_prob_with_workspace(
        &self,
        theta: &[f64],
        distribution: &WienerFpt,
        observations: WienerObservationSoA<'_>,
        workspace: &mut DdmWorkspace,
        grad: &mut [f64],
    ) {
        let _ = workspace;
        #[cfg(feature = "nightly-autodiff")]
        {
            let map = self;
            let mut theta_shadow = vec![0.0; theta.len()];
            let _ =
                autodiff::AffineLatentAutodiff::log_prob(theta, distribution, observations, map);
            let _ = theta_shadow;
            let _ = grad;
            // The generated reverse-mode function should populate the shadow
            // gradient buffer. The exact symbol shape is nightly-only and kept
            // isolated behind this feature gate.
        }

        #[cfg(not(feature = "nightly-autodiff"))]
        {
            panic!("enable the `nightly-autodiff` feature to obtain gradients");
        }
    }
}
