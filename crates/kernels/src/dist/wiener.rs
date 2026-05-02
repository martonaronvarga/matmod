use crate::density::{FusedLogDensity, GradLogDensity, LogDensity};
use crate::dist::traits::Family;
use std::f64::consts::{PI, TAU};

// Constants

const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7; // 0.5 * ln(2π)
const SQRT_TWO_PI: f64 = 2.506_628_274_631_000_2;

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Domain Types
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// Which absorbing boundary terminated evidence accumulation.
///
/// Convention: `Upper` ≡ correct response (drift favours upper boundary
/// when `v > 0`); `Lower` ≡ error response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// Process hit the upper boundary at `a`.
    Upper,
    /// Process hit the lower boundary at `0`.
    Lower,
}

/// Wiener type data for scalar interfaces
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WienerObservation {
    pub rt: f64,
    pub boundary: Boundary,
}

/// Canonical Wiener first-passage families in Stan's convention:
/// Wiener4 = standard four parameter (alpha, tau, beta, delta) wiener family
/// Wiener5 = Wiener4 + variance parameter for drift (delta)
/// Wiener7 = Wiener5 + variance parameters for beta and tau
/// Reference:
/// https://mc-stan.org/docs/functions-reference/positive_lower-bounded_distributions.html#wiener-first-passage-time-distribution
struct Wiener4;
struct Wiener5;
struct Wiener7;

/// Canonical Wiener first-passage parameters in Stan's naming:
/// alpha: boundary separation, alpha \in R^+
/// tau: non-decision time, tau \in R^+
/// beta: relative starting point, beta \in (0, 1)
/// delta: drift rate, delta \in R
#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd, Display)]
pub struct Wiener4Params {
    /// Boundary separation (> 0). Larger -> slower, more accurate decisions
    pub alpha: f64,
    /// Non-decision time in seconds (> 0). Minimum possible RT
    pub tau: f64,
    /// Relative starting point `point / a ∈ (0, 1)`. `0.5` = unbiased
    pub beta: f64,
    /// Drift rate. Positive values bias the process toward the upper boundary
    pub delta: f64,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Wiener4Grad {
    pub alpha: f64,
    pub tau: f64,
    pub beta: f64,
    pub delta: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener4Eval {
    pub log_prob: f64,
    pub grad: Wiener4Grad,
}

#[derive(Copy, Clone, Debug)]
struct Wiener4Core {
    t: f64,
    a: f64,
    t_prime: f64,
    w_eff: f64,
    v_eff: f64,
    pref: f64,
    log_eps_eff: f64,
    beta_sign: f64,
    delta_sign: f64,
}

/// s_delta = standard deviation in drift rate, s_delta \in R^>=0
#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd, Display)]
pub struct Wiener5Params {
    base: Wiener4Params,
    s_delta: f64,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Wiener5Grad {
    pub alpha: f64,
    pub tau: f64,
    pub beta: f64,
    pub delta: f64,
    pub s_delta: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener5Eval {
    pub log_prob: f64,
    pub grad: Wiener5Grad,
}

#[derive(Copy, Clone, Debug)]
struct Wiener5Core {
    base: Wiener4Core,
    sv: f64,
    sv2: f64,
    lam: f64,
}

/// s_beta: standard deviation of beta
/// s_tau: standard deviation of tau
#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd, Display)]
pub struct Wiener7Params {
    base: Wiener5Params,
    s_beta: f64,
    s_tau: f64,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Wiener7Grad {
    pub alpha: f64,
    pub tau: f64,
    pub beta: f64,
    pub delta: f64,
    pub s_delta: f64,
    pub s_beta: f64,
    pub s_tau: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener7Eval {
    pub log_prob: f64,
    pub grad: Wiener7Grad,
}

#[derive(Copy, Clone, Debug)]
struct Wiener7Core {
    base: Wiener5Core,
    //TODO,
}

#[derive(Copy, Clone, Debug)]
struct SeriesEval {
    log_series: f64,
    dlog_dtprime: f64,
    dlog_dw: f64,
}

impl Wiener4Params {
    /// Default constructor
    /// Calls Default::default()
    /// Returns with default float parameters, invalid for direct use
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructor
    /// Use this to create a correct, validated parameter object
    /// with supplied parameter values
    #[inline]
    pub fn with_params(alpha: f64, tau: f64, beta: f64, delta: f64) -> Result<Self, Error> {
        if alpha.is_finite()
            && tau.is_finite()
            && beta.is_finite()
            && delta.is_finite()
            && alpha > 0.0
            && tau >= 0.0
            && 0_f64 < beta
            && 1_f64 > beta
        {
            Ok(Self {
                alpha,
                tau,
                beta,
                delta,
            })
        } else {
            Err("bad params")
        }
    }

    /// Constructor
    /// Same as with_params() but without validations
    #[inline]
    pub fn with_params_unchecked(alpha: f64, tau: f64, beta: f64, delta: f64) -> Self {
        Self {
            alpha,
            tau,
            beta,
            delta,
        }
    }
    /// Decision time for a given raw RT. Returns `None` if `rt ≤ tau`.
    #[inline]
    pub fn decision_time(&self, rt: f64) -> Option<f64> {
        let t = rt - self.tau;
        if t > 0.0 && t.is_finite() {
            Some(t)
        } else {
            None
        }
    }
    #[inline]
    pub fn valid(&self) -> bool {
        if self.alpha.is_finite()
            && self.tau.is_finite()
            && self.beta.is_finite()
            && self.delta.is_finite()
            && self.alpha > 0.0
            && self.tau >= 0.0
            && 0_f64 < self.beta
            && 1_f64 > self.beta
        {
            true
        } else {
            false
        }
    }
}

impl Wiener5Params {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline]
    pub fn with_params(
        alpha: f64,
        tau: f64,
        beta: f64,
        delta: f64,
        s_delta: f64,
    ) -> Result<Self, Error> {
        let base = Wiener4Params::with_params(alpha, tau, beta, delta)?;

        if s_delta.is_finite() {
            Ok(Self { base, s_delta })
        } else {
            Err("bad params")
        }
    }

    #[inline]
    pub fn with_params_unchecked(
        alpha: f64,
        tau: f64,
        beta: f64,
        delta: f64,
        s_delta: f64,
    ) -> Self {
        let base = Wiener4Params::with_params_unchecked(alpha, tau, beta, delta);
        Self { base, s_delta }
    }
}

impl Wiener7Params {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn with_params(
        alpha: f64,
        tau: f64,
        beta: f64,
        delta: f64,
        s_beta: f64,
        s_tau: f64,
        s_delta: f64,
    ) -> Result<Self, Error> {
        let base = Wiener5::with_params(alpha, tau, beta, delta, s_delta)?;
        if (0.0 <= s_beta && 1.0 > s_beta) && (s_tau.is_finite() && s_tau >= 0_f64) {
            Ok(Self {
                base,
                s_beta,
                s_tau,
            })
        } else {
            Err("bad params")
        }
    }

    #[inline]
    pub fn with_params_unchecked(
        alpha: f64,
        tau: f64,
        beta: f64,
        delta: f64,
        s_beta: f64,
        s_tau: f64,
        s_delta: f64,
    ) -> Self {
        let base = Wiener5Params::with_params_unchecked(alpha, tau, beta, delta, s_delta);
        Self {
            base,
            s_beta,
            s_tau,
        }
    }
}

impl Wiener4 {
    /// Large-time truncation count from the Navarro/Gondan style bound
    /// This is the count for the π-series
    ///
    /// t_prime = (y - tau) / alpha^2
    #[inline]
    pub fn k_l(t_prime: f64, log_eps: f64) -> usize {
        if !(t_prime.is_finite() && log_eps.is_finite()) || t_prime <= 0.0 || log_eps >= 0.0 {
            return 1;
        }
        let log_x = PI.ln() + t_prime.ln() + log_eps;

        let term1 = if log_x < 0.0 {
            let v = -2.0 * log_x / (PI * PI * t_prime);
            if v > 0.0 {
                v.sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        let term2 = 1.0 / (PI * t_prime.sqrt());

        term1.max(term2).ceil().max(1.0) as usize
    }

    /// Small-time truncation count.
    /// This is the tighter Gondan-style bound
    #[inline]
    pub fn k_s(t_prime: f64, beta: f64, log_eps: f64) -> usize {
        if !(t_prime.is_finite() && beta.is_finite() && log_eps.is_finite()) || t_prime <= 0.0 {
            return 0;
        }
        let w = 1.0 - beta;

        // u_eps = min(-1, ln(2pi t'^2 eps^2))
        let u_eps = (TAU.ln() + 2.0 * t_prime.ln() + 2.0 * log_eps).min(-1.0);
        let term1 = 0.5 * ((2.0 * t_prime).sqrt() - w);

        let term2 = {
            let inner = -2.0 * u_eps - 2.0;
            if inner > 0.0 {
                let arg = -t_prime * (u_eps - inner.sqrt());
                if arg > 0.0 {
                    0.5 * (arg.sqrt() - w)
                } else {
                    f64::NEG_INFINITY
                }
            } else {
                f64::NEG_INFINITY
            }
        };

        term1.max(term2).ceil().max(0.0) as usize
    }

    // /// Numerically stable log(exp(a) + exp(b))
    // #[inline(always)]
    // fn logsumexp(a: f64, b: f64) -> f64 {
    //     if a.is_infinite() && a.is_sign_negative() {
    //         return b;
    //     }
    //     if b.is_infinite() && b.is_sign_negative() {
    //         return a;
    //     }
    //     let m = a.max(b);
    //     m + ((a - m).exp() + (b - m).exp()).ln()
    // }

    // #[inline(always)]
    // fn logdiffexp(a: f64, b: f64) -> Option<f64> {
    //     if !(a.is_finite() && b.is_finite()) || a <= b {
    //         return None;
    //     }
    //     Some(a + (-(b - a).exp()).ln_1p())
    // }

    // #[inline]
    // fn normal_log_pdf(x: f64) -> f64 {
    //     -0.5 * x * x - LOG_SQRT_2PI
    // }

    /// Returns log(Mill's ratio) for the given x.
    // #[inline]
    // pub fn log_mill(x: f64) -> f64 {
    //     if !x.is_finite() {
    //         return f64::NAN;
    //     }

    //     // Exact central/moderate-range evaluation.
    //     // This is stable for negative x too.
    //     if x < 8.0 {
    //         let ccdf = 0.5 * libm::erfc(x / SQRT_2);
    //         if ccdf > 0.0 {
    //             return ccdf.ln() - Self::normal_log_pdf(x);
    //         }
    //     }

    //     // Tail asymptotic:
    //     // R(x) = Φ̄(x)/φ(x) ~ (1/x) * (1 - 1/x² + 3/x⁴ - 15/x⁶ + ...)
    //     let inv = 1.0 / x;
    //     let inv2 = inv * inv;
    //     let poly = 1.0 - inv2 + 3.0 * inv2 * inv2 - 15.0 * inv2 * inv2 * inv2;

    //     -x.ln() + poly.ln()
    // }

    // /// Returns exp(2 * v * y) - exp(2 * v * x).
    // pub fn phi1(x: f64, y: f64, v: f64) -> f64 {
    //     (2.0 * v * y).exp() - (2.0 * v * x).exp()
    // }

    #[inline]
    pub fn small_time_log_series(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 || w <= 0.0 || w >= 1.0 {
            return None;
        }

        let one_m_w = 1.0 - w;
        let inv_two_t = 0.5 / t_prime;

        let max_exponent = (one_m_w * one_m_w) * inv_two_t;
        // Factored out the dominant exponent, so the j=0 term is just `w * exp(0) = w`
        let mut sum = one_m_w;

        for j in 1..=k {
            let jf = j as f64;
            let xp = one_m_w + 2.0 * jf;
            let xm = 2.0 * jf - one_m_w;
            let xp2 = xp * xp;
            let xm2 = xm * xm;

            // Subtract max_exponent algebraically to prevent underflow
            let arg_p = (xp2) * inv_two_t - max_exponent;
            let arg_m = (xm2) * inv_two_t - max_exponent;

            sum += xp * (-arg_p).exp();
            sum -= xm * (-arg_m).exp();
        }

        if sum > 0.0 {
            // Re-apply the factored log_max outside the logarithm
            let log_pref = -0.5 * TAU.ln() - 1.5 * t_prime.ln();
            // Note: max_exponent is subtracted because we factored out exp(-max_exponent)
            Some(log_pref - max_exponent + sum.ln())
        } else {
            None
        }
    }

    /// Scaled raw small-time sum:
    /// R_s(t', w) = exp(w^2 / (2t')) * S(t', w)
    ///
    /// This is fine as an intermediate for gradients, but then the chain rule
    /// must include the missing factor derivatives.
    #[inline]
    pub fn small_time_series_raw(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 {
            return None;
        }
        let one_m_w = 1.0 - w;
        let inv_two_t = 0.5 / t_prime;

        let mut sum = one_m_w;

        for j in 1..=k {
            let jf = j as f64;
            let xp = one_m_w + 2.0 * jf;
            let xm = 2.0 * jf - one_m_w;

            let xp2 = xp * xp;
            let xm2 = xm * xm;

            let arg_p = (xp2 - one_m_w * one_m_w) * inv_two_t;
            let arg_m = (xm2 - one_m_w * one_m_w) * inv_two_t;

            sum += xp * (-arg_p).exp();
            sum -= xm * (-arg_m).exp();
        }

        Some(sum)
    }

    /// d/dt' of the *scaled* raw small-time sum R_s(t', w).
    /// This is not d/dt' log S.
    #[inline]
    pub fn small_time_log_dseries_dt(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 {
            return None;
        }
        let one_m_w = 1.0 - w;
        let inv_two_t = 0.5 / t_prime;
        let tt = t_prime * t_prime;

        let mut sum = 0.0;

        for j in 1..=k {
            let jf = j as f64;

            let xp = one_m_w + 2.0 * jf;
            let xm = 2.0 * jf - one_m_w;

            let xp2 = xp * xp;
            let xm2 = xm * xm;

            let arg_p = (xp2 - one_m_w * one_m_w) * inv_two_t;
            let arg_m = (xm2 - one_m_w * one_m_w) * inv_two_t;

            let exp_p = (-arg_p).exp();
            let exp_m = (-arg_m).exp();

            sum += 0.5 * xp * (xp2 - one_m_w * one_m_w) * exp_p / tt;
            sum -= 0.5 * xm * (xm2 - one_m_w * one_m_w) * exp_m / tt;
        }

        Some(sum)
    }

    /// d/dw of the scaled raw small-time sum R_s(t', w).
    #[inline]
    pub fn small_time_log_dseries_dw(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 {
            return None;
        }
        let one_m_w = 1.0 - w;
        let inv_two_t = 0.5 / t_prime;

        let mut sum = 1.0; // d/dw of the j=0 term in the scaled representation

        for j in 1..=k {
            let jf = j as f64;

            let xp = one_m_w + 2.0 * jf;
            let xm = 2.0 * jf - one_m_w;

            let xp2 = xp * xp;
            let xm2 = xm * xm;

            let arg_p = (xp2 - one_m_w * one_m_w) * inv_two_t;
            let arg_m = (xm2 - one_m_w * one_m_w) * inv_two_t;

            let exp_p = (-arg_p).exp();
            let exp_m = (-arg_m).exp();

            // derivative of x * exp(-(x^2 - w^2)/(2t'))
            sum += exp_p * (1.0 - (2.0 * jf * xp) / t_prime);
            sum += exp_m * (1.0 - (2.0 * jf * xm) / t_prime);
        }

        Some(sum)
    }

    #[inline]
    pub fn large_time_log_series(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 {
            return None;
        }

        let pi2_t_half = (PI * PI * t_prime) * 0.5;
        // The dominant term's exponent (j = 1)
        let max_exponent = pi2_t_half;

        let mut sum = 0.0;

        for j in 1..=k {
            let jf = j as f64;
            let s = (jf * PI * (1.0 - w)).sin();
            if s == 0.0 {
                continue;
            }

            // Subtract max_exponent: (j^2 * pi^2 * t / 2) - (pi^2 * t / 2) = (j^2 - 1) * max_exponent
            let arg = (jf * jf - 1.0) * max_exponent;
            sum += jf * s * (-arg).exp();
        }

        if sum > 0.0 {
            // Re-apply the factored log_max
            Some(PI.ln() - max_exponent + sum.ln())
        } else {
            None
        }
    }

    /// Scaled raw large-time sum:
    /// R_l(t', w) = exp(π² t'/2) * S(t', w)
    #[inline]
    pub fn large_time_series_raw(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 {
            return None;
        }

        let pi2_half = 0.5 * PI * PI;
        let mut sum = 0.0;

        for j in 1..=k {
            let jf = j as f64;
            let s = (jf * PI * (1.0 - w)).sin();
            if s == 0.0 {
                continue;
            }
            let arg = (jf * jf * pi2_half) * t_prime;
            sum += jf * s * (-arg).exp();
        }

        Some(sum)
    }

    /// d/dt' of the scaled raw large-time sum R_l(t', w).
    #[inline]
    pub fn large_time_log_dseries_dt(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 {
            return None;
        }

        let pi2 = PI * PI;
        let pi2_half = 0.5 * pi2;
        let mut sum = 0.0;

        for j in 1..=k {
            let jf = j as f64;
            let s = (jf * PI * (1.0 - w)).sin();
            if s == 0.0 {
                continue;
            }
            let arg = (jf * jf * pi2_half) * t_prime;
            let exp_term = (-arg).exp();

            sum -= 0.5 * pi2 * jf * jf * jf * s * exp_term;
        }

        Some(sum)
    }

    /// d/dw of the scaled raw large-time sum R_l(t', w).
    #[inline]
    pub fn large_time_log_dseries_dw(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 {
            return None;
        }

        let pi2_half = 0.5 * PI * PI;
        let mut sum = 0.0;

        for j in 1..=k {
            let jf = j as f64;
            let c = (jf * PI * (1.0 - w)).cos();
            let arg = (jf * jf * pi2_half) * t_prime;
            let exp_term = (-arg).exp();

            sum += jf * jf * PI * c * exp_term;
        }

        Some(sum)
    }

    #[inline]
    fn core(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        eps: f64,
    ) -> Option<Wiener4Core> {
        if !params.valid() || !obs.is_finite() || !eps.is_finite() || eps < 0.0 {
            return None;
        }
        let t = y - params.tau;

        if t <= 0.0 {
            return None;
        }

        let a = params.alpha;
        let a2 = a * a;
        let t_prime = t / a2;
        let (v_eff, w_eff, beta_sign, delta_sign) = match obs.boundary {
            Boundary::Upper => (params.delta, params.beta, 1.0, 1.0),
            Boundary::Lower => (-params.delta, 1.0 - params.beta, -1.0, -1.0),
        };

        let pref = -2.0 * a.ln() - a * v_eff * w_eff - 0.5 * v_eff * v_eff * t;
        let log_eps_eff = eps.ln() - pref;

        Some(Wiener4Core {
            t,
            a,
            t_prime,
            w_eff,
            v_eff,
            pref,
            log_eps_eff,
            beta_sign,
            delta_sign,
        })
    }

    #[inline]
    fn trunc_counts(core: &Wiener4Core) -> (usize, usize) {
        let kl = Self::k_l(core.t_prime, core.log_eps_eff);
        let ks = Self::k_s(core.t_prime, core.w_eff, core.log_eps_eff);
        (ks, kl)
    }

    #[inline]
    fn eval_series(core: &Wiener4Core) -> Option<f64> {
        let (ks, kl) = Self::trunc_counts(core);

        if ks < kl {
            Self::small_time_log_series(core.t_prime, core.w_eff, ks)
        } else {
            Self::large_time_log_series(core.t_prime, core.w_eff, kl)
        }
    }

    #[inline]
    fn eval_fused(core: &Wiener4Core) -> Option<SeriesEval> {
        let (ks, kl) = Self::trunc_counts(core);

        if ks < kl {
            let log_series = Self::small_time_log_series(core.t_prime, core.w_eff, ks)?;
            let raw = Self::small_time_series_raw(core.t_prime, core.w_eff, ks)?;
            if raw <= 0.0 {
                return None;
            }

            let d_raw_dt = Self::small_time_log_dseries_dt(core.t_prime, core.w_eff, ks)?;
            let d_raw_dw = Self::small_time_log_dseries_dw(core.t_prime, core.w_eff, ks)?;

            Some(SeriesEval {
                log_series,
                dlog_dtprime: -1.5 / core.t_prime + d_raw_dt / raw,
                dlog_dw: d_raw_dw / raw,
            })
        } else {
            let log_series = Self::large_time_log_series(core.t_prime, core.w_eff, kl)?;
            let raw = Self::large_time_series_raw(core.t_prime, core.w_eff, kl)?;
            if raw <= 0.0 {
                return None;
            }

            let d_raw_dt = Self::large_time_log_dseries_dt(core.t_prime, core.w_eff, kl)?;
            let d_raw_dw = Self::large_time_log_dseries_dw(core.t_prime, core.w_eff, kl)?;

            Some(SeriesEval {
                log_series,
                dlog_dtprime: -0.5 * PI * PI + d_raw_dt / raw,
                dlog_dw: d_raw_dw / raw,
            })
        }
    }

    /// Upper-bound first-passage-time log density for the 4-parameter Wiener model
    ///
    /// Parameters:
    /// * `y`     - random variable (observed reaction time)
    /// * `tau`   – non-decision time (> 0)
    /// * `beta`  – relative starting point (in (0,1))
    /// * `delta` – drift rate
    /// * `alpha` – boundary separation (> 0)
    /// * `eps`   – truncation tolerance
    /// supply as `Wiener4Params`
    /// Returns `f64::NEG_INFINITY` on invalid inputs or numerically degenerate
    /// series values
    #[inline]
    pub fn log_prob(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        eps: f64,
    ) -> Wiener4Eval {
        let core = match self.core(obs, params, eps) {
            Some(c) => c,
            None => {
                return Wiener4Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener4Grad {
                        alpha: f64::NAN,
                        tau: f64::NAN,
                        beta: f64::NAN,
                        delta: f64::NAN,
                    },
                }
            }
        };

        match Self::eval_series(&core) {
            Some(ls) => Wiener4Eval {
                log_prob: core.pref + ls,
                grad: Wiener4Grad {
                    alpha: f64::NAN,
                    tau: f64::NAN,
                    beta: f64::NAN,
                    delta: f64::NAN,
                },
            },
            None => {
                return Wiener4Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener4Grad {
                        alpha: f64::NAN,
                        tau: f64::NAN,
                        beta: f64::NAN,
                        delta: f64::NAN,
                    },
                }
            }
        }
    }

    #[inline]
    pub fn fused(&self, obs: &WienerObservation, params: &Wiener4Params, eps: f64) -> Wiener4Eval {
        let core = match self.core(obs, params, eps) {
            Some(c) => c,
            None => {
                return Wiener4Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener4Grad::default(),
                }
            }
        };

        let series = match Self::eval_fused(&core) {
            Some(s) => s,
            None => {
                return Wiener4Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener4Grad::default(),
                }
            }
        };

        let log_prob = core.pref + series.log_series;

        let a = core.a;
        let t = core.t;
        let a2 = a * a;

        let grad = Wiener4Grad {
            alpha: d_pref_da + series.dlog_dtprime * (-2.0 * t / (a * a2)),
            tau: -(d_pref_dt + series.dlog_dtprime / a2),
            beta: core.beta_sign * (d_pref_dw + series.dlog_dw),
            delta: core.delta_sign * d_pref_dv,
        };

        Wiener4Eval { log_prob, grad }
    }
}

impl FusedLogDensity for Target<Wiener4, [WienerObservation]> {
    type Gradient = [f64; 4];
    fn log_prob_and_grad(&self, theta: &Self::Point, grad: &mut [f64]) -> f64 {
        let mut total_lp = 0.0;
        grad.fill(0.0);

        for obs in &self.data {
            let fused = Wiener4.fused(obs, theta, 1e-12);
            let (lp, obs_grad) = (fused.log_prob, fused.grad);
            total_lp += lp;
            for i in 0..=4 {
                grad[i] += obs_grad[i]
            }
        }
        total_lp
    }
}

impl Wiener5 {
    #[inline]
    fn core(
        &self,
        obs: &WienerObservation,
        params: &Wiener5Params,
        eps: f64,
    ) -> Option<Wiener5Core> {
        if !params.base.valid() || params.s_delta < 0.0 || !obs.rt.is_finite() || eps <= 0.0 {
            return None;
        }

        let t = obs.rt - params.base.tau;
        if t <= 0.0 {
            return None;
        }

        let a = params.base.alpha;
        let a2 = a * a;
        let t_prime = t / a2;

        let (v_eff, w_eff, beta_sign, delta_sign) = match obs.boundary {
            Boundary::Upper => (params.base.delta, params.base.beta, 1.0, 1.0),
            Boundary::Lower => (-params.base.delta, 1.0 - params.base.beta, -1.0, -1.0),
        };

        let sv = params.var_delta;
        let sv2 = sv * sv;
        let lam = 1.0 + sv2 * t;

        let pref = (sv2 * a2 * w_eff * w_eff - 2.0 * a * v_eff * w_eff - v_eff * v_eff * t)
            / (2.0 * lam)
            - 0.5 * lam.ln()
            - 2.0 * a.ln();

        let log_eps_eff = eps.ln() - pref;

        Some(Wiener5Core {
            base: Wiener4Core {
                t,
                a,
                t_prime,
                w_eff,
                v_eff,
                pref,
                log_eps_eff,
                beta_sign,
                delta_sign,
            },
            sv,
            sv2,
            lam,
        })
    }

    #[inline]
    pub fn log_prob(
        &self,
        obs: &WienerObservation,
        params: &Wiener5Params,
        eps: f64,
    ) -> Wiener5Eval {
        let core = match self.core(obs, params, eps) {
            Some(x) => x,
            None => {
                return Wiener5Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener5Grad {
                        alpha: f64::NAN,
                        tau: f64::NAN,
                        beta: f64::NAN,
                        delta: f64::NAN,
                        s_delta: f64::NAN,
                    },
                }
            }
        };
        let series = match Wiener4::eval_series(&core.base) {
            Some(s) => s,
            None => {
                return Wiener5Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener5Grad {
                        alpha: f64::NAN,
                        tau: f64::NAN,
                        beta: f64::NAN,
                        delta: f64::NAN,
                        s_delta: f64::NAN,
                    },
                }
            }
        };

        Wiener5Eval {
            log_prob: core.base.pref + series,
            grad: Wiener5Grad {
                alpha: f64::NAN,
                beta: f64::NAN,
                tau: f64::NAN,
                delta: f64::NAN,
                s_delta: f64::NAN,
            },
        }
    }

    #[inline]
    pub fn fused(&self, obs: &WienerObservation, params: &Wiener5Params, eps: f64) -> Wiener5Eval {
        let core = match self.core(obs, params, eps) {
            Some(c) => c,
            None => {
                return Wiener5Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener5Grad {
                        alpha: f64::NEG_INFINITY,
                        tau: f64::NEG_INFINITY,
                        beta: f64::NEG_INFINITY,
                        delta: f64::NEG_INFINITY,
                        s_delta: f64::NEG_INFINITY,
                    },
                }
            }
        };

        let series = match Wiener4::eval_fused(&core.base) {
            Some(s) => s,
            None => {
                return Wiener5Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener5Grad {
                        alpha: f64::NEG_INFINITY,
                        tau: f64::NEG_INFINITY,
                        beta: f64::NEG_INFINITY,
                        delta: f64::NEG_INFINITY,
                        s_delta: f64::NEG_INFINITY,
                    },
                }
            }
        };

        let b = &core.base;
        let a = b.a;
        let t = b.t;
        let w = b.w_eff;
        let v = b.v_eff;
        let sv = core.sv;
        let sv2 = core.sv2;
        let lam = core.lam;
        let lam2 = lam * lam;
        let a2 = a * a;

        let log_prob = b.pref + series.log_series;

        let d_pref_da = (sv2 * a * w * w - v * w) / lam - 2.0 / a;
        let d_pref_dt = ((-v * v) * lam - (sv2 * a2 * w * w - 2.0 * a * v * w - v * v * t) * sv2)
            / (2.0 * lam2)
            - 0.5 * sv2 / lam;
        let d_pref_dw = (sv2 * a2 * w - a * v) / lam;
        let d_pref_dv = -(a * w + v * t) / lam;
        let d_pref_dsv =
            sv * (a2 * w * w + 2.0 * a * t * v * w + v * v * t * t - t - sv2 * t * t) / lam2;

        let grad = Wiener5Grad {
            alpha: d_pref_da + series.dlog_dtprime * (-2.0 * t / (a * a2)),
            tau: -((-0.5 * v * v) + d_pref_dt + series.dlog_dtprime / a2),
            beta: b.beta_sign * (d_pref_dw + series.dlog_dw),
            delta: b.delta_sign * d_pref_dv,
            s_delta: d_pref_dsv,
        };

        Wiener5Eval { log_prob, grad }
    }
}

impl FusedLogDensity for Target<Wiener5, [WienerObservation]> {
    type Gradient = [f64; 5];

    fn log_prob_and_grad(&self, p: &Wiener5Params, grad: &mut Self::Gradient) -> f64 {
        let mut total_lp = 0.0;
        grad.fill(0.0);

        for obs in self.data.iter() {
            let fused = Wiener5.fused(obs, p, 1e-6);
            let (lp, g) = (fused.log_prob, fused.grad);
            total_lp += lp;
            grad[0] += g.alpha;
            grad[1] += g.tau;
            grad[2] += g.beta;
            grad[3] += g.delta;
            grad[4] += g.s_delta;
        }

        total_lp
    }
}

impl Wiener7 {
    /// Pre-computed 5-point Gauss-Legendre nodes and weights (interval [-1, 1])
    /// https://pomax.github.io/bezierinfo/legendre-gauss.html#n21
    const GL_NODES: [f64; 21] = [
        0.0000000000000000,
        -0.1455618541608951,
        0.1455618541608951,
        -0.2880213168024011,
        0.2880213168024011,
        -0.4243421202074388,
        0.4243421202074388,
        -0.5516188358872198,
        0.5516188358872198,
        -0.6671388041974123,
        0.6671388041974123,
        -0.7684399634756779,
        0.7684399634756779,
        -0.8533633645833173,
        0.8533633645833173,
        -0.9200993341504008,
        0.9200993341504008,
        -0.9672268385663063,
        0.9672268385663063,
        -0.9937521706203895,
        0.9937521706203895,
    ];
    const GL_WEIGHTS: [f64; 21] = [
        0.1460811336496904,
        0.1445244039899700,
        0.1445244039899700,
        0.1398873947910731,
        0.1398873947910731,
        0.1322689386333375,
        0.1322689386333375,
        0.1218314160537285,
        0.1218314160537285,
        0.1087972991671484,
        0.1087972991671484,
        0.0934444234560339,
        0.0934444234560339,
        0.0761001136283793,
        0.0761001136283793,
        0.0571344254268572,
        0.0571344254268572,
        0.0369537897708525,
        0.0369537897708525,
        0.0160172282577743,
        0.0160172282577743,
    ];

    /// 7-Parameter density via numerical integration
    #[inline]
    pub fn fused(obs: &WienerObservation, params: &Wiener7Params, eps: f64) -> (f64, [f64; 7]) {
        let mut total_density = 0.0;
        let mut grad_density = [0.0; 7];

        let st_half = params.s_tau / 2.0;
        let sw_half = params.s_beta / 2.0;

        // 2D Gauss-Legendre Quadrature over uniform distributions of tau and beta
        for (x_tau, w_tau) in Self::GL_NODES.iter().zip(Self::GL_WEIGHTS.iter()) {
            let current_tau = params.base.base.tau + x_tau * st_half;
            if obs.rt <= current_tau {
                continue;
            }

            for (x_beta, w_beta) in Self::GL_NODES.iter().zip(Self::GL_WEIGHTS.iter()) {
                let current_beta = params.base.base.beta + x_beta * sw_half;

                let p5 = Wiener5Params::with_params_unchecked(
                    params.base.base.alpha,
                    current_tau,
                    current_beta,
                    params.base.base.delta,
                    params.base.s_delta,
                );

                let fused = Wiener5.fused(obs, &p5, eps);
                let (log_pdf, grad5) = (fused.log_prob, fused.grad);

                if log_pdf.is_finite() {
                    let pdf = log_pdf.exp();
                    let weight = w_tau * w_beta;
                    let weighted_pdf = pdf * weight;

                    total_density += weighted_pdf;

                    // Chain rule mapping from inner 5-param gradients to 7-param gradients
                    grad_density[0] += weighted_pdf * grad5[0]; // d_alpha
                    grad_density[1] += weighted_pdf * grad5[1]; // d_tau
                    grad_density[2] += weighted_pdf * grad5[2]; // d_beta
                    grad_density[3] += weighted_pdf * grad5[3]; // d_delta
                    grad_density[4] += weighted_pdf * grad5[4]; // d_s_delta

                    // Gradients w.r.t the uniform interval widths (sw, st)
                    grad_density[5] += weighted_pdf * grad5[2] * (x_beta / 2.0); // d_s_beta
                    grad_density[6] += weighted_pdf * grad5[1] * (x_tau / 2.0); // d_s_tau
                }
            }
        }

        let norm = (params.s_tau * params.s_beta).max(f64::EPSILON);
        total_density /= norm;

        let mut final_grad = [0.0; 7];
        if total_density > 0.0 {
            // Apply log-derivative trick: d/dx log(f(x)) = f'(x) / f(x)
            for i in 0..7 {
                final_grad[i] = (grad_density[i] / norm) / total_density;
            }
            // Add gradient of the `-log(var_tau * var_beta)` normalization factor
            final_grad[5] -= 1.0 / params.s_beta.max(f64::EPSILON);
            final_grad[6] -= 1.0 / params.s_tau.max(f64::EPSILON);

            (total_density.ln(), final_grad)
        } else {
            (f64::NEG_INFINITY, [0.0; 7])
        }
    }
}

impl Family for Wiener5 {
    type Params = Wiener5Params;
    type Data = [WienerObservation]; // Dataset batch

    fn log_prob(params: &Self::Params, data: &Self::Data) -> f64 {
        let eps = 1e-6; // Could be passed in via a context or config
        data.iter()
            .map(|obs| {
                let lp = Wiener5.log_prob(obs, params, eps);
                lp.log_prob
            })
            .sum()
    }
}

// --- 2. Parameter Transformation (with Jacobian) ---
// In HMC, you sample in unconstrained space (R^N). You must add the
// log-determinant of the Jacobian to the target density.
impl Parameter for Wiener5Params {
    type Constrained = Self;
    type Unconstrained = [f64; 5];

    fn to_unconstrained(c: &Self::Constrained) -> Self::Unconstrained {
        [
            c.base.alpha.ln(),
            c.base.tau.ln(),
            crate::numeric::logit(c.base.beta),
            c.base.delta, // Drift is already unconstrained (R)
            c.var_delta.ln(),
        ]
    }

    fn from_unconstrained(u: &Self::Unconstrained) -> Self::Constrained {
        Wiener5Params::with_params_unchecked(
            u[0].exp(),
            u[1].exp(),
            crate::numeric::sigmoid(u[2]),
            u[3],
            u[4].exp(),
        )
    }

    // Add this to your Parameter trait!
    fn log_abs_det_jacobian(u: &Self::Unconstrained) -> f64 {
        let log_jac_alpha = u[0];
        let log_jac_tau = u[1];
        let log_jac_beta = crate::numeric::log_sigmoid(u[2]) + crate::numeric::log1m_sigmoid(u[2]);
        let log_jac_var_delta = u[4];

        log_jac_alpha + log_jac_tau + log_jac_beta + log_jac_var_delta
    }
}

// --- 3. Fused Gradients (Performance Critical for HMC) ---
// Instead of using std::autodiff, implement the analytical gradients here.
impl GradLogDensity for Target<Wiener5, [WienerObservation]> {
    type Gradient = [f64; 5];

    fn grad_log_prob(&self, x: &Wiener5Params, grad: &mut Self::Gradient) {
        // Shell: In production, translate `dxdwiener` from C++ here.
        // Accumulate analytical gradients over self.data.
    }
}
