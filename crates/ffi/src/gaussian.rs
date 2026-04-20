use crate::ffi;
use kernels::density::{GradLogDensity, LogDensity};

#[derive(Debug, Clone, Copy, Default)]
pub struct Gaussian;

impl LogDensity for Gaussian {
    fn log_prob(&self, x: &[f64]) -> f64 {
        ffi::gaussian_log_prob(x)
    }
}

impl GradLogDensity for Gaussian {
    fn grad_log_prob(&self, x: &[f64], grad: &mut [f64]) {
        ffi::gaussian_grad(x, grad)
    }
}
