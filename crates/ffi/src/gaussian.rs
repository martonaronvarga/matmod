use crate::ffi;

pub struct Gaussian;

impl kernels::density::LogDensity for Gaussian {
    fn log_prob(&self, x: &[f64]) -> f64 {
        ffi::gaussian_log_prob(x)
    }
}

impl kernels::density::Gradient for Gaussian {
    fn grad_log_prob(&self, x: &[f64], grad: &mut [f64]) {
        ffi::gaussian_grad(x, grad)
    }
}
