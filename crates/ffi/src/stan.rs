#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("ffi/include/kernel.hpp");

        fn log_prob(x: &Vec<f64>) -> f64;
        fn grad_log_prob(x: &Vec<f64>, grad: &mut Vec<f64>);
    }
}

use kernels::density::{GradLogDensity, LogDensity};

#[derive(Debug, Clone, Copy, Default)]
pub struct StanTarget;

impl LogDensity for StanTarget {
    fn log_prob(&self, x: &[f64]) -> f64 {
        let x_vec = x.to_vec();
        ffi::log_prob(&x_vec)
    }
}

impl GradLogDensity for StanTarget {
    fn grad_log_prob(&self, x: &[f64], grad: &mut [f64]) {
        let x_vec = x.to_vec();
        let mut tmp = grad.to_vec();
        ffi::grad_log_prob(&x_vec, &mut tmp);
        grad.copy_from_slice(&tmp);
    }
}
