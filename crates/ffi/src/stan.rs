#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("ffi/include/kernel.hpp");

        fn log_prob(x: &Vec<f64>) -> f64;
        fn grad_log_prob(x: &Vec<f64>, grad: &mut Vec<f64>);
    }
}

pub struct CppKernel;

impl kernels::density::LogDensity for CppKernel {
    fn log_prob(&self, x: &[f64]) -> f64 {
        ffi::log_prob(&x.to_vec())
    }
}

impl kernels::density::Gradient for CppKernel {
    fn grad_log_prob(&self, x: &[f64], grad: &mut [f64]) {
        let mut g = grad.to_vec();
        ffi::grad_log_prob(&x.to_vec(), &mut g);
        grad.copy_from_slice(&g);
    }
}
