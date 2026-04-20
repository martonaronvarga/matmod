#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("gaussian.hpp");
        fn gaussian_log_prob(x: &[f64]) -> f64;
        fn gaussian_grad(x: &[f64], grad: &mut [f64]);
    }
}

pub mod ddm;
pub mod gaussian;
// pub mod stan;
