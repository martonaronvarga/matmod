pub trait LogDensity {
    fn log_prob(&self, x: &[f64]) -> f64;
}

pub trait Gradient: LogDensity {
    fn grad_log_prob(&self, x: &[f64], grad: &mut [f64]);
}
