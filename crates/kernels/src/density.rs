pub trait LogDensity {
    fn log_prob(&self, x: &[f64]) -> f64;
}

pub trait GradLogDensity: LogDensity {
    fn grad_log_prob(&self, x: &[f64], grad: &mut [f64]);
}

pub trait HessianLogDensity: GradLogDensity {
    fn hessian(&self, x: &[f64], h: &mut [f64]);
}

pub trait TransitionDensity<S> {
    fn log_transition(&self, prev: &S, next: &S, t: usize) -> f64;
}

pub trait ObservationDensity<S, Y> {
    fn log_likelihood(&self, state: &S, obs: &Y) -> f64;
}

// Enzyme:
// the differentiated region should be side-effect free;
// inputs and outputs should be explicitly separated, with no aliasing between primal input and adjoint output buffers;
// keep allocations and I/O outside the differentiated region;
// prefer straight-line or predictable control flow in the primal;

pub struct EnzymeDensity<F> {
    f: F,
    raw: std::ptr::NonNull<std::ffi::c_void>,
    dim: usize,
}

impl<F> LogDensity for EnzymeDensity<F>
where
    F: Fn(&[f64]) -> f64,
{
    #[inline(always)]
    fn log_prob(&self, x: &[f64]) -> f64 {
        (self.f)(x)
    }
}

impl<F> GradLogDensity for EnzymeDensity<F>
where
    F: Fn(&[f64]) -> f64,
{
    #[inline(always)]
    fn grad_log_prob(&self, x: &[f64], grad: &mut [f64]) {
        // enzyme_grad(&self.f, x, grad);
    }
}
