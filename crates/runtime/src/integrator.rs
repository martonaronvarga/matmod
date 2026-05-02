use kernels::{density::GradLogDensity, metric::Metric, state::GradientState};

#[inline]
pub fn leapfrog_step<D, M, S>(
    metric: &M,
    step_size: f64,
    state: &mut S,
    target: &D,
    velocity: &mut [f64],
) where
    D: GradLogDensity,
    M: Metric,
    S: GradientState,
{
    let dim = state.dim();
    velocity.copy_from_slice(state.gradient());
    for i in 0..dim {
        state.momentum_mut()[i] += 0.5 * step_size * velocity[i];
    }
    metric.apply_inverse(state.momentum(), velocity);
    for (q, v) in state.position_mut().iter_mut().zip(velocity.iter()) {
        *q += step_size * v;
    }
    state.initialize_gradient(target);
    velocity.copy_from_slice(state.gradient());
    for i in 0..dim {
        state.momentum_mut()[i] += 0.5 * step_size * velocity[i];
    }
}
