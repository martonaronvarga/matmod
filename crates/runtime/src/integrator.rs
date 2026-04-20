pub struct Leapfrog<M> {
    pub metric: M,
    pub step_size: f64,
}

impl<M> Leapfrog<M>
where
    M: Metric,
{
    pub fn step<D, S>(&self, state: &mut S, target: &D)
    where
        D: GradLogDensity,
        S: GradientState + MomentumState,
    {
        let eps = self.step_size;

        // p ← p + (ε/2) ∇ log π(q)
        for (p, g) in state.momentum_mut().iter_mut().zip(state.gradient()) {
            *p += 0.5 * eps * g;
        }

        // q ← q + ε M^{-1} p
        let mut velocity = vec![0.0; state.dim()].as_mut_slice();
        self.metric.apply_inverse(state.momentum(), &mut velocity);

        for (q, v) in state.position_mut().iter_mut().zip(&velocity) {
            *q += eps * v;
        }

        // recompute gradient
        target.grad_log_prob(state.position(), state.gradient_mut());

        // p ← p + (ε/2) ∇ log π(q)
        for (p, g) in state.momentum_mut().iter_mut().zip(state.gradient()) {
            *p += 0.5 * eps * g;
        }
    }
}
