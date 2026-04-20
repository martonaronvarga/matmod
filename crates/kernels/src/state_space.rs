use rand::Rng;

pub trait TransitionModel<S> {
    fn log_transition(&self, prev: &S, next: &S, t: usize) -> f64;

    fn sample_next<R: Rng + ?Sized>(&self, prev: &S, rng: &mut R) -> S;
}

pub trait ObservationModel<S, Y> {
    fn log_likelihood(&self, state: &S, obs: &Y) -> f64;
}
