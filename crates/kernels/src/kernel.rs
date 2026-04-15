use crate::{
    density::{Gradient, LogDensity},
    state::State,
};
use rand::Rng;

pub trait TransitionKernel {
    fn initialize<D: LogDensity>(&mut self, state: &mut State, density: &D) {
        state.initialize_log_prob(density);
    }
    fn step<D: LogDensity, R: Rng + ?Sized>(
        &mut self,
        state: &mut State,
        density: &D,
        rng: &mut R,
    ) -> bool;
}

pub trait GradientKernel {
    fn initialize<D: Gradient>(&mut self, state: &mut State, density: &D) {
        state.initialize_log_prob(density);
    }
    fn step<D: Gradient>(&mut self, state: &mut State, density: &D, rng: &mut impl Rng) -> bool;
}

pub trait StateTransition {
    type S; // latent state
    fn log_transition(&self, s_prev: &Self::S, s_next: &Self::S, t: usize) -> f64;
    fn sample_next(&self, s_prev: &Self::S, rng: &mut impl Rng) -> Self::S;
}

pub trait Observation {
    type S;
    type Y;
    fn log_likelihood(&self, state: &Self::S, obs: &Self::Y) -> f64;
}
