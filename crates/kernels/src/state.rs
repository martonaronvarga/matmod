use crate::{buffer::OwnedBuffer, density::LogDensity};

pub struct State {
    pub position: OwnedBuffer,
    pub gradient: OwnedBuffer,
    pub momentum: OwnedBuffer,
    pub log_prob: f64,
}

impl State {
    pub fn new(dim: usize) -> Self {
        Self {
            position: OwnedBuffer::new(dim),
            gradient: OwnedBuffer::new(dim),
            momentum: OwnedBuffer::new(dim),
            log_prob: f64::NAN,
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.position.len()
    }

    #[inline]
    pub fn initialize_log_prob<D: LogDensity>(&mut self, density: &D) {
        self.log_prob = density.log_prob(self.position.as_slice());
    }
}
