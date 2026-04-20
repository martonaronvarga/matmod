use kernels::{buffer::OwnedBuffer, metric::CholeskyFactor};
use rand::Rng;
use rand_distr::StandardNormal;

#[cfg(feature = "openblas")]
use cblas::{dtrmv, Diagonal, Layout, Part, Transpose};

#[inline(always)]
pub fn fill_normals<R: Rng + ?Sized>(dst: &mut [f64], rng: &mut R) {
    for v in dst.iter_mut() {
        *v = rng.sample::<f64, _>(StandardNormal);
    }
}

pub trait Proposal {
    fn dim(&self) -> usize;

    fn propose_into<R: Rng + ?Sized>(
        &mut self,
        x: &[f64],
        out: &mut [f64],
        step_size: f64,
        rng: &mut R,
    );
}

#[derive(Debug)]
pub struct IsotropicProposal {
    dim: usize,
}

impl IsotropicProposal {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    #[inline(always)]
    fn propose_scalar<R: Rng + ?Sized>(
        &mut self,
        x: &[f64],
        out: &mut [f64],
        step_size: f64,
        rng: &mut R,
    ) {
        fill_normals(out, rng);
        for i in 0..self.dim {
            out[i] = x[i] + step_size * out[i];
        }
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    fn propose_simd<R: Rng + ?Sized>(
        &mut self,
        x: &[f64],
        out: &mut [f64],
        step_size: f64,
        rng: &mut R,
    ) {
        use std::simd::prelude::*;
        use std::simd::Simd;

        const LANES: usize = 8;
        fill_normals(out, rng);

        let step = Simd::<f64, LANES>::splat(step_size);
        let mut i = 0;
        while i + LANES <= self.dim {
            let xv = Simd::<f64, LANES>::from_slice(&x[i..i + LANES]);
            let zv = Simd::<f64, LANES>::from_slice(&out[i..i + LANES]);
            (xv + step * zv).copy_to_slice(&mut out[i..i + LANES]);
            i += LANES;
        }

        while i < self.dim {
            out[i] = x[i] + step_size * out[i];
            i += 1;
        }
    }
}

impl Proposal for IsotropicProposal {
    #[inline(always)]
    fn dim(&self) -> usize {
        self.dim
    }

    #[inline(always)]
    fn propose_into<R: Rng + ?Sized>(
        &mut self,
        x: &[f64],
        out: &mut [f64],
        step_size: f64,
        rng: &mut R,
    ) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);

        #[cfg(feature = "simd")]
        {
            self.propose_simd(x, out, step_size, rng);
        }

        #[cfg(not(feature = "simd"))]
        {
            self.propose_scalar(x, out, step_size, rng);
        }
    }
}

#[derive(Debug)]
pub struct DenseCholeskyProposal {
    factor: CholeskyFactor,
    z: OwnedBuffer,
}

impl DenseCholeskyProposal {
    pub fn new(factor: CholeskyFactor) -> Self {
        let dim = factor.dim();
        Self {
            factor,
            z: OwnedBuffer::new(dim),
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.factor.dim()
    }

    #[inline(always)]
    fn propose_scalar<R: Rng + ?Sized>(
        &mut self,
        x: &[f64],
        out: &mut [f64],
        step_size: f64,
        rng: &mut R,
    ) {
        fill_normals(self.z.as_mut_slice(), rng);

        out.copy_from_slice(x);

        let z = self.z.as_slice();
        let chol = self.factor.as_slice();
        let dim = self.factor.dim();

        for j in 0..dim {
            let zj = step_size * z[j];
            let col = &chol[j * dim..j * dim + dim];
            for i in j..dim {
                out[i] += col[i] * zj;
            }
        }
    }

    #[cfg(feature = "openblas")]
    #[inline(always)]
    fn propose_blas<R: Rng + ?Sized>(
        &mut self,
        x: &[f64],
        out: &mut [f64],
        step_size: f64,
        rng: &mut R,
    ) {
        fill_normals(self.z.as_mut_slice(), rng);
        out.copy_from_slice(x);

        let chol = self.factor.as_slice();
        let z = self.z.as_mut_slice();

        unsafe {
            dtrmv(
                Layout::ColumnMajor,
                Part::Lower,
                Transpose::None,
                Diagonal::Generic,
                self.factor.dim() as i32,
                chol,
                self.factor.dim() as i32,
                z,
                1,
            )
        };

        for i in 0..self.factor.dim() {
            out[i] += step_size * z[i];
        }
    }
}

impl Proposal for DenseCholeskyProposal {
    #[inline(always)]
    fn dim(&self) -> usize {
        self.factor.dim()
    }

    #[inline(always)]
    fn propose_into<R: Rng + ?Sized>(
        &mut self,
        x: &[f64],
        out: &mut [f64],
        step_size: f64,
        rng: &mut R,
    ) {
        debug_assert_eq!(x.len(), self.factor.dim());
        debug_assert_eq!(out.len(), self.factor.dim());

        #[cfg(feature = "openblas")]
        {
            self.propose_blas(x, out, step_size, rng);
        }

        #[cfg(not(feature = "openblas"))]
        {
            self.propose_scalar(x, out, step_size, rng);
        }
    }
}
