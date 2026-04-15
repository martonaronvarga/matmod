use kernels::buffer::OwnedBuffer;
use rand::{Rng, RngExt};
use rand_distr::StandardNormal;

#[cfg(feature = "openblas")]
use cblas::{dtrmv, Diagonal, Layout, Part, Transpose};

#[inline(always)]
pub fn fill_normals<R: Rng + ?Sized>(dst: &mut [f64], rng: &mut R) {
    for v in dst.iter_mut() {
        *v = rng.sample(StandardNormal);
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
    z: OwnedBuffer,
}

impl IsotropicProposal {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            z: OwnedBuffer::new(dim),
        }
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

        let z = self.z.as_slice();
        for i in 0..self.dim {
            out[i] = x[i] + step_size * z[i];
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

        fill_normals(self.z.as_mut_slice(), rng);

        let z = self.z.as_slice();
        let step = Simd::<f64, LANES>::splat(step_size);

        let mut i = 0;
        while i + LANES <= self.dim {
            let xv = Simd::<f64, LANES>::from_slice(&x[i..i + LANES]);
            let zv = Simd::<f64, LANES>::from_slice(&z[i..i + LANES]);
            (xv + step * zv).copy_to_slice(&mut out[i..i + LANES]);
            i += LANES;
        }

        while i < self.dim {
            out[i] = x[i] + step_size * z[i];
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
    dim: usize,
    /// Column-major lower-triangular factor. Only entries with row >= col are read.
    chol: OwnedBuffer,
    z: OwnedBuffer,
}

impl DenseCholeskyProposal {
    pub fn new(dim: usize, chol: OwnedBuffer) -> Self {
        let expected = dim.checked_mul(dim).expect("dimension overflow");
        assert_eq!(chol.len(), expected, "Cholesky factor must be dim × dim");

        #[cfg(debug_assertions)]
        {
            let diag = chol.as_slice();
            for i in 0..dim {
                let d = diag[i * dim + i];
                assert!(
                    d.is_finite() && d > 0.0,
                    "Cholesky diagonal must be positive"
                );
            }
        }

        Self {
            dim,
            chol,
            z: OwnedBuffer::new(dim),
        }
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
        let chol = self.chol.as_slice();

        // out = x + step_size * L * z
        // L is stored column-major, lower-triangular.
        for j in 0..self.dim {
            let zj = step_size * z[j];
            let col = &chol[j * self.dim..j * self.dim + self.dim];
            for i in j..self.dim {
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

        let chol = self.chol.as_slice();
        let z = self.z.as_mut_slice();

        unsafe {
            dtrmv(
                Layout::ColumnMajor,
                Part::Lower,
                Transpose::None,
                Diagonal::Generic,
                self.dim as i32,
                chol,
                self.dim as i32,
                z,
                1,
            )
        };

        for i in 0..self.dim {
            out[i] += step_size * z[i];
        }
    }
}

impl Proposal for DenseCholeskyProposal {
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
