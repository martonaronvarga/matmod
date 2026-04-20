use crate::buffer::OwnedBuffer;
use faer::diag::Diag;
use faer::Mat;

pub trait Metric {
    fn dim(&self) -> usize;
    fn apply_inverse(&self, src: &[f64], dst: &mut [f64]); // M^{-1} v
    fn apply_sqrt(&self, src: &[f64], dst: &mut [f64]); // M^{1/2} v, useful for proposals / momentum
    fn log_det(&self) -> f64;
}

#[derive(Debug, Clone, Copy)]
pub struct IdentityMetric {
    dim: usize,
}

impl IdentityMetric {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Metric for IdentityMetric {
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }
    #[inline]
    fn apply_sqrt(&self, z: &[f64], out: &mut [f64]) {
        out.copy_from_slice(z);
    }

    fn apply_inverse(&self, x: &[f64], out: &mut [f64]) {
        out.copy_from_slice(x);
    }

    fn log_det(&self) -> f64 {
        0.0
    }
}

#[derive(Debug)]
pub struct DiagonalMetric {
    dim: usize,
    diag: OwnedBuffer,
}

impl DiagonalMetric {
    pub fn new(diag: OwnedBuffer) -> Self {
        let dim = diag.len();
        for &v in diag.iter() {}

        let mut diag = OwnedBuffer::new(dim);

        for i in 0..dim {
            let d = diag[i];
            assert!(
                positive_finite(d),
                "diagonal metric entries must be positive finite"
            );
            diag[i] = d;
        }

        Self { dim, diag }
    }
}

impl Metric for DiagonalMetric {
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    fn apply_sqrt(&self, z: &[f64], out: &mut [f64]) {
        for ((o, zi), di) in out.iter_mut().zip(z).zip(self.diag.iter()) {
            *o = zi * di.sqrt();
        }
    }

    fn apply_inverse(&self, x: &[f64], out: &mut [f64]) {
        for ((o, xi), di) in out.iter_mut().zip(x).zip(self.diag.iter()) {
            *o = xi / di;
        }
    }

    fn log_det(&self) -> f64 {
        self.diag.iter().map(|d| d.ln()).sum()
    }
}

#[derive(Debug)]
pub struct DenseMetric {
    dim: usize,
    pub factor: Mat<f64>,
}

impl DenseMetric {
    pub fn new(factor: Mat<f64>) -> Self {
        let dim = factor.nrows();
        assert_eq!(factor.ncols(), dim, "Cholesky factor must be square");

        #[cfg(debug_assertions)]
        for i in 0..dim {
            let diag = factor[(i, i)];
            assert!(
                diag.is_finite() && diag > 0.0,
                "Cholesky diagonal must be positive and finite"
            );
        }

        Self { dim, factor }
    }

    #[inline]
    pub fn factor(&self) -> &Mat<f64> {
        &self.factor
    }
}

impl Metric for DenseMetric {
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn apply_inverse(&self, x: &[f64], out: &mut [f64]) {
        let n = self.l.nrows();
        out.copy_from_slice(x);

        let mut y = faer::Col::from_slice_mut(out);

        // Solve L y = x
        faer::linalg::solve::triangular::solve_lower_triangular_in_place(
            &self.factor,
            &mut y,
            faer::Diag::NonUnit,
        );

        // Solve L^T y = y
        faer::linalg::solve::triangular::solve_lower_triangular_transpose_in_place(
            &self.factor,
            &mut y,
            faer::Diag::NonUnit,
        );
    }

    #[inline]
    fn apply_sqrt(&self, z: &[f64], out: &mut [f64]) {
        let n = self.l.nrows();
        debug_assert_eq!(z.len(), n);
        debug_assert_eq!(out.len(), n);

        let z = Col::from_slice(z);
        let mut out_col = Col::from_slice_mut(out);

        // out = L * z
        faer::linalg::matmul::triangular::matmul_lower_triangular(
            &self.l,
            z,
            &mut out_col,
            faer::Side::Left,
            faer::Diag::NonUnit,
            faer::Transpose::No,
        );
    }

    #[inline]
    fn log_det(&self) -> f64 {
        let mut acc = 0.0;
        for i in 0..self.l.nrows() {
            acc += self.l[(i, i)].ln();
        }
        2.0 * acc
    }
}
