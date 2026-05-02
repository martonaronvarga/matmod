pub trait LogDensity {
    type Point;
    fn log_prob(&self, x: &Self::Point) -> f64;
}

pub trait GradLogDensity: LogDensity {
    type Gradient;
    fn grad_log_prob(&self, x: &Self::Point, grad: &mut Self::Gradient);
}

pub trait FusedLogDensity: GradLogDensity {
    type Gradient;
    fn log_prob_and_grad(&self, x: &Self::Point, grad: &mut Self::Gradient) -> f64;
}

pub trait HessianLogDensity: GradLogDensity {
    type Hessian;
    fn hessian(&self, x: &Self::Point, h: &mut Self::Hessian);
}
