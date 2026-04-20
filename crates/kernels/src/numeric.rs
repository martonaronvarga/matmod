#[inline(always)]
pub fn positive_finite(x: f64) -> f64 {
    x.max(f64::MIN_POSITIVE)
}

#[inline(always)]
pub fn finite_or_neg_inf(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        f64::NEG_INFINITY
    }
}

#[inline(always)]
pub fn log_accept_ratio(current: f64, proposal: f64) -> f64 {
    if proposal.is_finite() {
        proposal - current
    } else {
        f64::NEG_INFINITY
    }
}
