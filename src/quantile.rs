//! Quantile computation supporting both R/NumPy Type-7 and MATLAB Type-5 (Hazen) definitions.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantileMethod {
    /// Hyndman & Fan Type 7 (linear interpolation on (i)/(n-1)): NumPy default, R default, heatwaveR, marineHeatWaves, m_mhw 'python'.
    Type7,
    /// Hyndman & Fan Type 5 (Hazen, linear interpolation on (i+0.5)/n): MATLAB quantile(), m_mhw default 'matlab'.
    Type5,
}

/// Compute quantile `q` in [0, 1] from a **sorted** slice of finite f64 values.
#[inline]
pub fn quantile_sorted(sorted: &[f64], q: f64, method: QuantileMethod) -> f64 {
    let n = sorted.len();
    if n == 0 || !q.is_finite() {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    match method {
        // R/NumPy default: linear interpolation at index q * (n - 1).
        QuantileMethod::Type7 => {
            let pos = (n - 1) as f64 * q;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = pos - lo as f64;
            sorted[lo] + frac * (sorted[hi] - sorted[lo])
        }
        // MATLAB Hazen rule: piecewise-linear on midpoint grid (i + 0.5) / n.
        QuantileMethod::Type5 => {
            let nf = n as f64;
            let p_min = 0.5 / nf;
            let p_max = (nf - 0.5) / nf;
            if q <= p_min {
                return sorted[0];
            }
            if q >= p_max {
                return sorted[n - 1];
            }
            let pos = (q * nf - 0.5).clamp(0.0, (n - 1) as f64);
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = pos - lo as f64;
            sorted[lo] + frac * (sorted[hi] - sorted[lo])
        }
    }
}

/// Sort a scratch buffer of finite f64 values in-place (NaNs must already be excluded).
#[inline]
pub fn sort_finite(buf: &mut [f64]) {
    buf.sort_unstable_by(|a, b| a.total_cmp(b));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type7_matches_numpy() {
        let mut v = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        sort_finite(&mut v);
        assert!((quantile_sorted(&v, 0.9, QuantileMethod::Type7) - 46.0).abs() < 1e-12);
        assert!((quantile_sorted(&v, 0.1, QuantileMethod::Type7) - 14.0).abs() < 1e-12);
        assert!((quantile_sorted(&v, 0.5, QuantileMethod::Type7) - 30.0).abs() < 1e-12);
    }

    #[test]
    fn test_type5_matches_matlab() {
        // MATLAB quantile([10 20 30 40 50], 0.9) == 50.0 (since 0.9 == (5-0.5)/5)
        // MATLAB quantile([10 20 30 40 50], 0.8) == 45.0 (pos = 0.8*5 - 0.5 = 3.5 -> avg of 40 and 50)
        let mut v = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        sort_finite(&mut v);
        assert!((quantile_sorted(&v, 0.9, QuantileMethod::Type5) - 50.0).abs() < 1e-12);
        assert!((quantile_sorted(&v, 0.8, QuantileMethod::Type5) - 45.0).abs() < 1e-12);
        assert!((quantile_sorted(&v, 0.1, QuantileMethod::Type5) - 10.0).abs() < 1e-12);
        assert!((quantile_sorted(&v, 0.2, QuantileMethod::Type5) - 15.0).abs() < 1e-12);
    }
}
