//! Compatibility configuration controlling exact numerical conventions across MHW/MCS tools.

use crate::quantile::QuantileMethod;

/// How February 29 (DOY 60 in 1-based 366-day leap-aligned calendars) is handled during climatology computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feb29Mode {
    /// Compute DOYs != 60 from calendar windows, then set DOY 60 = mean(DOY 59, DOY 61) before smoothing.
    /// Matches m_mhw1.0 and standard Hobday convention.
    Interp5961,
    /// Compute DOY 60 directly from its own window samples (used for all_leap or non-366 calendars).
    WindowDirect,
    /// Expand non-leap years by inserting 0.5*(Feb 28 + Mar 1) at DOY 60 before building sliding windows,
    /// then compute all 366 DOYs (matches heatwaveR ts2clm).
    HeatwaveRSeries,
    /// Build a (n_years, 366) calendar matrix with non-leap Feb 29 = 0.5*(Feb 28 + Mar 1) and circularly
    /// wrap the +/- window_half_width across DOY 1..366 (matches ecjoliver/marineHeatWaves).
    OliverMatrix,
}

/// How NaNs are treated inside the circular moving-mean smoothing window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothNanPolicy {
    /// Average over finite values in the window; NaN only if the entire window is NaN (MATLAB smoothdata default).
    OmitNan,
    /// Propagate NaN if any value inside the smoothing window is NaN.
    PropagateNan,
}

/// Policy for enforcing a physical lower bound (e.g., seawater freezing point -1.8 C or salinity 0.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundPolicy {
    /// Detect normally, guard division-by-zero in category calculation, and flag affected events.
    Flag,
    /// Mask out days where climatology or lower threshold hits/falls below the lower bound.
    Mask,
    /// Clamp values and lower threshold to >= lower_bound before detection, and flag affected events.
    Floor,
}

/// Event detection direction selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventMode {
    /// Upper extremes only (e.g., MHW).
    High,
    /// Lower extremes only (e.g., MCS).
    Low,
    /// Both upper and lower extremes in one pass.
    Both,
}

/// Boundary behavior for moving baselines when the window extends beyond data start/end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MovingBoundary {
    /// Shift window inward to keep full window_years width wherever possible.
    Clamp,
    /// Clip window to available years (width shrinks near start/end).
    Shrink,
}

/// Detrending polynomial order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetrendOrder {
    None,
    Linear,
    Quadratic,
}

/// Full configuration controlling numerical compatibility with reference tools.
#[derive(Debug, Clone, Copy)]
pub struct CompatConfig {
    pub quantile_method: QuantileMethod,
    pub strict_inequality: bool,
    pub feb29_mode: Feb29Mode,
    pub smooth_nan_policy: SmoothNanPolicy,
    /// Degrees of freedom for variance/std (1 = sample, 0 = population).
    pub var_ddof: usize,
}

impl Default for CompatConfig {
    fn default() -> Self {
        Self {
            quantile_method: QuantileMethod::Type7,
            strict_inequality: true,
            feb29_mode: Feb29Mode::Interp5961,
            smooth_nan_policy: SmoothNanPolicy::OmitNan,
            var_ddof: 1,
        }
    }
}
