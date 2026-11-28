//! Core per-pixel extreme event detection, gap merging, lower-bound handling, and metric computation.

use crate::compat::{BoundPolicy, CompatConfig, EventMode};

/// Sentinel value in int8 category arrays representing land or missing data.
pub const CATEGORY_MISSING: i8 = -128;

/// Single detected extreme event (high or low) for one pixel.
#[derive(Debug, Clone)]
pub struct EventRecord {
    /// Original flattened pixel index in the caller's batch/grid.
    pub pixel_idx: i64,
    /// +1 for high extreme (e.g., MHW / high salinity), -1 for low extreme (e.g., MCS / low salinity).
    pub event_type: i8,
    pub index_start: i32,
    pub index_peak: i32,
    pub index_end: i32,
    pub duration: i32,
    pub int_max: f64,
    pub int_mean: f64,
    pub int_std: f64,
    pub int_var: f64,
    pub int_cum: f64,
    pub int_max_rel_thresh: f64,
    pub int_mean_rel_thresh: f64,
    pub int_cum_rel_thresh: f64,
    pub rate_onset: f64,
    pub rate_decline: f64,
    pub category_max: i8,
    pub p_moderate: f64,
    pub p_strong: f64,
    pub p_severe: f64,
    pub p_extreme: f64,
    pub truncated_start: bool,
    pub truncated_end: bool,
    pub near_lower_bound: bool,
}

/// Detection parameters for one run.
#[derive(Debug, Clone, Copy)]
pub struct DetectParams {
    pub event_mode: EventMode,
    pub min_duration: usize,
    pub max_gap: usize,
    pub max_pad_length: usize,
    pub lower_bound: Option<f64>,
    pub bound_policy: BoundPolicy,
    pub compat: CompatConfig,
}

/// Linearly interpolate internal NaN gaps whose run length is `<= max_pad_length`.
/// Leading/trailing NaNs and gaps longer than `max_pad_length` remain NaN.
pub fn pad_short_gaps(values: &[f64], max_pad_length: usize, out: &mut Vec<f64>) {
    out.clear();
    out.extend_from_slice(values);
    if max_pad_length == 0 || values.len() < 3 {
        return;
    }
    let n = out.len();
    let mut i = 0usize;
    while i < n {
        if out[i].is_finite() {
            i += 1;
            continue;
        }
        let gap_start = i;
        while i < n && !out[i].is_finite() {
            i += 1;
        }
        let gap_end = i;
        let gap_len = gap_end - gap_start;
        if gap_len <= max_pad_length && gap_start > 0 && gap_end < n {
            let v_left = out[gap_start - 1];
            let v_right = out[gap_end];
            if v_left.is_finite() && v_right.is_finite() {
                let denom = (gap_len + 1) as f64;
                for k in 0..gap_len {
                    let frac = (k + 1) as f64 / denom;
                    out[gap_start + k] = v_left + frac * (v_right - v_left);
                }
            }
        }
    }
}

/// Unified day-level sample resolution applying lower-bound Mask/Floor policies consistently
/// across exceedance scanning, first/second-pass event statistics, and category grids.
#[inline]
fn resolve_day_sample(
    mut v: f64,
    mut s: f64,
    mut th: f64,
    lb_opt: Option<f64>,
    bound_policy: BoundPolicy,
) -> Option<(f64, f64, f64)> {
    if !(v.is_finite() && s.is_finite() && th.is_finite()) {
        return None;
    }
    if let Some(lb) = lb_opt {
        match bound_policy {
            BoundPolicy::Mask => {
                if s <= lb + 1e-9 || th <= lb + 1e-9 {
                    return None;
                }
            }
            BoundPolicy::Floor => {
                if v < lb {
                    v = lb;
                }
                if s < lb {
                    s = lb;
                }
                if th < lb {
                    th = lb;
                }
            }
            BoundPolicy::Flag => {}
        }
    }
    Some((v, s, th))
}

/// Apply lower-bound floor clamping to a single value when `BoundPolicy::Floor` is active.
#[inline]
fn floor_val(v: f64, lb_opt: Option<f64>, bound_policy: BoundPolicy) -> f64 {
    if bound_policy == BoundPolicy::Floor {
        if let Some(lb) = lb_opt {
            if v.is_finite() && v < lb {
                return lb;
            }
        }
    }
    v
}

/// Run detection on a single pixel, writing requested gridded outputs directly into caller slices.
#[allow(clippy::too_many_arguments)]
pub fn detect_pixel_with_curves(
    pixel_idx: i64,
    raw_values: &[f64],
    seas_t: &[f64],
    thresh_high_t: &[f64],
    thresh_low_t: &[f64],
    params: &DetectParams,
    pad_buf: &mut Vec<f64>,
    mut intensity_out: Option<&mut [f64]>,
    mut intensity_high_out: Option<&mut [f64]>,
    mut intensity_low_out: Option<&mut [f64]>,
    mut daily_cat_out: Option<&mut [i8]>,
    mut event_cat_out: Option<&mut [i8]>,
) -> Vec<EventRecord> {
    let n = raw_values.len();
    pad_short_gaps(raw_values, params.max_pad_length, pad_buf);

    let low_is_masked = |t: usize| -> bool {
        if params.bound_policy == BoundPolicy::Mask {
            if let Some(lb) = params.lower_bound {
                return seas_t[t] <= lb + 1e-9 || thresh_low_t[t] <= lb + 1e-9;
            }
        }
        false
    };
    let day_is_active = |t: usize| -> bool {
        pad_buf[t].is_finite()
            && seas_t[t].is_finite()
            && !(params.event_mode == EventMode::Low && low_is_masked(t))
    };

    if let Some(ref mut v) = intensity_out {
        for t in 0..n {
            v[t] = if day_is_active(t) { 0.0 } else { f64::NAN };
        }
    }
    if let Some(ref mut v) = intensity_high_out {
        for t in 0..n {
            v[t] = if pad_buf[t].is_finite() && seas_t[t].is_finite() { 0.0 } else { f64::NAN };
        }
    }
    if let Some(ref mut v) = intensity_low_out {
        for t in 0..n {
            v[t] = if pad_buf[t].is_finite() && seas_t[t].is_finite() && !low_is_masked(t) {
                0.0
            } else {
                f64::NAN
            };
        }
    }
    if let Some(ref mut v) = daily_cat_out {
        for t in 0..n {
            v[t] = if day_is_active(t) { 0 } else { CATEGORY_MISSING };
        }
    }
    if let Some(ref mut v) = event_cat_out {
        for t in 0..n {
            v[t] = if day_is_active(t) { 0 } else { CATEGORY_MISSING };
        }
    }

    let mut events = Vec::new();

    if matches!(params.event_mode, EventMode::High | EventMode::Both) {
        detect_direction(
            pixel_idx,
            pad_buf,
            seas_t,
            thresh_high_t,
            true,
            params,
            &mut events,
            intensity_out.as_deref_mut(),
            intensity_high_out.as_deref_mut(),
            daily_cat_out.as_deref_mut(),
            event_cat_out.as_deref_mut(),
        );
    }

    if matches!(params.event_mode, EventMode::Low | EventMode::Both) {
        detect_direction(
            pixel_idx,
            pad_buf,
            seas_t,
            thresh_low_t,
            false,
            params,
            &mut events,
            intensity_out.as_deref_mut(),
            intensity_low_out.as_deref_mut(),
            daily_cat_out.as_deref_mut(),
            event_cat_out.as_deref_mut(),
        );
    }

    if params.event_mode == EventMode::Both && events.len() > 1 {
        events.sort_unstable_by_key(|e| (e.index_start, -e.event_type));
    }

    events
}

/// Detect events along one direction (`is_high=true` for MHW/high, `false` for MCS/low).
#[allow(clippy::too_many_arguments)]
fn detect_direction(
    pixel_idx: i64,
    values: &[f64],
    seas_t: &[f64],
    thresh_t: &[f64],
    is_high: bool,
    params: &DetectParams,
    out_events: &mut Vec<EventRecord>,
    mut intensity_ts: Option<&mut [f64]>,
    mut dir_intensity_ts: Option<&mut [f64]>,
    mut daily_cat_ts: Option<&mut [i8]>,
    mut event_cat_ts: Option<&mut [i8]>,
) {
    let n = values.len();
    if n == 0 || params.min_duration == 0 {
        return;
    }

    let strict = params.compat.strict_inequality;
    let lb_opt = if !is_high { params.lower_bound } else { None };
    let bound_policy = params.bound_policy;

    // 1. Scan exceedance runs and filter by min_duration.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut in_run = false;
    let mut run_start = 0usize;

    for t in 0..n {
        let exceed = if let Some((v, _, th)) =
            resolve_day_sample(values[t], seas_t[t], thresh_t[t], lb_opt, bound_policy)
        {
            if is_high {
                if strict { v > th } else { v >= th }
            } else if strict {
                v < th
            } else {
                v <= th
            }
        } else {
            false
        };

        if exceed && !in_run {
            in_run = true;
            run_start = t;
        } else if !exceed && in_run {
            let run_end = t - 1;
            if run_end - run_start + 1 >= params.min_duration {
                runs.push((run_start, run_end));
            }
            in_run = false;
        }
    }
    if in_run {
        let run_end = n - 1;
        if run_end - run_start + 1 >= params.min_duration {
            runs.push((run_start, run_end));
        }
    }

    if runs.is_empty() {
        return;
    }

    // 2. Transitive gap merging in O(K).
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    let mut cur = runs[0];
    for &(ns, ne) in runs.iter().skip(1) {
        let gap = ns - cur.1 - 1;
        if gap <= params.max_gap {
            cur.1 = ne;
        } else {
            merged.push(cur);
            cur = (ns, ne);
        }
    }
    merged.push(cur);

    // 3. Compute metrics per merged event.
    let sign = if is_high { 1.0f64 } else { -1.0f64 };
    let sign_i8: i8 = if is_high { 1 } else { -1 };
    let ddof = params.compat.var_ddof;

    for (ev_start, ev_end) in merged {
        let dur = ev_end - ev_start + 1;

        let mut peak_idx = ev_start;
        let mut best_mag = f64::NEG_INFINITY;
        let mut int_max = f64::NAN;

        let mut best_rel_mag = f64::NEG_INFINITY;
        let mut int_max_rel = f64::NAN;

        let mut sum_anom = 0.0f64;
        let mut sum_rel = 0.0f64;
        let mut valid_n = 0usize;

        let mut cat_max: i8 = 1;
        let mut c_mod = 0usize;
        let mut c_str = 0usize;
        let mut c_sev = 0usize;
        let mut c_ext = 0usize;
        let mut near_lb = false;

        // First pass: accumulate sums, extremum, categories, and write daily grids.
        for t in ev_start..=ev_end {
            let raw_v = values[t];
            let raw_th = thresh_t[t];
            if let Some(lb) = lb_opt {
                if (raw_v.is_finite() && raw_v <= lb + 0.05)
                    || (raw_th.is_finite() && raw_th <= lb + 0.05)
                {
                    near_lb = true;
                }
            }
            let Some((v, s, th)) =
                resolve_day_sample(raw_v, seas_t[t], raw_th, lb_opt, bound_policy)
            else {
                continue;
            };

            let anom = v - s;
            let rel = v - th;
            let delta = th - s;

            if delta.abs() <= 1e-6 && lb_opt.is_some() {
                near_lb = true;
            }

            let mag = sign * anom;
            if mag > best_mag {
                best_mag = mag;
                peak_idx = t;
                int_max = anom;
            }

            let rel_mag = sign * rel;
            if rel_mag > best_rel_mag {
                best_rel_mag = rel_mag;
                int_max_rel = rel;
            }

            sum_anom += anom;
            sum_rel += rel;
            valid_n += 1;

            let exceed_here = if is_high {
                if strict { v > th } else { v >= th }
            } else if strict {
                v < th
            } else {
                v <= th
            };

            // Hobday (2018) category = floor((v - seas) / (thresh - seas)); guard zero/sign-flipped delta.
            let ratio = if sign * delta > 1e-12 {
                anom / delta
            } else if exceed_here || mag >= 0.0 {
                1.0
            } else {
                0.0
            };
            let cat_day = (ratio.floor() as i32).clamp(0, 4) as i8;
            if cat_day > cat_max {
                cat_max = cat_day;
            }
            match cat_day {
                1 => c_mod += 1,
                2 => c_str += 1,
                3 => c_sev += 1,
                4 => c_ext += 1,
                _ => {}
            }
            // Dedicated per-direction series never overwrites across high/low gaps; combined intensity_ts preserves legacy precedence.
            if let Some(ref mut dits) = dir_intensity_ts {
                dits[t] = anom;
            }
            if let Some(ref mut its) = intensity_ts {
                if is_high || exceed_here || its[t] == 0.0 {
                    its[t] = anom;
                }
            }
            if let Some(ref mut dts) = daily_cat_ts {
                if is_high || exceed_here || dts[t] == 0 {
                    dts[t] = sign_i8 * cat_day;
                }
            }
        }

        if let Some(ref mut ets) = event_cat_ts {
            let signed_peak_cat = sign_i8 * cat_max;
            for t in ev_start..=ev_end {
                if let Some((v, _, th)) =
                    resolve_day_sample(values[t], seas_t[t], thresh_t[t], lb_opt, bound_policy)
                {
                    let exceed_here = if is_high {
                        if strict { v > th } else { v >= th }
                    } else if strict {
                        v < th
                    } else {
                        v <= th
                    };
                    if is_high || exceed_here || ets[t] == 0 {
                        ets[t] = signed_peak_cat;
                    }
                }
            }
        }

        let int_mean = if valid_n > 0 {
            sum_anom / (valid_n as f64)
        } else {
            f64::NAN
        };
        let int_mean_rel = if valid_n > 0 {
            sum_rel / (valid_n as f64)
        } else {
            f64::NAN
        };

        // Second pass for numerically stable variance / std (exact same valid day filter as pass 1).
        let (int_var, int_std) = if valid_n > ddof {
            let mut ss = 0.0f64;
            for t in ev_start..=ev_end {
                if let Some((v, s, _)) =
                    resolve_day_sample(values[t], seas_t[t], thresh_t[t], lb_opt, bound_policy)
                {
                    let diff = (v - s) - int_mean;
                    ss += diff * diff;
                }
            }
            let var = ss / ((valid_n - ddof) as f64);
            (var, var.sqrt())
        } else if valid_n == 1 && ddof == 0 {
            (0.0, 0.0)
        } else {
            (f64::NAN, f64::NAN)
        };

        // Onset and decline rates on anomalies (T - seas), positive magnitudes in units/step,
        // applying Mask/Floor policies consistently when enabled.
        let a_peak = floor_val(values[peak_idx], lb_opt, bound_policy)
            - floor_val(seas_t[peak_idx], lb_opt, bound_policy);
        let a_ev_start = floor_val(values[ev_start], lb_opt, bound_policy)
            - floor_val(seas_t[ev_start], lb_opt, bound_policy);
        let (a_start, dt_onset) = if ev_start > 0 {
            if let Some((v_prev, s_prev, _)) = resolve_day_sample(
                values[ev_start - 1],
                seas_t[ev_start - 1],
                thresh_t[ev_start - 1],
                lb_opt,
                bound_policy,
            ) {
                let a_prev = v_prev - s_prev;
                (
                    0.5 * (a_ev_start + a_prev),
                    (peak_idx - ev_start) as f64 + 0.5,
                )
            } else {
                let dt = if peak_idx == ev_start {
                    1.0
                } else {
                    (peak_idx - ev_start) as f64
                };
                (a_ev_start, dt)
            }
        } else {
            let dt = if peak_idx == ev_start {
                1.0
            } else {
                (peak_idx - ev_start) as f64
            };
            (a_ev_start, dt)
        };
        let rate_onset = if a_peak.is_finite() && a_start.is_finite() {
            sign * (a_peak - a_start) / dt_onset
        } else {
            f64::NAN
        };

        let a_ev_end = floor_val(values[ev_end], lb_opt, bound_policy)
            - floor_val(seas_t[ev_end], lb_opt, bound_policy);
        let (a_end, dt_decline) = if ev_end + 1 < n {
            if let Some((v_next, s_next, _)) = resolve_day_sample(
                values[ev_end + 1],
                seas_t[ev_end + 1],
                thresh_t[ev_end + 1],
                lb_opt,
                bound_policy,
            ) {
                let a_next = v_next - s_next;
                (
                    0.5 * (a_ev_end + a_next),
                    (ev_end - peak_idx) as f64 + 0.5,
                )
            } else {
                let dt = if peak_idx == ev_end {
                    1.0
                } else {
                    (ev_end - peak_idx) as f64
                };
                (a_ev_end, dt)
            }
        } else {
            let dt = if peak_idx == ev_end {
                1.0
            } else {
                (ev_end - peak_idx) as f64
            };
            (a_ev_end, dt)
        };
        let rate_decline = if a_peak.is_finite() && a_end.is_finite() {
            sign * (a_peak - a_end) / dt_decline
        } else {
            f64::NAN
        };

        let dur_f = dur as f64;
        out_events.push(EventRecord {
            pixel_idx,
            event_type: sign_i8,
            index_start: ev_start as i32,
            index_peak: peak_idx as i32,
            index_end: ev_end as i32,
            duration: dur as i32,
            int_max,
            int_mean,
            int_std,
            int_var,
            int_cum: sum_anom,
            int_max_rel_thresh: int_max_rel,
            int_mean_rel_thresh: int_mean_rel,
            int_cum_rel_thresh: sum_rel,
            rate_onset,
            rate_decline,
            category_max: cat_max,
            p_moderate: c_mod as f64 / dur_f,
            p_strong: c_str as f64 / dur_f,
            p_severe: c_sev as f64 / dur_f,
            p_extreme: c_ext as f64 / dur_f,
            truncated_start: ev_start == 0,
            truncated_end: ev_end + 1 == n,
            near_lower_bound: near_lb,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_short_gaps() {
        let v = vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN, f64::NAN, f64::NAN, 8.0];
        let mut out = Vec::new();
        pad_short_gaps(&v, 2, &mut out);
        assert!((out[1] - 2.0).abs() < 1e-12);
        assert!((out[2] - 3.0).abs() < 1e-12);
        assert!(out[4].is_nan());
        assert!(out[5].is_nan());
        assert!(out[6].is_nan());
    }

    #[test]
    fn test_gap_merge_and_metrics() {
        // Two 5-day high events separated by a 2-day gap -> merged into one 12-day event.
        let mut vals = vec![20.0; 30];
        for t in 5..10 {
            vals[t] = 23.0;
        }
        // gap at 10, 11 = 20.5 (below thresh 22.0)
        vals[10] = 20.5;
        vals[11] = 20.5;
        for t in 12..17 {
            vals[t] = 24.0;
        }
        vals[14] = 26.0; // peak: anom = 6.0, delta = 2.0 -> Cat 3

        let seas = vec![20.0; 30];
        let th_h = vec![22.0; 30];
        let th_l = vec![18.0; 30];
        let params = DetectParams {
            event_mode: EventMode::High,
            min_duration: 5,
            max_gap: 2,
            max_pad_length: 0,
            lower_bound: None,
            bound_policy: BoundPolicy::Flag,
            compat: CompatConfig::default(),
        };
        let mut buf = Vec::new();
        let mut its = vec![f64::NAN; 30];
        let mut dcat = vec![CATEGORY_MISSING; 30];
        let mut ecat = vec![CATEGORY_MISSING; 30];
        let evs = detect_pixel_with_curves(
            0,
            &vals,
            &seas,
            &th_h,
            &th_l,
            &params,
            &mut buf,
            Some(&mut its),
            None,
            None,
            Some(&mut dcat),
            Some(&mut ecat),
        );
        assert_eq!(evs.len(), 1);
        let e = &evs[0];
        assert_eq!(e.index_start, 5);
        assert_eq!(e.index_peak, 14);
        assert_eq!(e.index_end, 16);
        assert_eq!(e.duration, 12);
        assert!((e.int_max - 6.0).abs() < 1e-12);
        assert_eq!(e.category_max, 3);
        assert!(!e.truncated_start);
        assert!(!e.truncated_end);

        assert_eq!(its[0], 0.0); // non-event ocean day is 0.0, not NaN
        assert!((its[14] - 6.0).abs() < 1e-12);

        assert_eq!(dcat[0], 0);
        assert_eq!(dcat[10], 0); // gap day below threshold -> daily cat 0
        assert_eq!(dcat[14], 3); // peak day -> daily cat 3

        assert_eq!(ecat[5], 3); // event peak cat across entire event
        assert_eq!(ecat[10], 3);
    }

    #[test]
    fn test_mcs_freezing_div_zero_guard() {
        let mut vals = vec![-1.75; 20];
        for t in 5..12 {
            vals[t] = -1.80;
        }
        let seas = vec![-1.79; 20];
        let th_l = vec![-1.79; 20]; // delta == 0.0!
        let th_h = vec![-1.5; 20];
        let params = DetectParams {
            event_mode: EventMode::Low,
            min_duration: 5,
            max_gap: 2,
            max_pad_length: 0,
            lower_bound: Some(-1.8),
            bound_policy: BoundPolicy::Flag,
            compat: CompatConfig::default(),
        };
        let mut buf = Vec::new();
        let evs = detect_pixel_with_curves(0, &vals, &seas, &th_h, &th_l, &params, &mut buf, None, None, None, None, None);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].category_max, 1); // safely guarded, never Inf/NaN
        assert!(evs[0].near_lower_bound);
    }

    #[test]
    fn test_split_high_low_intensity_both_and_mask() {
        let mut vals = vec![0.0; 25];
        for t in 2..8 {
            vals[t] = 3.0; // high event
        }
        for t in 12..18 {
            vals[t] = -2.5; // low event
        }
        let seas = vec![0.0; 25];
        let mut th_h = vec![1.0; 25];
        let mut th_l = vec![-1.0; 25];
        // Mask step 20 on low bound (-1.8): low intensity should be NaN there, high keeps 0.0
        th_l[20] = -1.8;
        th_h[20] = 1.0;
        let params = DetectParams {
            event_mode: EventMode::Both,
            min_duration: 5,
            max_gap: 2,
            max_pad_length: 0,
            lower_bound: Some(-1.8),
            bound_policy: BoundPolicy::Mask,
            compat: CompatConfig::default(),
        };
        let mut buf = Vec::new();
        let mut its = vec![f64::NAN; 25];
        let mut its_hi = vec![f64::NAN; 25];
        let mut its_lo = vec![f64::NAN; 25];
        let evs = detect_pixel_with_curves(
            0, &vals, &seas, &th_h, &th_l, &params, &mut buf,
            Some(&mut its), Some(&mut its_hi), Some(&mut its_lo), None, None,
        );
        assert_eq!(evs.len(), 2);
        assert!((its_hi[4] - 3.0).abs() < 1e-12);
        assert_eq!(its_hi[14], 0.0);
        assert_eq!(its_lo[4], 0.0);
        assert!((its_lo[14] + 2.5).abs() < 1e-12);
        assert_eq!(its_hi[20], 0.0);
        assert!(its_lo[20].is_nan());
    }
}
