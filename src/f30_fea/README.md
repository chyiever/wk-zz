# F30 Feature README

This package extracts waveform, ridge, and harmonic statistics for the F130A pure flow-noise samples.

## Pipeline

1. Detrend raw waveform.
2. High-pass filter.
3. Band-pass filter in the main analysis band.
4. Compute Hilbert envelope.
5. Estimate event support from cumulative envelope energy.
6. Compute STFT power.
7. Detect the fundamental ridge with a harmonic-summed dynamic-programming score.
8. Track the second and third harmonic ridges.
9. Aggregate per-sample features into a batch table.

## Core Symbols

- Raw waveform: x[n]
- Filtered waveform: x_f[n]
- Envelope: e[n] = |Hilbert(x_f[n])|
- Event support: [n_on, n_off]
- STFT power: P(f, t)
- Fundamental ridge: f0(t)
- Second harmonic ridge: f2(t)
- Third harmonic ridge: f3(t)

## Fundamental Ridge Detection

The fundamental ridge is not detected by using only the low-frequency band.
For each candidate f in the f0 search band, a harmonic-summed score is used:

S(f, t) = w1 * log(P(f, t) + eps)
        + w2 * log(P(2f, t) + eps)
        + w3 * log(P(3f, t) + eps)

Default weights:
- w1 = 1.0
- w2 = 1.35
- w3 = 0.55

The dynamic-programming recursion is:

J_t(i) = S(f_i, t) + max_j [J_{t-1}(j) - |f_i - f_j| / lambda]

where lambda is ridge_jump_penalty_hz.
This is important when the second harmonic is stronger than the fundamental.

## Time-Domain Features

### signal_duration_ms
Total file duration in milliseconds.

### event_support_ms
Effective event duration from envelope cumulative energy:
T_support = (n_off - n_on) / fs

### rise_ms
Rise time from onset to envelope peak:
T_rise = (n_peak - n_on) / fs

### decay_ms
Decay time from envelope peak to offset:
T_decay = (n_off - n_peak) / fs

### peak_pos_ratio
Relative envelope peak position inside event support:
r_peak = (n_peak - n_on) / (n_off - n_on)

### decay_rise_ratio
Decay-to-rise asymmetry ratio:
R_dr = T_decay / T_rise

### envelope_peak
Maximum envelope amplitude:
A_env = max e[n]

### envelope_symmetry
Correlation between the envelope and its reversed copy:
Sym_env = corr(e[n], e[N-1-n])

### envelope_energy_ratio_before_after
Energy ratio before and after the envelope peak:
R_ba = sum(e[n]^2, n_on...n_peak) / sum(e[n]^2, n_peak...n_off)

### bulge_coverage
Fraction of the event support where the envelope stays above 60 percent of its peak.

## Harmonic Duration Features

### h2_duration_ms
Total valid duration of the second harmonic.
A frame is valid if:
- |f2(t) - 2 f0(t)| <= tolerance
- P(f2, t) / P(f0, t) >= threshold_h2

### h2_longest_duration_ms
Longest continuous valid second-harmonic segment.

### h3_duration_ms
Total valid duration of the third harmonic.
A frame is valid if:
- |f3(t) - 3 f0(t)| <= tolerance
- P(f3, t) / P(f0, t) >= threshold_h3

### h3_longest_duration_ms
Longest continuous valid third-harmonic segment.

### h2_presence_ratio and h3_presence_ratio
Valid-frame fraction for the second and third harmonic.

### h2_energy_ratio_to_f0 and h3_energy_ratio_to_f0
Ridge-energy ratio of valid harmonic frames relative to the full fundamental ridge.

### h2_rel_error_median and h3_rel_error_median
Median relative mismatch for valid harmonic frames:
Err_h2 = median(|f2 - 2f0| / (2f0))
Err_h3 = median(|f3 - 3f0| / (3f0))

## Fundamental Trajectory Features

### f0_start_khz
Start frequency of the smoothed fundamental ridge.

### f0_end_khz
End frequency of the smoothed fundamental ridge.

### f0_peak_khz
Peak frequency of the smoothed fundamental ridge.

### f0_peak_time_ms
Time location of the peak frequency.

### f0_mean_khz
Mean fundamental frequency.

### f0_span_khz
Frequency span of the smoothed fundamental ridge:
Span_f0 = max(f0) - min(f0)

### f0_peak_count
Number of significant local maxima in the smoothed fundamental ridge.
A peak must satisfy a minimum prominence, minimum peak spacing, and minimum height ratio relative to the global maximum.

### f0_curve_up_ratio
Fraction of frames whose slope is larger than +lambda.

### f0_curve_down_ratio
Fraction of frames whose slope is smaller than -lambda.

### f0_curve_turn_count
Number of sign changes in the ridge slope.

### f0_arch_r2
Quadratic-fit R2 for the smoothed fundamental ridge:
f0(t) ~= a t^2 + b t + c

### f0_arch_score
Arch score for a clean increase-then-decrease trajectory.
This equals f0_arch_r2 only when:
- a < 0
- the parabola vertex is inside the central part of the event
Otherwise it is 0.

### f0_arch_vertex_ms
Time of the quadratic-fit vertex.

## Auxiliary Features

### raw_ptp
Peak-to-peak range of the raw waveform.

### filtered_rms
RMS value of the filtered waveform.

### filtered_ptp
Peak-to-peak range of the filtered waveform.

### ridge_band_energy_ratio
Fraction of STFT energy inside the selected analysis band.

### stft_spectral_peak_khz
Peak frequency of the time-averaged STFT spectrum.

## Notes

1. File duration is not the same as harmonic duration.
2. Absolute amplitude depends on the chosen filter band, so amplitude values should always be interpreted together with the filter settings.
3. This implementation is designed for statistical batch analysis, not only for a single sample.
