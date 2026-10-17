//! DC removal for captured PCM. Some microphones deliver a constant offset
//! (an AMD ACP PDM digital mic sat at -13 dBFS, #64): every level reading
//! counts it as signal, so a silent room reads as continuous voice and a
//! toggle session never ends on silence. A one-pole high-pass well below the
//! speech band takes the offset out and leaves speech as it was.

use myna_core::AudioFormat;

/// Corner of the high-pass. Speech carries nothing this low; at 100 Hz the
/// loss is under 0.05 dB.
const CUTOFF_HZ: f32 = 10.0;

/// Per-session DC blocker over interleaved S16LE: `y[n] = x[n] - x[n-1] +
/// R * y[n-1]`, one state per channel, carried across calls so a chunk
/// boundary is invisible. Other sample widths pass through untouched, like
/// the level meter that reads them.
pub(crate) struct DcBlocker {
    enabled: bool,
    /// Pole of the filter, `exp(-2 pi fc / fs)`.
    pole: f32,
    channels: Vec<ChannelState>,
    /// Channel of the next sample: a call may end inside a frame.
    next: usize,
}

/// One channel's filter memory. Primed from its first sample, so a large
/// offset is already "the past" and does not ring out as a step.
#[derive(Clone, Copy, Default)]
struct ChannelState {
    primed: bool,
    previous_in: f32,
    previous_out: f32,
}

impl DcBlocker {
    pub(crate) fn new(format: AudioFormat) -> Self {
        let rate = format.sample_rate_hz.max(1) as f32;
        Self {
            enabled: format.sample_width_bytes == 2,
            pole: (-2.0 * std::f32::consts::PI * CUTOFF_HZ / rate).exp(),
            channels: vec![ChannelState::default(); usize::from(format.channels.max(1))],
            next: 0,
        }
    }

    /// Filter `data` in place. Whole samples only; a trailing odd byte is
    /// left as it is.
    pub(crate) fn process(&mut self, data: &mut [u8]) {
        if !self.enabled {
            return;
        }
        let channels = self.channels.len();
        for sample in data.chunks_exact_mut(2) {
            let state = &mut self.channels[self.next];
            self.next = (self.next + 1) % channels;
            let input = f32::from(i16::from_le_bytes([sample[0], sample[1]]));
            if !state.primed {
                state.primed = true;
                state.previous_in = input;
            }
            // Rounded only on the way out: rounding the feedback would let
            // the filter hold a residue of its own.
            let output = (input - state.previous_in + self.pole * state.previous_out)
                .clamp(f32::from(i16::MIN), f32::from(i16::MAX));
            state.previous_in = input;
            state.previous_out = output;
            sample.copy_from_slice(&(output.round() as i16).to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;
    /// -13 dBFS, negative, as the reporter's microphone delivered it.
    const OFFSET: i16 = -7336;

    fn mono() -> AudioFormat {
        AudioFormat::default()
    }

    fn to_bytes(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    fn to_samples(bytes: &[u8]) -> Vec<i16> {
        bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect()
    }

    fn filtered(format: AudioFormat, samples: &[i16]) -> Vec<i16> {
        let mut bytes = to_bytes(samples);
        DcBlocker::new(format).process(&mut bytes);
        to_samples(&bytes)
    }

    /// `seconds` of a sine at `hz` and linear `amplitude`, plus `offset`.
    fn sine(hz: f32, amplitude: f32, offset: i16, seconds: f32) -> Vec<i16> {
        let n = (RATE as f32 * seconds) as usize;
        (0..n)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * hz * i as f32 / RATE as f32;
                let v = amplitude * 32768.0 * phase.sin() + offset as f32;
                v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
            })
            .collect()
    }

    fn rms(samples: &[i16]) -> f32 {
        let sum: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
        ((sum / samples.len() as f64).sqrt() / 32768.0) as f32
    }

    fn mean(samples: &[i16]) -> f32 {
        let sum: f64 = samples.iter().map(|&s| s as f64).sum();
        (sum / samples.len() as f64 / 32768.0) as f32
    }

    fn db(ratio: f32) -> f32 {
        20.0 * ratio.log10()
    }

    /// Samples from `ms` in, past the filter's settling.
    fn after(samples: &[i16], ms: usize) -> &[i16] {
        &samples[RATE as usize * ms / 1000..]
    }

    #[test]
    fn a_constant_offset_is_removed() {
        let out = filtered(mono(), &vec![OFFSET; RATE as usize]);
        let worst = after(&out, 100).iter().map(|s| s.unsigned_abs()).max();
        assert!(worst <= Some(2), "residual {worst:?} LSB");
    }

    #[test]
    fn speech_band_passes_unchanged() {
        for hz in [100.0, 300.0, 1_000.0, 4_000.0] {
            let input = sine(hz, 0.1, 0, 1.0);
            let out = filtered(mono(), &input);
            let loss = db(rms(after(&out, 200)) / rms(after(&input, 200)));
            assert!(loss.abs() <= 0.2, "{hz} Hz changed by {loss} dB");
        }
    }

    #[test]
    fn the_offset_is_removed_from_speech_on_top_of_it() {
        let speech = sine(300.0, 0.1, 0, 1.0);
        let out = filtered(mono(), &sine(300.0, 0.1, OFFSET, 1.0));
        let settled = after(&out, 200);
        assert!(mean(settled).abs() < 1e-3, "mean {}", mean(settled));
        let change = db(rms(settled) / rms(after(&speech, 200)));
        assert!(change.abs() <= 0.2, "speech changed by {change} dB");
    }

    #[test]
    fn state_carries_across_chunks() {
        let input = to_bytes(&sine(300.0, 0.1, OFFSET, 1.0));
        let mut whole = input.clone();
        DcBlocker::new(mono()).process(&mut whole);

        let mut pieces = input.clone();
        let mut blocker = DcBlocker::new(mono());
        let mut rest = pieces.as_mut_slice();
        for samples in [1, 3, 160, 7, 999, 4_000] {
            let (head, tail) = rest.split_at_mut(samples * 2);
            blocker.process(head);
            rest = tail;
        }
        blocker.process(rest);
        assert_eq!(pieces, whole);
    }

    #[test]
    fn channel_state_carries_across_a_split_inside_a_frame() {
        let stereo = AudioFormat {
            channels: 2,
            ..mono()
        };
        let right = sine(300.0, 0.1, 0, 0.5);
        let interleaved: Vec<i16> = right.iter().flat_map(|&r| [OFFSET, r]).collect();
        let input = to_bytes(&interleaved);
        let mut whole = input.clone();
        DcBlocker::new(stereo).process(&mut whole);

        // Odd sample counts: every boundary but the last falls between a
        // frame's left and right samples.
        let mut pieces = input.clone();
        let mut blocker = DcBlocker::new(stereo);
        let mut rest = pieces.as_mut_slice();
        for samples in [1, 3, 161, 7, 999] {
            let (head, tail) = rest.split_at_mut(samples * 2);
            blocker.process(head);
            rest = tail;
        }
        blocker.process(rest);
        assert_eq!(pieces, whole);
    }

    #[test]
    fn a_trailing_odd_byte_is_left_alone() {
        let mut bytes = to_bytes(&vec![OFFSET; 160]);
        bytes.push(0xAB);
        DcBlocker::new(mono()).process(&mut bytes);
        assert_eq!(bytes.last(), Some(&0xAB));
    }

    #[test]
    fn no_startup_step_on_a_large_offset() {
        let out = filtered(mono(), &vec![OFFSET; RATE as usize / 10]);
        let first = &out[..RATE as usize / 50];
        let peak = first.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        assert!(
            (peak as f32 / 32768.0) <= 0.01,
            "first 20 ms peak {peak} LSB"
        );
    }

    #[test]
    fn output_is_clamped_not_wrapped() {
        let mut input = vec![i16::MIN; 100];
        input.extend(vec![i16::MAX; 100]);
        let out = filtered(mono(), &input);
        assert_eq!(out[100], i16::MAX, "the step up wrapped to {}", out[100]);
    }

    #[test]
    fn a_clipped_step_does_not_hold_full_scale() {
        // The filter's memory is clamped too, so it starts decaying from
        // full scale at once instead of draining an overshoot first.
        let mut input = vec![i16::MIN; 100];
        input.extend(vec![i16::MAX; 100]);
        let out = filtered(mono(), &input);
        assert!(out[101] < i16::MAX, "still at full scale: {}", out[101]);
    }

    #[test]
    fn the_filter_settles_back_to_zero_after_speech_ends() {
        let mut input = sine(300.0, 0.1, OFFSET, 0.5);
        input.extend(vec![OFFSET; RATE as usize / 2]);
        let out = filtered(mono(), &input);
        let quiet = &out[RATE as usize * 800 / 1000..];
        let worst = quiet.iter().map(|s| s.unsigned_abs()).max();
        assert!(worst <= Some(2), "residual {worst:?} LSB after speech");
    }

    #[test]
    fn digital_silence_stays_exact_zero() {
        let out = filtered(mono(), &vec![0; RATE as usize]);
        assert!(out.iter().all(|&s| s == 0));
    }

    #[test]
    fn channels_are_filtered_independently() {
        let stereo = AudioFormat {
            channels: 2,
            ..mono()
        };
        let right = sine(300.0, 0.1, 0, 1.0);
        let interleaved: Vec<i16> = right.iter().flat_map(|&r| [OFFSET - 100, r]).collect();
        let out = filtered(stereo, &interleaved);
        let (left_out, right_out): (Vec<i16>, Vec<i16>) =
            out.chunks_exact(2).map(|f| (f[0], f[1])).unzip();

        let worst = after(&left_out, 100).iter().map(|s| s.unsigned_abs()).max();
        assert!(worst <= Some(2), "left residual {worst:?} LSB");
        let change = db(rms(after(&right_out, 200)) / rms(after(&right, 200)));
        assert!(change.abs() <= 0.2, "right changed by {change} dB");
    }

    #[test]
    fn other_sample_widths_pass_through() {
        let s32 = AudioFormat {
            sample_width_bytes: 4,
            ..mono()
        };
        let input: Vec<u8> = (0..4_000u32)
            .flat_map(|i| (i * 1_000).to_le_bytes())
            .collect();
        let mut out = input.clone();
        DcBlocker::new(s32).process(&mut out);
        assert_eq!(out, input);
    }
}
