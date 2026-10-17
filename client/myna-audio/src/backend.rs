//! The capture-backend seam. A [`CaptureBackend`] opens
//! the device and delivers raw PCM through a [`Producer`]; the adapter core
//! ([`crate::CaptureSource`]) owns everything behind it — re-chunking, the
//! stats tap, the bounded ring. Backends: [`crate::ScriptedBackend`] (fake,
//! T50), `PipeWireBackend` (native, T52 — the sole live-capture backend).

use std::sync::Arc;
use std::time::Duration;

use bytes::{Bytes, BytesMut};
use myna_core::{AudioFormat, CaptureError, CaptureHealth, PcmChunk, StopHandle};
use tokio::sync::watch;

use crate::dc_block::DcBlocker;
use crate::ring::Ring;
use crate::stats::AudioStats;
use crate::voice::{VoiceTracker, FRAME as VOICE_FRAME};

/// What to capture. The adapter passes this through from its builder.
pub struct CaptureSpec {
    /// Produce EXACTLY this; the backend owns any conversion (§7).
    pub format: AudioFormat,
    /// PipeWire node to capture from, by stable `node.name`; `None` = default.
    pub target: Option<String>,
    /// Channel indices to pick/downmix on multi-channel devices (§9). Honored
    /// by the native backend (T52); the subprocess backend must error on
    /// `Some` rather than silently capture the wrong channels.
    pub channels: Option<Vec<u8>>,
    /// Graceful-stop flag; the backend must observe it within ~250 ms.
    pub stop: StopHandle,
}

/// Where a backend delivers PCM. `push` is synchronous and bounded but not
/// realtime-safe: it copies, allocates and takes the ring's mutex, so call it
/// from a tokio task or a plain thread, never a realtime callback. Overload
/// is the buffer's problem, never the backend's.
///
/// The producer also publishes [`CaptureHealth`]: `Capturing` on the first
/// non-empty push, `Faulted` on overload or a fatal finish, `Ended` on a clean
/// one. Dropping it without [`Producer::finish`] is a backend fault, so a lost
/// producer never leaves the consumer waiting forever.
pub struct Producer {
    ring: Arc<Ring>,
    stats: watch::Sender<AudioStats>,
    health: watch::Sender<CaptureHealth>,
    concluded: bool,
    format: AudioFormat,
    chunk_bytes: usize,
    frame_bytes: usize,
    pending: BytesMut,
    captured: Duration,
    session_peak: f32,
    voice: VoiceTracker,
    /// Set by [`Producer::remove_dc`]; filters each chunk before it is
    /// measured or delivered.
    dc_block: Option<DcBlocker>,
}

impl Producer {
    pub(crate) fn new(
        ring: Arc<Ring>,
        stats: watch::Sender<AudioStats>,
        health: watch::Sender<CaptureHealth>,
        format: AudioFormat,
        chunk_bytes: usize,
        frame_bytes: usize,
    ) -> Self {
        Self {
            ring,
            stats,
            health,
            concluded: false,
            format,
            chunk_bytes,
            frame_bytes,
            pending: BytesMut::new(),
            captured: Duration::ZERO,
            session_peak: 0.0,
            voice: VoiceTracker::default(),
            dc_block: None,
        }
    }

    /// Take a microphone's DC offset out of everything this producer
    /// measures and delivers from now on (#64). Live capture only: the
    /// scripted backend's bytes stay exactly as written.
    pub(crate) fn remove_dc(&mut self) {
        self.dc_block = Some(DcBlocker::new(self.format));
    }

    /// Deliver raw PCM (any buffer size; the adapter re-chunks to whole-frame
    /// ~100 ms chunks). Returns `false` once the consumer is gone or capture
    /// has ended — the backend should stop producing.
    pub fn push(&mut self, data: Bytes) -> bool {
        if self.ring.is_terminated() {
            return false;
        }
        if !data.is_empty() {
            self.publish(CaptureHealth::Capturing);
        }
        self.pending.extend_from_slice(&data);
        while self.pending.len() >= self.chunk_bytes {
            let data = self.pending.split_to(self.chunk_bytes).freeze();
            self.emit(data);
        }
        !self.ring.is_terminated()
    }

    /// End capture: clean (`None`) after a graceful stop / device EOF, or
    /// fatal (`Some`) — becomes the stream's single `Err`. Pending whole
    /// frames flush as a final short chunk; a trailing partial frame (a
    /// misbehaving backend) is dropped, not padded.
    pub fn finish(mut self, fault: Option<CaptureError>) {
        self.conclude(fault);
    }

    fn conclude(&mut self, fault: Option<CaptureError>) {
        if std::mem::replace(&mut self.concluded, true) {
            return;
        }
        let whole = self.pending.len() - self.pending.len() % self.frame_bytes;
        if whole > 0 {
            let data = self.pending.split_to(whole).freeze();
            self.emit(data);
        }
        let terminal = match &fault {
            Some(err) if !self.ring.is_closed() => CaptureHealth::Faulted(err.clone()),
            _ => CaptureHealth::Ended,
        };
        self.publish(terminal);
        self.ring.finish(fault);
    }

    /// Advance health; terminal states are never left.
    fn publish(&self, next: CaptureHealth) {
        self.health.send_if_modified(|current| {
            let terminal = matches!(current, CaptureHealth::Faulted(_) | CaptureHealth::Ended);
            if terminal || *current == next {
                return false;
            }
            *current = next;
            true
        });
    }

    fn emit(&mut self, data: Bytes) {
        let data = match self.dc_block.as_mut() {
            Some(dc_block) => {
                let mut owned = BytesMut::from(&data[..]);
                dc_block.process(&mut owned);
                owned.freeze()
            }
            None => data,
        };
        let chunk = PcmChunk::new(data, self.format);
        let (rms, peak, clipped) = levels(&chunk);
        for (frame_rms, frame) in frame_levels(&chunk) {
            self.voice.observe(frame_rms, frame);
        }
        self.captured += chunk.duration();
        self.session_peak = self.session_peak.max(peak);
        if let Some(overload) = self.ring.push(chunk) {
            self.publish(CaptureHealth::Faulted(overload));
        }
        let _ = self.stats.send(AudioStats {
            rms,
            peak,
            session_peak: self.session_peak,
            clipped,
            captured: self.captured,
            noise_floor: self.voice.noise_floor(),
            speech_level: self.voice.speech_level(),
            last_voice: self.voice.last_voice(),
        });
    }
}

impl Drop for Producer {
    fn drop(&mut self) {
        self.conclude(Some(CaptureError::Backend(
            "capture backend exited without reporting an outcome".into(),
        )));
    }
}

/// A capture backend: opens the device and produces raw PCM in exactly
/// `spec.format`, pushing into `producer` from wherever it runs.
pub trait CaptureBackend: Send {
    /// Must return promptly without waiting for the device (spawn a task or
    /// thread for opening and capture). Every outcome, including a failure to
    /// open, goes through `producer.finish`.
    fn start(self: Box<Self>, spec: CaptureSpec, producer: Producer);
}

/// Per-chunk levels, linear full-scale (§8). S16LE only — other widths report
/// silent levels until T33 settles the encoding story.
fn levels(chunk: &PcmChunk) -> (f32, f32, bool) {
    if chunk.format.sample_width_bytes != 2 || chunk.data.len() < 2 {
        return (0.0, 0.0, false);
    }
    let mut sum_sq = 0f64;
    let mut peak = 0i32;
    let mut clipped = false;
    for sample in chunk.data.chunks_exact(2) {
        let v = i16::from_le_bytes([sample[0], sample[1]]) as i32;
        let mag = v.abs();
        peak = peak.max(mag);
        clipped |= mag >= i16::MAX as i32;
        sum_sq += (v as f64) * (v as f64);
    }
    let n = (chunk.data.len() / 2) as f64;
    let rms = ((sum_sq / n).sqrt() / 32768.0) as f32;
    (rms, peak as f32 / 32768.0, clipped)
}

/// The chunk cut into [`VOICE_FRAME`]-long frames, each with its RMS and its
/// actual duration (the last frame of a short final chunk may be shorter).
/// Non-S16LE chunks yield nothing, like [`levels`].
fn frame_levels(chunk: &PcmChunk) -> Vec<(f32, Duration)> {
    if chunk.format.sample_width_bytes != 2 {
        return Vec::new();
    }
    let frame_bytes = (chunk.format.channels as usize * 2).max(1);
    let per_frame = (chunk.format.sample_rate_hz as f64 * VOICE_FRAME.as_secs_f64()) as usize;
    let bytes = (per_frame * frame_bytes).max(frame_bytes);
    chunk
        .data
        .chunks(bytes)
        .filter(|frame| frame.len() >= 2)
        .map(|frame| {
            let sum_sq: f64 = frame
                .chunks_exact(2)
                .map(|s| {
                    let v = i16::from_le_bytes([s[0], s[1]]) as f64;
                    v * v
                })
                .sum();
            let n = (frame.len() / 2) as f64;
            let rms = ((sum_sq / n).sqrt() / 32768.0) as f32;
            let duration = Duration::from_secs_f64(
                frame.len() as f64 / chunk.format.bytes_per_second().max(1) as f64,
            );
            (rms, duration)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s16_chunk(sample: i16, count: usize) -> PcmChunk {
        let mut data = Vec::with_capacity(count * 2);
        for _ in 0..count {
            data.extend_from_slice(&sample.to_le_bytes());
        }
        PcmChunk::new(data, AudioFormat::default())
    }

    #[test]
    fn full_scale_square_is_loud_and_clipped() {
        let (rms, peak, clipped) = levels(&s16_chunk(i16::MAX, 1600));
        assert!(rms > 0.999 && peak > 0.999);
        assert!(clipped);
    }

    #[test]
    fn tenth_scale_signal_reads_a_tenth() {
        let (rms, peak, clipped) = levels(&s16_chunk(3277, 1600));
        assert!((rms - 0.1).abs() < 0.01, "rms {rms}");
        assert!((peak - 0.1).abs() < 0.01, "peak {peak}");
        assert!(!clipped);
    }

    #[test]
    fn silence_is_silent() {
        assert_eq!(levels(&s16_chunk(0, 1600)), (0.0, 0.0, false));
    }

    #[test]
    fn a_chunk_cuts_into_twenty_millisecond_frames() {
        let frames = frame_levels(&s16_chunk(3277, 1600));
        assert_eq!(frames.len(), 5);
        for (rms, duration) in frames {
            assert!((rms - 0.1).abs() < 0.01, "rms {rms}");
            assert_eq!(duration, Duration::from_millis(20));
        }
    }

    #[test]
    fn a_short_final_chunk_keeps_its_partial_frame() {
        let frames = frame_levels(&s16_chunk(0, 400));
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[1].1, Duration::from_millis(5));
    }

    /// -13 dBFS, negative, as the reporter's microphone delivered it (#64).
    const OFFSET: i16 = -7336;
    const RATE: usize = 16_000;

    /// A producer that removes DC, with its stats tap and the ring it
    /// delivers to; 100 ms chunks of 16 kHz mono.
    fn dc_free_producer() -> (Producer, watch::Receiver<AudioStats>, Arc<Ring>) {
        let format = AudioFormat::default();
        let ring = Ring::new(1 << 24, format);
        let (stats, stats_rx) = watch::channel(AudioStats::default());
        let (health, _) = watch::channel(CaptureHealth::Opening);
        let mut producer = Producer::new(ring.clone(), stats, health, format, 3_200, 2);
        producer.remove_dc();
        (producer, stats_rx, ring)
    }

    /// `seconds` of a 300 Hz sine of linear RMS `level`, on `offset`.
    fn speech(level: f32, offset: i16, seconds: f32) -> Vec<i16> {
        let amplitude = level * std::f32::consts::SQRT_2 * 32768.0;
        (0..(RATE as f32 * seconds) as usize)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * 300.0 * i as f32 / RATE as f32;
                (amplitude * phase.sin() + offset as f32).round() as i16
            })
            .collect()
    }

    fn bytes_of(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn a_dc_offset_mic_reads_as_silence() {
        let (mut producer, stats, _ring) = dc_free_producer();
        for _ in 0..50 {
            producer.push(Bytes::from(bytes_of(&[OFFSET; RATE / 10])));
        }
        let stats = *stats.borrow();
        assert!(stats.rms < 0.001, "a silent room read at rms {}", stats.rms);
        assert_eq!(stats.last_voice, None, "the offset was taken for voice");
    }

    #[test]
    fn speech_over_a_dc_offset_is_still_heard() {
        let (mut producer, stats, _ring) = dc_free_producer();
        let mut input = vec![OFFSET; RATE];
        input.extend(speech(0.01, OFFSET, 1.0));
        input.extend(vec![OFFSET; RATE * 2]);
        for piece in input.chunks(RATE / 10) {
            producer.push(Bytes::from(bytes_of(piece)));
        }
        let stats = *stats.borrow();
        let heard = stats.last_voice.expect("the speech was heard");
        assert!(
            heard >= Duration::from_millis(2_000) && heard <= Duration::from_millis(2_100),
            "voice last heard at {heard:?}, not where the speech ended (2 s)"
        );
        assert!(stats.speech_level > 0.008, "speech {}", stats.speech_level);
    }

    #[tokio::test]
    async fn the_filter_runs_across_chunk_boundaries() {
        let input = bytes_of(&speech(0.07, OFFSET, 1.0));
        let mut expected = input.clone();
        crate::dc_block::DcBlocker::new(AudioFormat::default()).process(&mut expected);

        let (mut producer, _stats, ring) = dc_free_producer();
        let mut rest = input.as_slice();
        for samples in [1, 7, 1_601, 999, 3_000, 33] {
            let (head, tail) = rest.split_at(samples * 2);
            producer.push(Bytes::copy_from_slice(head));
            rest = tail;
        }
        producer.push(Bytes::copy_from_slice(rest));
        producer.finish(None);
        let mut delivered = Vec::new();
        while let Some(Ok(chunk)) = ring.next().await {
            delivered.extend_from_slice(&chunk.data);
        }
        assert_eq!(delivered.len(), expected.len());
        assert!(delivered == expected, "the delivered audio is not DC-free");
    }

    #[test]
    fn wide_samples_yield_no_frames() {
        let chunk = PcmChunk::new(
            vec![0u8; 64],
            AudioFormat {
                sample_width_bytes: 4,
                ..AudioFormat::default()
            },
        );
        assert!(frame_levels(&chunk).is_empty());
    }
}
