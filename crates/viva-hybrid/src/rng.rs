//! Seedable RNG for the hybrid simulator.
//!
//! Wraps `rand_chacha::ChaCha20Rng` (seeded via `seed_from_u64`) and exposes a
//! single primitive, [`Rng::next_uniform`], returning a `f64` strictly inside
//! `(0, 1)`. Strictly positive output guarantees `-ln(u)` is finite, which the
//! Haseltine–Rawlings threshold draw relies on.
//!
//! The Python reference twin must reproduce this stream exactly: draw a `u64`
//! from the same ChaCha20 stream, keep the top 53 bits, divide by 2^53 to land
//! in `[0, 1)`, and reject an exact 0.0 (redrawing another `u64`).

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// 2^53, the denominator for a 53-bit mantissa uniform.
const TWO_POW_53: f64 = 9_007_199_254_740_992.0;

/// Deterministic ChaCha20-backed uniform source.
pub struct Rng {
    inner: ChaCha20Rng,
}

impl Rng {
    /// Seed the stream from a `u64` (ChaCha20's `seed_from_u64`).
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            inner: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    /// Next uniform in the open interval `(0, 1)`.
    ///
    /// Takes the top 53 bits of a fresh `u64` and scales by `1/2^53`, giving a
    /// value in `[0, 1)`; an exact `0.0` is rejected and the draw retried (a
    /// `2^-53` event), so the result is strictly positive.
    pub fn next_uniform(&mut self) -> f64 {
        loop {
            let bits = self.inner.next_u64() >> 11; // top 53 bits
            let u = bits as f64 / TWO_POW_53;
            if u > 0.0 {
                return u;
            }
        }
    }
}
