//! Haseltine–Rawlings partitioned hybrid ODE/SSA simulator.
//!
//! The simulator advances an augmented state `z = [x_0 .. x_{m-1}, g]` where
//! `x` are the `m` species amounts and `g` (the trailing component, index `m`)
//! is the integrated total propensity of the *stochastic* reactions. Continuous
//! reactions drive a coupled ODE on `x`; stochastic reactions fire as discrete
//! jumps when `g` crosses an exponential threshold `T = -ln(u)`.
//!
//! The exact procedure (stepping, crossing, selection, sampling, and RNG draw
//! order) is the cross-language contract — the Python reference twin mirrors it
//! bit-for-bit. See `task-6-report.md` for the prose specification.

use crate::ir::{Model, Partition};
use crate::ode::Rk4Workspace;
use crate::ratelaw::{compile, RateLaw};
use crate::rng::Rng;
use std::collections::BTreeMap;

/// Tolerance for "the clock has reached a sample time". Absolute, so that a
/// sample landing exactly on the `dt` lattice is captured despite fp drift in
/// the accumulated clock. Mirrored exactly by the Python twin.
const SAMPLE_EPS: f64 = 1e-9;

/// Loop-termination slack: stop once the clock is within this of `t_end`.
const STOP_EPS: f64 = 1e-12;

/// Simulation configuration.
pub struct SimConfig {
    /// Fixed integration step.
    pub dt: f64,
    /// Simulation end time (integration never advances past this).
    pub t_end: f64,
    /// Ascending times at which to record species; all must lie in `[0, t_end]`.
    pub sample_times: Vec<f64>,
    /// RNG seed.
    pub seed: u64,
}

/// Recorded trajectory. `species` is species-major: `species[i]` is the sampled
/// series for species `i` (Model species order); `times == sample_times`.
pub struct Trajectory {
    pub times: Vec<f64>,
    pub species: Vec<Vec<f64>>,
}

type Continuous = (Box<dyn RateLaw>, Vec<(usize, f64)>);
type Stochastic = (Box<dyn RateLaw>, Vec<(usize, i64)>);

/// Record species for every pending sample time the clock `t` has now reached.
fn capture(
    t: f64,
    species_vals: &[f64],
    sample_times: &[f64],
    next: &mut usize,
    out: &mut [Vec<f64>],
) {
    while *next < sample_times.len() && sample_times[*next] <= t + SAMPLE_EPS {
        for (i, series) in out.iter_mut().enumerate() {
            series[*next] = species_vals[i];
        }
        *next += 1;
    }
}

/// Work counters (always on; two integer increments per step).
#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    /// Accepted-step attempts (one full RK4 step each, excl. crossing re-integration).
    pub steps: u64,
    /// Stochastic events fired.
    pub events: u64,
}

/// Coarse phase timers in nanoseconds (only filled by `simulate_instrumented`).
#[derive(Default, Clone, Copy, Debug)]
pub struct Phases {
    pub rhs_continuous: u64,
    pub rhs_stochastic: u64,
    /// Total time inside RK4 `step` calls (includes the RHS time above).
    pub rk4_total: u64,
    /// Event selection, stoichiometry, RNG redraw, sample capture.
    pub event_bookkeeping: u64,
    pub total: u64,
}

/// Run the Haseltine–Rawlings hybrid simulation.
pub fn simulate(model: &Model, cfg: &SimConfig) -> Trajectory {
    simulate_impl(model, cfg, None).0
}

/// Like `simulate` but returns work counters.
pub fn simulate_stats(model: &Model, cfg: &SimConfig) -> (Trajectory, Stats) {
    let (t, s) = simulate_impl(model, cfg, None);
    (t, s)
}

/// Like `simulate` but with coarse phase timers (adds `Instant::now` calls; never
/// use for headline timings).
pub fn simulate_instrumented(model: &Model, cfg: &SimConfig) -> (Trajectory, Stats, Phases) {
    let mut ph = Phases::default();
    let (t, s) = simulate_impl(model, cfg, Some(&mut ph));
    (t, s, ph)
}

fn simulate_impl(
    model: &Model,
    cfg: &SimConfig,
    mut phases: Option<&mut Phases>,
) -> (Trajectory, Stats) {
    use std::cell::Cell;
    use std::time::Instant;
    let timed = phases.is_some();
    let t_cont = Cell::new(0u64);
    let t_stoch = Cell::new(0u64);
    let mut stats = Stats::default();
    let t_all = Instant::now();
    let m = model.species.len();
    let n = m + 1; // augmented: species + g

    let species_index: BTreeMap<String, usize> = model
        .species
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id.clone(), i))
        .collect();

    // Compile each rate law once; split reactions by partition.
    let mut continuous: Vec<Continuous> = Vec::new();
    let mut stochastic: Vec<Stochastic> = Vec::new();
    for rxn in &model.reactions {
        let rate = compile(&rxn.rate, &species_index, &model.params);
        match rxn.partition {
            Partition::Continuous => {
                let stoich = rxn
                    .stoichiometry
                    .iter()
                    .map(|(s, c)| (species_index[s], *c as f64))
                    .collect();
                continuous.push((rate, stoich));
            }
            Partition::Stochastic => {
                let stoich = rxn
                    .stoichiometry
                    .iter()
                    .map(|(s, c)| (species_index[s], *c))
                    .collect();
                stochastic.push((rate, stoich));
            }
        }
    }

    // Augmented derivative: continuous reactions drive species; g integrates
    // the total stochastic propensity.
    let deriv = |y: &[f64], d: &mut [f64]| {
        for v in d.iter_mut() {
            *v = 0.0;
        }
        let x = &y[..m];
        let t0 = if timed { Some(Instant::now()) } else { None };
        for (rate, stoich) in &continuous {
            let a = rate.eval(x);
            for (s, nu) in stoich {
                d[*s] += nu * a;
            }
        }
        let t1 = if timed { Some(Instant::now()) } else { None };
        let mut gsum = 0.0;
        for (rate, _) in &stochastic {
            gsum += rate.eval(x);
        }
        d[m] = gsum;
        if let (Some(a), Some(b)) = (t0, t1) {
            t_cont.set(t_cont.get() + (b - a).as_nanos() as u64);
            t_stoch.set(t_stoch.get() + b.elapsed().as_nanos() as u64);
        }
    };

    // State buffers.
    let mut z = vec![0.0; n];
    for (i, s) in model.species.iter().enumerate() {
        z[i] = s.initial;
    }
    // z[m] = g = 0 already.
    let mut z_saved = vec![0.0; n];
    let mut z_next = vec![0.0; n];
    let mut ws = Rk4Workspace::new(n);
    let mut props = vec![0.0; stochastic.len()];

    let mut rng = Rng::seed_from_u64(cfg.seed);

    // Outputs.
    let mut species_out: Vec<Vec<f64>> = vec![vec![0.0; cfg.sample_times.len()]; m];
    let mut sample_next = 0usize;

    // Initial threshold draw (RNG draw #1), then capture any t=0 samples.
    let mut threshold = -rng.next_uniform().ln();
    let mut t = 0.0;
    capture(t, &z[..m], &cfg.sample_times, &mut sample_next, &mut species_out);

    while t + STOP_EPS < cfg.t_end {
        // Step size, capped at t_end so integration never runs past the end.
        let h = cfg.dt.min(cfg.t_end - t);

        z_saved.copy_from_slice(&z);
        stats.steps += 1;
        let ts = if timed { Some(Instant::now()) } else { None };
        ws.step(&deriv, &z_saved, h, &mut z_next);
        if let (Some(p), Some(a)) = (phases.as_deref_mut(), ts) {
            p.rk4_total += a.elapsed().as_nanos() as u64;
        }

        if z_next[m] < threshold {
            // No stochastic event in this step: accept.
            z.copy_from_slice(&z_next);
            t += h;
            capture(t, &z[..m], &cfg.sample_times, &mut sample_next, &mut species_out);
        } else {
            // g crossed the threshold within this step. Estimate the crossing
            // fraction linearly in g and re-integrate from the saved pre-step
            // state to land at the crossing.
            let g_before = z_saved[m];
            let g_after = z_next[m];
            let f = ((threshold - g_before) / (g_after - g_before)).clamp(0.0, 1.0);
            let ts = if timed { Some(Instant::now()) } else { None };
            ws.step(&deriv, &z_saved, f * h, &mut z_next);
            if let (Some(p), Some(a)) = (phases.as_deref_mut(), ts) {
                p.rk4_total += a.elapsed().as_nanos() as u64;
            }
            let te = if timed { Some(Instant::now()) } else { None };
            stats.events += 1;
            z.copy_from_slice(&z_next);
            t += f * h;

            // Select the firing stochastic reaction proportional to its
            // propensity at the crossing (RNG draw: selection).
            let u2 = rng.next_uniform();
            let x = &z[..m];
            let mut total = 0.0;
            for (i, (rate, _)) in stochastic.iter().enumerate() {
                props[i] = rate.eval(x);
                total += props[i];
            }
            let target = u2 * total;
            let mut chosen = stochastic.len() - 1;
            let mut cum = 0.0;
            for (i, p) in props.iter().enumerate() {
                cum += *p;
                if cum >= target {
                    chosen = i;
                    break;
                }
            }

            // Apply the integer stoichiometry, guarded against negatives.
            let jstoich = &stochastic[chosen].1;
            let mut ok = true;
            for (s, nu) in jstoich {
                if z[*s] + (*nu as f64) < 0.0 {
                    ok = false;
                    break;
                }
            }
            if ok {
                for (s, nu) in jstoich {
                    z[*s] += *nu as f64;
                }
            }

            // The event is consumed regardless: reset g and redraw T
            // (RNG draw: new threshold).
            z[m] = 0.0;
            threshold = -rng.next_uniform().ln();

            capture(t, &z[..m], &cfg.sample_times, &mut sample_next, &mut species_out);
            if let (Some(p), Some(a)) = (phases.as_deref_mut(), te) {
                p.event_bookkeeping += a.elapsed().as_nanos() as u64;
            }
        }
    }

    if let Some(p) = phases.as_deref_mut() {
        p.rhs_continuous += t_cont.get();
        p.rhs_stochastic += t_stoch.get();
        p.total += t_all.elapsed().as_nanos() as u64;
    }
    (
        Trajectory {
            times: cfg.sample_times.clone(),
            species: species_out,
        },
        stats,
    )
}
