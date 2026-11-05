//! `run_ir <ir.json> <run-config.json> <out.json>`
//!
//! Run-config JSON: `{"dt": f64, "t_end": f64, "sample_times": [f64..],`
//! `"seeds": [u64..]}` or, instead of `seeds`, `{"seed_start": u64, "n_replicates": usize}`.
//!
//! Output JSON: `{"times": [f64..], "species_ids": [str..], "seeds": [u64..],`
//! `"replicates": [[[f64..]..]..]}` where `replicates[r][i][k]` is species `i`
//! (model order) of replicate `r` at `times[k]`. Also `sim_seconds` (wall time of the
//! simulate loops only), `steps`, `events`, and `phase_ns` (nonzero only when the
//! config sets `"phases": true`, which uses instrumented timers).
use serde::Deserialize;
use std::{env, fs, process};
use std::time::Instant;
use viva_hybrid::hybrid::{simulate_instrumented, simulate_stats, SimConfig, Stats};
use viva_hybrid::ir::Model;

#[derive(Deserialize)]
struct RunConfig {
    dt: f64,
    t_end: f64,
    sample_times: Vec<f64>,
    seeds: Option<Vec<u64>>,
    seed_start: Option<u64>,
    n_replicates: Option<usize>,
    /// If true, run instrumented (phase timers) instead of the plain loop.
    #[serde(default)]
    phases: bool,
}

fn fail(msg: String) -> ! {
    eprintln!("run_ir: {msg}");
    process::exit(1);
}

fn main() {
    let a: Vec<String> = env::args().collect();
    if a.len() != 4 {
        fail("usage: run_ir <ir.json> <run-config.json> <out.json>".into());
    }
    let rd = |p: &str| fs::read_to_string(p).unwrap_or_else(|e| fail(format!("{p}: {e}")));
    let model = Model::from_json(&rd(&a[1])).unwrap_or_else(|e| fail(format!("ir: {e}")));
    let cfg: RunConfig =
        serde_json::from_str(&rd(&a[2])).unwrap_or_else(|e| fail(format!("config: {e}")));
    let seeds: Vec<u64> = match (cfg.seeds, cfg.seed_start, cfg.n_replicates) {
        (Some(s), _, _) => s,
        (None, Some(s0), Some(n)) => (s0..s0 + n as u64).collect(),
        _ => fail("config needs `seeds` or `seed_start`+`n_replicates`".into()),
    };
    let mut stats = Stats::default();
    let mut ph = [0u64; 5];
    // Timed region: only the simulate loops (no parse, no serialization).
    let t0 = Instant::now();
    let reps: Vec<Vec<Vec<f64>>> = seeds
        .iter()
        .map(|&seed| {
            let c = SimConfig {
                dt: cfg.dt,
                t_end: cfg.t_end,
                sample_times: cfg.sample_times.clone(),
                seed,
            };
            let (tr, s) = if cfg.phases {
                let (tr, s, p) = simulate_instrumented(&model, &c);
                for (a, b) in ph.iter_mut().zip([
                    p.rhs_continuous,
                    p.rhs_stochastic,
                    p.rk4_total,
                    p.event_bookkeeping,
                    p.total,
                ]) {
                    *a += b;
                }
                (tr, s)
            } else {
                simulate_stats(&model, &c)
            };
            stats.steps += s.steps;
            stats.events += s.events;
            tr.species
        })
        .collect();
    let sim_seconds = t0.elapsed().as_secs_f64();
    let ids: Vec<&str> = model.species.iter().map(|s| s.id.as_str()).collect();
    let out = serde_json::json!({
        "times": cfg.sample_times, "species_ids": ids, "seeds": seeds, "replicates": reps,
        "sim_seconds": sim_seconds, "steps": stats.steps, "events": stats.events,
        "phase_ns": {"rhs_continuous": ph[0], "rhs_stochastic": ph[1], "rk4_total": ph[2],
                     "event_bookkeeping": ph[3], "total": ph[4]},
    });
    fs::write(&a[3], out.to_string()).unwrap_or_else(|e| fail(format!("{}: {e}", a[3])));
}
