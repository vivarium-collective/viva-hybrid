//! `run_ir <ir.json> <run-config.json> <out.json>`
//!
//! Run-config JSON: `{"dt": f64, "t_end": f64, "sample_times": [f64..],`
//! `"seeds": [u64..]}` or, instead of `seeds`, `{"seed_start": u64, "n_replicates": usize}`.
//!
//! Output JSON: `{"times": [f64..], "species_ids": [str..], "seeds": [u64..],`
//! `"replicates": [[[f64..]..]..]}` where `replicates[r][i][k]` is species `i`
//! (model order) of replicate `r` at `times[k]`.
use serde::Deserialize;
use std::{env, fs, process};
use viva_hybrid::hybrid::{simulate, SimConfig};
use viva_hybrid::ir::Model;

#[derive(Deserialize)]
struct RunConfig {
    dt: f64,
    t_end: f64,
    sample_times: Vec<f64>,
    seeds: Option<Vec<u64>>,
    seed_start: Option<u64>,
    n_replicates: Option<usize>,
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
    let reps: Vec<Vec<Vec<f64>>> = seeds
        .iter()
        .map(|&seed| {
            simulate(
                &model,
                &SimConfig {
                    dt: cfg.dt,
                    t_end: cfg.t_end,
                    sample_times: cfg.sample_times.clone(),
                    seed,
                },
            )
            .species
        })
        .collect();
    let ids: Vec<&str> = model.species.iter().map(|s| s.id.as_str()).collect();
    let out = serde_json::json!({
        "times": cfg.sample_times, "species_ids": ids, "seeds": seeds, "replicates": reps,
    });
    fs::write(&a[3], out.to_string()).unwrap_or_else(|e| fail(format!("{}: {e}", a[3])));
}
