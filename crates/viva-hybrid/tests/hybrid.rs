//! Tests for the Haseltine–Rawlings partitioned hybrid simulator.
//!
//! These exercise the five contract properties from the task brief:
//!   pure_continuous, pure_stochastic_birth_death, determinism,
//!   big_dt_single_event, no_negative_species.

use std::collections::BTreeMap;
use viva_hybrid::hybrid::{simulate, SimConfig};
use viva_hybrid::ir::{Ast, Model, Partition, Reaction, Species};

// ---- small model-building helpers -----------------------------------------

fn species(id: &str, initial: f64) -> Species {
    Species {
        id: id.to_string(),
        initial,
    }
}

fn stoich(pairs: &[(&str, i64)]) -> BTreeMap<String, i64> {
    pairs.iter().map(|(s, c)| (s.to_string(), *c)).collect()
}

fn params(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
    pairs.iter().map(|(s, v)| (s.to_string(), *v)).collect()
}

fn mul(a: Ast, b: Ast) -> Ast {
    Ast::Mul(Box::new(a), Box::new(b))
}
fn sp(name: &str) -> Ast {
    Ast::Species(name.to_string())
}
fn param(name: &str) -> Ast {
    Ast::Param(name.to_string())
}

// ---------------------------------------------------------------------------

#[test]
fn pure_continuous() {
    // Single continuous decay reaction: X' = -k X, X(0)=100, k=0.5.
    // Analytic: X(t) = 100 * exp(-k t). g must stay 0 (no stochastic deriv).
    let model = Model {
        species: vec![species("X", 100.0)],
        params: params(&[("k", 0.5)]),
        reactions: vec![Reaction {
            id: "decay".into(),
            partition: Partition::Continuous,
            stoichiometry: stoich(&[("X", -1)]),
            rate: mul(param("k"), sp("X")),
        }],
    };
    let cfg = SimConfig {
        dt: 0.001,
        t_end: 5.0,
        sample_times: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        seed: 7,
    };
    let traj = simulate(&model, &cfg);
    assert_eq!(traj.times, cfg.sample_times);
    assert_eq!(traj.species.len(), 1);
    for (k, &t) in cfg.sample_times.iter().enumerate() {
        let analytic = 100.0 * (-0.5 * t).exp();
        let got = traj.species[0][k];
        assert!(
            (got - analytic).abs() < 1e-5,
            "t={t}: got {got}, analytic {analytic}"
        );
    }
}

#[test]
fn pure_stochastic_birth_death() {
    // Stochastic birth (constant rate k) + death (rate d*X).
    // Linear birth-death: E[X] stationary at k/d. Mean over many seeds must
    // sit within 3 s.e. of k/d (Poisson variance = k/d at stationarity).
    let k = 10.0;
    let d = 1.0;
    let ss = k / d; // = 10
    let n_seeds: u64 = 2000;

    let make = || Model {
        species: vec![species("X", ss)],
        params: params(&[("k", k), ("d", d)]),
        reactions: vec![
            Reaction {
                id: "birth".into(),
                partition: Partition::Stochastic,
                stoichiometry: stoich(&[("X", 1)]),
                rate: param("k"),
            },
            Reaction {
                id: "death".into(),
                partition: Partition::Stochastic,
                stoichiometry: stoich(&[("X", -1)]),
                rate: mul(param("d"), sp("X")),
            },
        ],
    };
    let model = make();

    let mut sum = 0.0;
    for seed in 0..n_seeds {
        let cfg = SimConfig {
            dt: 0.1,
            t_end: 50.0,
            sample_times: vec![50.0],
            seed,
        };
        let traj = simulate(&model, &cfg);
        sum += traj.species[0][0];
    }
    let mean = sum / n_seeds as f64;
    let se = (ss / n_seeds as f64).sqrt();
    assert!(
        (mean - ss).abs() < 3.0 * se,
        "mean {mean} not within 3 s.e. ({}) of {ss}",
        3.0 * se
    );
}

#[test]
fn determinism() {
    let model = Model {
        species: vec![species("X", 10.0)],
        params: params(&[("k", 10.0), ("d", 1.0)]),
        reactions: vec![
            Reaction {
                id: "birth".into(),
                partition: Partition::Stochastic,
                stoichiometry: stoich(&[("X", 1)]),
                rate: param("k"),
            },
            Reaction {
                id: "death".into(),
                partition: Partition::Stochastic,
                stoichiometry: stoich(&[("X", -1)]),
                rate: mul(param("d"), sp("X")),
            },
        ],
    };
    let cfg = SimConfig {
        dt: 0.1,
        t_end: 20.0,
        sample_times: vec![0.0, 5.0, 10.0, 15.0, 20.0],
        seed: 12345,
    };
    let a = simulate(&model, &cfg);
    let b = simulate(&model, &cfg);
    assert_eq!(a.times, b.times);
    assert_eq!(a.species.len(), b.species.len());
    for i in 0..a.species.len() {
        for k in 0..a.species[i].len() {
            assert_eq!(
                a.species[i][k].to_bits(),
                b.species[i][k].to_bits(),
                "species[{i}][{k}] differs"
            );
        }
    }
}

#[test]
fn big_dt_single_event() {
    // Death-only, rate d*X, X(0)=1, with dt enormous relative to the rate so a
    // single integration step overshoots the threshold massively. The crossing
    // must fire EXACTLY one event (X: 1 -> 0); afterwards propensity is 0 so no
    // further events ever occur. A naive "floor(g/T)" counter would fire many.
    let model = Model {
        species: vec![species("X", 1.0)],
        params: params(&[("d", 1000.0)]),
        reactions: vec![Reaction {
            id: "death".into(),
            partition: Partition::Stochastic,
            stoichiometry: stoich(&[("X", -1)]),
            rate: mul(param("d"), sp("X")),
        }],
    };
    let cfg = SimConfig {
        dt: 1000.0,
        t_end: 1000.0,
        sample_times: vec![0.0, 500.0, 1000.0],
        seed: 1,
    };
    let traj = simulate(&model, &cfg);
    assert_eq!(traj.species[0][0], 1.0, "initial sample");
    assert_eq!(traj.species[0][1], 0.0, "after single death");
    assert_eq!(traj.species[0][2], 0.0, "stays at 0, no double-count");
}

#[test]
fn no_negative_species() {
    // Zeroth-order ("constant rate") death keeps propensity > 0 even at X=0, so
    // events keep firing, but every jump would drive X negative and must be
    // rejected by the guard. X must stay exactly 0 throughout.
    let model = Model {
        species: vec![species("X", 0.0)],
        params: params(&[("d", 5.0)]),
        reactions: vec![Reaction {
            id: "death".into(),
            partition: Partition::Stochastic,
            stoichiometry: stoich(&[("X", -1)]),
            rate: param("d"),
        }],
    };
    let cfg = SimConfig {
        dt: 0.5,
        t_end: 10.0,
        sample_times: vec![0.0, 2.5, 5.0, 7.5, 10.0],
        seed: 3,
    };
    let traj = simulate(&model, &cfg);
    for (k, v) in traj.species[0].iter().enumerate() {
        assert_eq!(*v, 0.0, "species went off zero at sample {k}: {v}");
    }
}
