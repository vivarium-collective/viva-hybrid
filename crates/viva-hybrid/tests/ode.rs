use viva_hybrid::ode::rk4_step;

#[test]
fn rk4_exponential_decay() {
    let deriv = |y: &[f64], d: &mut [f64]| {
        d[0] = -y[0];
    };
    let mut y = vec![1.0];
    let mut out = vec![0.0];
    let dt = 1e-3;
    for _ in 0..1000 {
        rk4_step(&deriv, &y, dt, &mut out);
        y.copy_from_slice(&out);
    }
    assert!((y[0] - (-1.0f64).exp()).abs() < 1e-6, "y = {}", y[0]);
}

#[test]
fn rk4_multi_component() {
    // y0' = -y0, y1' = 2 (linear growth, exact under RK4)
    let deriv = |y: &[f64], d: &mut [f64]| {
        d[0] = -y[0];
        d[1] = 2.0 + 0.0 * y[1];
    };
    let mut y = vec![1.0, 0.0];
    let mut out = vec![0.0; 2];
    for _ in 0..1000 {
        rk4_step(&deriv, &y, 1e-3, &mut out);
        y.copy_from_slice(&out);
    }
    assert!((y[1] - 2.0).abs() < 1e-9);
}
