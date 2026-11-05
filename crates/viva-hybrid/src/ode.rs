//! Fixed-step classic RK4 integrator.

/// One classic 4th-order Runge-Kutta step of size `dt`.
///
/// `deriv(y, dydt)` writes the derivative of the whole state vector into
/// `dydt`. The advanced state is written into `out` (same length as `y`).
pub fn rk4_step(deriv: &dyn Fn(&[f64], &mut [f64]), y: &[f64], dt: f64, out: &mut [f64]) {
    let n = y.len();
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut tmp = vec![0.0; n];

    deriv(y, &mut k1);
    for i in 0..n {
        tmp[i] = y[i] + 0.5 * dt * k1[i];
    }
    deriv(&tmp, &mut k2);
    for i in 0..n {
        tmp[i] = y[i] + 0.5 * dt * k2[i];
    }
    deriv(&tmp, &mut k3);
    for i in 0..n {
        tmp[i] = y[i] + dt * k3[i];
    }
    deriv(&tmp, &mut k4);
    for i in 0..n {
        out[i] = y[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}
