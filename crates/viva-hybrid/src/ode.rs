//! Fixed-step classic RK4 integrator.

/// Reusable scratch buffers for RK4; `step` does no heap allocation.
pub struct Rk4Workspace {
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    tmp: Vec<f64>,
}

impl Rk4Workspace {
    pub fn new(n: usize) -> Self {
        Self {
            k1: vec![0.0; n],
            k2: vec![0.0; n],
            k3: vec![0.0; n],
            k4: vec![0.0; n],
            tmp: vec![0.0; n],
        }
    }

    /// One classic RK4 step of size `dt`; `deriv(y, dydt)` writes the
    /// derivative in place. Advanced state is written into `out`.
    pub fn step(&mut self, deriv: &dyn Fn(&[f64], &mut [f64]), y: &[f64], dt: f64, out: &mut [f64]) {
        let n = self.k1.len();
        debug_assert_eq!(y.len(), n);
        debug_assert_eq!(out.len(), n);

        deriv(y, &mut self.k1);
        for i in 0..n {
            self.tmp[i] = y[i] + 0.5 * dt * self.k1[i];
        }
        deriv(&self.tmp, &mut self.k2);
        for i in 0..n {
            self.tmp[i] = y[i] + 0.5 * dt * self.k2[i];
        }
        deriv(&self.tmp, &mut self.k3);
        for i in 0..n {
            self.tmp[i] = y[i] + dt * self.k3[i];
        }
        deriv(&self.tmp, &mut self.k4);
        for i in 0..n {
            out[i] = y[i]
                + dt / 6.0 * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]);
        }
    }
}

/// Convenience wrapper: allocates a workspace per call. Prefer
/// `Rk4Workspace::step` in hot loops.
pub fn rk4_step(deriv: &dyn Fn(&[f64], &mut [f64]), y: &[f64], dt: f64, out: &mut [f64]) {
    Rk4Workspace::new(y.len()).step(deriv, y, dt, out);
}
