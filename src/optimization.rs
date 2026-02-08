use crate::gp::GaussianProcess;
use crate::acquisition::maximize_acquisition;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

pub struct BayesianOptimizer {
    gp: GaussianProcess,
    bounds: Vec<(f64, f64)>,
    n_iterations: usize,
    best_x: Vec<f64>,
    best_y: f64,
    rng: StdRng,
}

impl BayesianOptimizer {
    pub fn with_seed(bounds: Vec<(f64, f64)>, n_iterations: usize, seed: Option<u64>) -> Self {
        let gp = GaussianProcess::new(1.0, 1e-6);
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        BayesianOptimizer {
            gp,
            bounds,
            n_iterations,
            best_x: vec![],
            best_y: f64::INFINITY,
            rng,
        }
    }

    pub fn minimize<F>(&mut self, objective_function: F) -> (Vec<f64>, f64)
    where
        F: Fn(&[f64]) -> f64,
    {
        for _ in 0..self.n_iterations {
            let x = self.suggest();
            let y = objective_function(&x);
            self.gp.update(&x, y);
            if y < self.best_y {
                self.best_x = x.clone();
                self.best_y = y;
            }
        }
        (self.best_x.clone(), self.best_y)
    }

    fn suggest(&mut self) -> Vec<f64> {
        if self.gp.is_empty() {
            return self.bounds.iter()
                .map(|(low, high)| self.rng.gen_range(*low..*high))
                .collect();
        }

        let n_candidates = 100;
        let candidates: Vec<Vec<f64>> = (0..n_candidates)
            .map(|_| {
                self.bounds.iter()
                    .map(|(low, high)| self.rng.gen_range(*low..*high))
                    .collect()
            })
            .collect();

        maximize_acquisition(&self.gp, candidates, self.best_y)
    }
}
