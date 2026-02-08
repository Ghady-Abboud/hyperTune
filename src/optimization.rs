use crate::gp::GaussianProcess;
use crate::acquisition::maximize_acquisition;
use rand::Rng;

pub struct BayesianOptimizer {
    gp : GaussianProcess,
    bounds : Vec<(f64, f64)>,
    n_iterations : usize,
    best_x : Vec<f64>,
    best_y : f64
}

impl BayesianOptimizer {
    pub fn new(bounds : Vec<(f64, f64)>, n_iterations : usize) -> Self {
        let gp = GaussianProcess::new(1.0, 1e-6);
        BayesianOptimizer {
            gp,
            bounds,
            n_iterations,
            best_x: vec![],
            best_y: std::f64::INFINITY
        }
    }

    pub fn minimize<F>(&mut self, objective_function : F) {
        for _ in 0..self.n_iterations {
            let x = self.suggest();
            let y = objective_function(&x);
            self.gp.update(&x, y);
            if y < self.best_y {
                self.best_x = x.clone();
                self.best_y = y;
            }
        }
    }

    pub fn suggest(&self) -> Vec<f64> {
        let mut rng = rand::thread_rng();

        if self.gp.is_empty() {
            return self.bounds.iter()
                .map(|(low, high)| rng.gen_range(*low..*high))
                .collect();
        }

        let n_candidates = 100;
        let candidates: Vec<Vec<f64>> = (0..n_candidates)
            .map(|_| {
                self.bounds.iter()
                    .map(|(low, high)| rng.gen_range(*low..*high))
                    .collect()
            })
            .collect();

        maximize_acquisition(&self.gp, candidates, self.best_y)
    }
}
