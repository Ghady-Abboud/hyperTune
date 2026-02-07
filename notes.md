Various Techniques that can be used to tune hyperparameters:

- Grid Search
- Random Search
- Bayesian Optimization
- Tree Prazen Estimators
- Genetic Algorithms
- Metaheuristic based Algorithms

Bayesian Optimization Steps:
- Gather starting points
- Build a probabilistic model based on these 3 observations
- The gaussian process takes our 3 points and produces predictions about function values at unobserved locations
- We use an expected improvement function for a sample at a given location to get the fourth observation

The 2 key components of Bayesian Optimization:
- Surrogate Model (Gaussian Process)
- Acquisition Function (eg: Expected Improvement)

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BAYESIAN OPTIMIZER                                │
│                                                                             │
│  Input:                                                                     │
│    - objective_fn: fn(params) -> f64  (the expensive function to optimize) │
│    - bounds: [(min, max), ...]        (search space for each parameter)    │
│    - n_iterations: usize              (evaluation budget)                  │
│                                                                             │
│  Output:                                                                    │
│    - best_params: Vec<f64>                                                  │
│    - best_value: f64                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OPTIMIZATION LOOP                                 │
│                                                                             │
│  1. Initialize with random samples (2-5 points)                            │
│                        │                                                    │
│                        ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  2. FIT: GaussianProcess.fit(x_train, y_train)                       │  │
│  │         - Build kernel matrix K using RBF kernel                     │  │
│  │         - Add noise to diagonal (numerical stability)                │  │
│  │         - Compute K_inverse                                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                        │                                                    │
│                        ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  3. PREDICT: GaussianProcess.predict(x_new) -> (mean, variance)      │  │
│  │         - mean = k_star^T @ K_inv @ y_train                          │  │
│  │         - variance = k(x_new, x_new) - k_star^T @ K_inv @ k_star     │  │
│  │                                                                      │  │
│  │         k_star = [k(x_new, x_1), k(x_new, x_2), ..., k(x_new, x_n)]  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                        │                                                    │
│                        ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  4. ACQUIRE: Find x_next that maximizes acquisition function         │  │
│  │                                                                      │  │
│  │     Expected Improvement (EI):                                       │  │
│  │                                                                      │  │
│  │         EI(x) = (y_best - mean) * Φ(Z) + stddev * φ(Z)              │  │
│  │         where Z = (y_best - mean) / stddev                          │  │
│  │                                                                      │  │
│  │     Φ = CDF of standard normal                                       │  │
│  │     φ = PDF of standard normal                                       │  │
│  │                                                                      │  │
│  │     High EI when: low mean (exploit) OR high variance (explore)     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                        │                                                    │
│                        ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  5. EVALUATE: y_new = objective_fn(x_next)                           │  │
│  │         - This is the expensive call (e.g., train a neural net)     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                        │                                                    │
│                        ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  6. UPDATE: Add (x_next, y_new) to training data                     │  │
│  │         - x_train.push(x_next)                                       │  │
│  │         - y_train.push(y_new)                                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                        │                                                    │
│                        ▼                                                    │
│              Loop back to step 2 until budget exhausted                     │
└─────────────────────────────────────────────────────────────────────────────┘


## Code Structure (What to Implement)

src/
├── main.rs
│   └── CLI entry point, example usage
│
├── gp.rs (Gaussian Process)
│   ├── struct GaussianProcess
│   │     ├── x_train: Vec<f64>
│   │     ├── y_train: Vec<f64>
│   │     ├── k_inv: DMatrix<f64>
│   │     ├── length_scale: f64
│   │     └── noise: f64
│   │
│   ├── fn new(length_scale, noise) -> Self
│   ├── fn rbf_kernel(&self, x1, x2) -> f64           ✅ DONE
│   ├── fn fit(&mut self, x, y)                       ✅ DONE
│   └── fn predict(&self, x_new) -> (mean, variance)  ⬜ TODO
│
├── acquisition.rs
│   ├── fn expected_improvement(mean, var, y_best) -> f64
│   └── fn maximize_acquisition(gp, bounds) -> x_next
│
└── optimizer.rs
    ├── struct BayesianOptimizer
    └── fn minimize(objective_fn, bounds, n_iter) -> (best_x, best_y)


## Current Progress

- [x] GaussianProcess struct
- [x] RBF kernel
- [x] fit() with kernel matrix construction
- [x] Matrix inversion
- [x] predict() function
- [x] Expected Improvement acquisition
- [ ] Optimizer loop
- [ ] Multi-dimensional support (currently 1D only)
