# HyperTune Roadmap

## Objective Function

```rust
// What you provide - the function to minimize
Fn(&[f64]) -> f64

// Example: optimize a 2D function
let objective = |x: &[f64]| -> f64 {
    x[0].powi(2) + x[1].powi(2)  // returns loss/error
};
```

---

## Phase 1: Core Optimizer

Create `BayesianOptimizer` that runs the optimization loop.

```rust
let optimizer = BayesianOptimizer::new(bounds);
let (best_x, best_y) = optimizer.minimize(objective, n_iterations);
```

**Tasks:**
- [ ] Create `optimizer.rs` module
- [ ] `new(bounds)` - set parameter bounds
- [ ] `random_point()` - sample within bounds
- [ ] `tell(x, y)` - record observation
- [ ] `suggest_next()` - use GP + EI to pick next point
- [ ] `minimize(objective, n_iter)` - run the loop

---

## Phase 2: Better Initialization

- [ ] Latin Hypercube Sampling (spread initial points evenly)
- [ ] Allow user to provide starting points

---

## Phase 3: Parameter Types

- [ ] Log-scale (for learning rates: 1e-5 to 1e-1)
- [ ] Integer (for batch size, layers)
- [ ] Categorical (for optimizer type: adam/sgd)

---

## Phase 4: More Acquisition Functions

- [ ] Upper Confidence Bound (UCB)
- [ ] Probability of Improvement (PI)

---

## Phase 5: Production Features

- [ ] Save/resume optimization
- [ ] Early stopping
- [ ] Parallel suggestions (batch mode)

---

## Phase 6: Testing & Polish

- [ ] Unit tests
- [ ] Benchmark on test functions
- [ ] Real ML example
- [ ] Documentation

---

## Current Status

| Done | To Do |
|------|-------|
| Gaussian Process | Phase 1: Optimizer loop |
| RBF Kernel | Phase 2: Better init |
| Expected Improvement | Phase 3+: Everything else |
| Multi-dim support | |
