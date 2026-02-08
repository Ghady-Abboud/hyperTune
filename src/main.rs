mod gp;
mod acquisition;
mod optimization;

use optimization::BayesianOptimizer;

fn main() {
    // Global minimum is at (1, 1) with value 0
    let rosenbrock = |x: &[f64]| -> f64 {
        let a = 1.0;
        let b = 100.0;
        (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
    };

    // Search space: x in [-2, 2], y in [-2, 2]
    let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)];
    let n_iterations = 30;

    // Use a fixed seed for reproducible results
    let mut optimizer = BayesianOptimizer::with_seed(bounds, n_iterations, Some(42));
    let (best_x, best_y) = optimizer.minimize(rosenbrock);

    println!("Optimization complete!");
    println!("Best parameters: [{:.4}, {:.4}]", best_x[0], best_x[1]);
    println!("Best value: {:.6}", best_y);
    println!("True optimum: [1.0, 1.0] with value 0.0");
}
