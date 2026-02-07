mod gp;
mod acquisition;

fn main() {
    let mut gp = gp::GaussianProcess::new(1.0, 0.1);

    // 2D example: 3 observed points, each with 2 dimensions
    let x_train = vec![
        vec![0.0, 0.0],
        vec![1.0, 2.0],
        vec![3.0, 1.0],
    ];
    let y_train = vec![1.0, 0.5, 2.0];

    gp.fit(x_train.clone(), y_train.clone());

    let x_new = vec![1.5, 1.5];
    let (mean, variance) = gp.predict(&x_new);
    println!("Prediction at {:?}: mean={:.4}, variance={:.4}", x_new, mean, variance);

    // Find the best observed value (for minimization)
    let y_min = y_train.iter().cloned().fold(f64::INFINITY, f64::min);

    // Generate candidate points to evaluate (2D grid)
    let mut x_candidates: Vec<Vec<f64>> = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            x_candidates.push(vec![i as f64 * 0.5, j as f64 * 0.5]);
        }
    }

    // Find the next point to sample using Expected Improvement
    let next_x = acquisition::maximize_acquisition(&gp, x_candidates, y_min);
    println!("Next point to sample: {:?}", next_x);

    let (mean, variance) = gp.predict(&next_x);
    let std_dev = variance.sqrt();
    let ei = acquisition::expected_improvement(mean, std_dev, y_min);
    println!("Expected Improvement at {:?}: EI={:.4}", next_x, ei);
}
