mod gp;
mod acquisition;

fn main() {
    let mut gp = gp::GaussianProcess::new(1.0, 0.1);

    let x_train = vec![0.0, 2.0, 4.0];
    let y_train = vec![1.0, 0.5, 2.0];

    gp.fit(x_train.clone(), y_train.clone());

    let x_new = 1.0;
    let (mean, variance) = gp.predict(x_new);
    println!("Prediction at x={}: mean={:.5}, variance={:.4}", x_new, mean, variance);

    // Find the best observed value (for minimization)
    let y_min = y_train.iter().cloned().fold(f64::INFINITY, f64::min);

    // Generate candidate points to evaluate
    let x_candidates: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();

    // Find the next point to sample using Expected Improvement
    let next_x = acquisition::maximize_acquisition(&gp, x_candidates, y_min);
    println!("Next point to sample: x={:.4}", next_x);

    let (mean, variance) = gp.predict(next_x);
    let std_dev = variance.sqrt();
    let ei = acquisition::expected_improvement(mean, std_dev, y_min);
    println!("Expected Improvement at x={:.4}: EI={:.4}", next_x, ei);
}
