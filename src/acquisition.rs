use crate::gp;
use statrs::distribution::{Normal, Continuous, ContinuousCDF};

fn expected_improvement(mean : f64, std_dev : f64, ymax : f64) -> f64 {
    if std_dev <= 0. {
        return 0.;
    }
    // For maximization: diff = mean - ymax;
    // For minimization: diff = ymax - mean;
    let diff = ymax - mean;
    let u = diff / std_dev;

    // First term is the exploitation term
    // Second term is the exploration term
    let normal_distribution = Normal::new(0.0, 1.0).unwrap();
    let ei = diff * normal_distribution.cdf(u) + std_dev * normal_distribution.pdf(u);

    return ei;
}

fn maximize_acquisition(gp : gp::GaussianProcess, x_candidates : Vec<f64>, ymax : f64) -> f64 {
    let mut best_x = x_candidates[0];
    let (mut mean, mut std_dev) = gp.predict(best_x);
    let mut best_ei = expected_improvement(mean, std_dev, ymax);

    for &x in &x_candidates[1..] {
        (mean, std_dev) = gp.predict(x);
        let ei = expected_improvement(mean, std_dev, ymax);
        if ei > best_ei {
            best_ei = ei;
            best_x = x;
        }
    }
    return best_x;
}
