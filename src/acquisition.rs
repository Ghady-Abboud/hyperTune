use statrs::distribution::{Normal, Continuous, ContinuousCDF};

fn expected_improvement(mean : f64, std_dev : f64, ymax : f64) -> f64 {
    // For maximization: diff = mean - ymax;
    // For minimization: diff = ymax - mean;
    let diff = ymax - mean;
    let u = diff / std_dev;

    // First term is the exploitation term
    // Second term is the exploration term
    let normal_distribution = Normal::new(0.0, 1.0).unwrap();
    let ei = diff * normal_distribution.cdf(u) + std_dev * normal_distribution.pdf(u);
    if std_dev <= 0. {
        return 0.;
    }
    return ei;
}

fn maximize_acquisition(gp : GaussianProcess, bounds) {

}
