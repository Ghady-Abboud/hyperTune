use nalgebra::DMatrix;

pub struct GaussianProcess {
    x_train: Vec<f64>,
    y_train: Vec<f64>,
    k_inv: DMatrix<f64>,
    alpha: DMatrix<f64>,
    length_scale: f64,
    noise: f64,
}

impl GaussianProcess {
    pub fn new(length_scale : f64, noise : f64) -> Self {
        GaussianProcess {
            x_train: vec![],
            y_train: vec![],
            k_inv: DMatrix::zeros(0, 0),
            alpha: DMatrix::zeros(0,0),
            length_scale,
            noise
        }
    }

    pub fn rbf_kernel(&self, x1: f64, x2: f64) -> f64 {
        let squared_distance: f64 = (x1 - x2).powi(2);
        (-squared_distance / (2.0 * self.length_scale.powi(2))).exp()
    }

    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) {
        let n = x.len();
        self.x_train = x;
        self.y_train = y;

        let mut k: DMatrix<f64> = DMatrix::zeros(n,n);
        for i in 0..n {
            for j in 0..n {
                k[(i,j)] = self.rbf_kernel(self.x_train[i], self.x_train[j]);
                if i == j {
                    k[(i,j)] += self.noise.powi(2);
                }
            }
        }
        self.k_inv = k.try_inverse().unwrap();
        let y = DMatrix::from_vec(n, 1, self.y_train.clone());
        self.alpha = &self.k_inv * y;

    }

    pub fn predict(&self, x_new: f64) -> (f64, f64) {
        let n = self.x_train.len();
        let mut k_star : DMatrix<f64> = DMatrix::zeros(n, 1);
        for i in 0..n {
            let k_start_value = self.rbf_kernel(self.x_train[i], x_new);
            k_star[(i,0)] = k_start_value;
        }
        let k_star_star = self.rbf_kernel(x_new, x_new);
        let k_star_transpose = k_star.transpose();

        let mean = (&k_star_transpose * &self.alpha)[(0,0)];
        let variance = k_star_star -(&k_star_transpose * &self.k_inv * &k_star)[(0,0)];
        (mean,variance)
    }
}
