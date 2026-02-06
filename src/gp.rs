use nalgebra::DMatrix;

struct GaussianProcess {
    x_train: Vec<f64>,
   y_train: Vec<f64>,
   k_inv: DMatrix<f64>,
   length_scale: f64,
   noise: f64,
}

impl GaussianProcess {
    fn new(length_scale : f64, noise : f64) -> Self {
        GaussianProcess {
            x_train: vec![],
            y_train: vec![],
            k_inv: DMatrix::zeros(0, 0),
            length_scale,
            noise
        }
    }

    fn rbf_kernel(&self, x1: f64, x2: f64) -> f64 {
        let squared_distance: f64 = (x1 - x2).powi(2);
        (-squared_distance / (2.0 * self.length_scale.powi(2))).exp()
    }

    fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) {
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
    }
}
