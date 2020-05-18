//! In mathematics, the Ornstein–Uhlenbeck process is a stochastic process
//! with applications in financial mathematics and the physical sciences.
//! Its original application in physics was as a model for the velocity of
//! a massive Brownian particle under the influence of friction. It is named
//! after Leonard Ornstein and George Eugene Uhlenbeck. [[1]](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)
//!
//! The samples generated in this process are often used in reinforcement
//! learning for exploration, for example in deep mind's ddpg. [[2]](https://arxiv.org/abs/1509.02971)
//!
//! The implementation is inspired by [[3]](https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py).
//!
//! ```
//! use ornstein_uhlenbeck::OrnsteinUhlenbeckProcessBuilder;
//! use ndarray::{Array, array};
//!
//! const ACTION_MIN: f64 = -0.5;
//! const ACTION_MAX: f64 = 0.5;
//!
//! let mut ou_process = OrnsteinUhlenbeckProcessBuilder::default().build((3));
//! for step in 0..100 {
//!     let mut some_action: Array<f64, _> = array![0.1, 0.5, -0.4];
//!
//!     // Add some noise from the process for exploration.
//!     some_action += ou_process.sample_at(step);
//!
//!     // Now me might exceed our action space...
//!     some_action = some_action.mapv(|v| v.max(ACTION_MAX).min(ACTION_MIN));
//!
//!     // ... and use the action...
//! }
//! ```
use ndarray::{Array, Dimension, ShapeBuilder};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

/// The Ornstein-Uhlenbeck process for sampling.
#[derive(Debug, Clone)]
pub struct OrnsteinUhlenbeckProcess<D: Dimension> {
    mu: f64,
    theta: f64,
    max_sigma: f64,
    min_sigma: f64,
    decay_period: u64,
    state: Array<f64, D>,
    sigma: f64,
}

/// The builder for a process which uses default values for ommited
/// parameters.
///
/// ```
/// use ornstein_uhlenbeck::OrnsteinUhlenbeckProcessBuilder;
/// let ou_process = OrnsteinUhlenbeckProcessBuilder::default().build((2, 2));
///
/// let ou_process = OrnsteinUhlenbeckProcessBuilder::default()
///    .mu(0.0)
///    .theta(0.15)
///    .max_sigma(0.3)
///    .min_sigma(0.3)
///    .decay_period(100_000)
///    .build((2, 2));
/// ```
#[derive(Debug, Clone)]
pub struct OrnsteinUhlenbeckProcessBuilder {
    mu: f64,
    theta: f64,
    max_sigma: f64,
    min_sigma: f64,
    decay_period: u64,
}

impl Default for OrnsteinUhlenbeckProcessBuilder {
    fn default() -> Self {
        Self {
            mu: 0.0,
            theta: 0.15,
            max_sigma: 0.3,
            min_sigma: 0.3,
            decay_period: 100_000,
        }
    }
}

impl OrnsteinUhlenbeckProcessBuilder {
    /// Sets the mean for this process. Defaults to 0.0.
    pub fn mu(&mut self, mu: f64) -> &mut Self {
        self.mu = mu;
        self
    }

    /// Sets the speed for this process. Defaults to 0.15.
    pub fn theta(&mut self, theta: f64) -> &mut Self {
        self.theta = theta;
        self
    }

    /// Sets the maximum volatility of the Wiener process. Defaults to 0.3.
    pub fn max_sigma(&mut self, max_sigma: f64) -> &mut Self {
        self.max_sigma = max_sigma;
        self
    }

    /// Sets the minimum volatility for the Wiener process. Defaults to 0.3.
    pub fn min_sigma(&mut self, min_sigma: f64) -> &mut Self {
        self.min_sigma = min_sigma;
        self
    }

    /// Sets the decay period for this process. Defaults to 100 000.
    pub fn decay_period(&mut self, decay_period: u64) -> &mut Self {
        self.decay_period = decay_period;
        self
    }

    /// Builds the Ornstein-Uhlenbeck process with unset parameters set to default values.
    pub fn build<D: Dimension, Sh: ShapeBuilder<Dim = D>>(
        &self,
        shape: Sh,
    ) -> OrnsteinUhlenbeckProcess<D> {
        OrnsteinUhlenbeckProcess::<D>::new(
            shape,
            self.mu,
            self.theta,
            self.max_sigma,
            self.min_sigma,
            self.decay_period,
        )
    }
}

impl<D: Dimension> OrnsteinUhlenbeckProcess<D> {
    /// Creates a new Ornstein-Uhlenbeck process. For the meaning of the parameters
    /// and default values look at [OrnsteinUhlenbeckProcessBuilder](struct.OrnsteinUhlenbeckProcessBuilder.html).
    /// ```
    /// use ornstein_uhlenbeck::OrnsteinUhlenbeckProcess;
    ///
    /// // Scalar output.
    /// let ou_process_scalar = OrnsteinUhlenbeckProcess::new(1, 0.0, 0.15, 0.3, 0.3, 100_000);
    ///
    /// // Vector with dimension 3 as output.
    /// let ou_process_vector = OrnsteinUhlenbeckProcess::new(3, 0.0, 0.15, 0.3, 0.3, 100_000);
    ///
    /// // Matrix with shape (2, 2) as output.
    /// let ou_process_matric = OrnsteinUhlenbeckProcess::new((2, 2), 0.0, 0.15, 0.3, 0.3, 100_000);
    /// ```
    pub fn new<Sh: ShapeBuilder<Dim = D>>(
        shape: Sh,
        mu: f64,
        theta: f64,
        max_sigma: f64,
        min_sigma: f64,
        decay_period: u64,
    ) -> Self {
        let state = Array::from_elem(shape, mu);
        Self {
            state,
            mu,
            theta,
            max_sigma,
            min_sigma,
            decay_period,
            sigma: max_sigma,
        }
    }

    /// Resets the process.
    pub fn reset(&mut self) {
        self.state.fill(self.mu);
    }

    /// Returns a sample at time-`step` without counting up the steps (used for decay).
    ///
    /// ```
    /// use ornstein_uhlenbeck::OrnsteinUhlenbeckProcess;
    ///
    /// let mut ou_process = OrnsteinUhlenbeckProcess::new(1, 0.0, 0.15, 0.3, 0.3, 100_000);
    /// for step in 0..100 {
    ///     let _sample = ou_process.sample_at(step); // would be equivalent to ou.process.sample();
    /// }
    /// ```
    pub fn sample_at(&mut self, step: u64) -> &Array<f64, D> {
        let rands: Array<f64, D> = Array::random(self.state.dim(), StandardNormal {});
        let dx = (-&self.state + self.mu) * self.theta + rands * self.sigma;
        self.state = &self.state + &dx;
        self.sigma = self.max_sigma
            - (self.max_sigma - self.min_sigma)
                * if 1.0 < step as f64 / self.decay_period as f64 {
                    1.0
                } else {
                    step as f64 / self.decay_period as f64
                };
        &self.state
    }
}
