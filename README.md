# ornstein-uhlenbeck

In mathematics, the Ornstein–Uhlenbeck process is a stochastic process
with applications in financial mathematics and the physical sciences.
Its original application in physics was as a model for the velocity of
a massive Brownian particle under the influence of friction. It is named
after Leonard Ornstein and George Eugene Uhlenbeck. [[1]](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)

The samples generated in this process are often used in reinforcement
learning for exploration, for example in deep mind's ddpg. [[2]](https://arxiv.org/abs/1509.02971)

The implementation is inspired by [[3]](https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py).

```rust
use ornstein_uhlenbeck::OrnsteinUhlenbeckProcessBuilder;
use ndarray::{Array, array};

const ACTION_MIN: f64 = -0.5;
const ACTION_MAX: f64 = 0.5;

let mut ou_process = OrnsteinUhlenbeckProcessBuilder::default().build((3));
for step in 0..100 {
    let mut some_action: Array<f64, _> = array![0.1, 0.5, -0.4];

    // Add some noise from the process for exploration.
    some_action += ou_process.sample_at(step);

    // Now me might exceed our action space...
    some_action = some_action.mapv(|v| v.max(ACTION_MAX).min(ACTION_MIN));

    // ... and use the action...
}
```

License: Apache-2.0/MIT
