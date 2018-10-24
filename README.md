# Mixture Distribution [![Build Status](https://travis-ci.com/ordovicia/mix-distribution.svg?branch=master)](https://travis-ci.com/ordovicia/mix-distribution)

## Example

```rust
extern crate rand;
extern crate mix_distribution;

use rand::distributions::{Distribution, Normal};
use mix_distribution::Mix;

let mut rng = rand::thread_rng();

// Mixture of two distributions
let mix = {
    let dists = vec![Normal::new(0.0, 1.0), Normal::new(1.0, 2.0)];
    let weights = &[2, 1];
    Mix::new(dists, weights)
};

mix.sample(&mut rng);

// Mixture of three distributions
let mix = {
    let dists = vec![Normal::new(0.0, 1.0), Normal::new(1.0, 2.0), Normal::new(-1.0, 1.0)];
    let weights = &[2, 1, 3];
    Mix::new(dists, weights)
};

mix.sample(&mut rng);
```
