# Mixture Distribution

[![Build Status][build-img]][build-link]
[![mix-distribution][cratesio-img]][cratesio-link]
[![mix-distribution][docsrs-img]][docsrs-link]

[build-img]: https://travis-ci.com/ordovicia/mix-distribution.svg?branch=master
[build-link]: https://travis-ci.com/ordovicia/mix-distribution

[cratesio-img]: https://img.shields.io/crates/v/mix-distribution.svg
[cratesio-link]: https://crates.io/crates/mix-distribution

[docsrs-img]: https://docs.rs/mix-distribution/badge.svg
[docsrs-link]: https://docs.rs/mix-distribution

## Examples

```rust
use rand::distributions::{Distribution, Normal};
use mix_distribution::Mix;

let mut rng = rand::thread_rng();

// Mixture of two distributions
let mix = {
    let dists = vec![Normal::new(0.0, 1.0), Normal::new(1.0, 2.0)];
    let weights = &[2, 1];
    Mix::new(dists, weights).unwrap()
};

mix.sample(&mut rng);

// Mixture of three distributions
let mix = {
    let dists = vec![Normal::new(0.0, 1.0), Normal::new(1.0, 2.0), Normal::new(-1.0, 1.0)];
    let weights = &[2, 1, 3];
    Mix::new(dists, weights).unwrap()
};

mix.sample(&mut rng);
```

## License

Copyright 2018-2019 Hidehito Yabuuchi \<hdht.ybuc@gmail.com\>

Licensed under the MIT license <LICENSE-MIT or http://opensource.org/licenses/MIT>, or the Apache
License, Version 2.0 <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0> at your option.
All files in the project carrying such notice may not be copied, modified, or distributed except
according to those terms.
