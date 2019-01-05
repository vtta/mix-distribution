//! Mixuture distributions.

use std::{fmt, marker::PhantomData, ops::AddAssign};

use rand::{
    distributions::{
        uniform::{SampleBorrow, SampleUniform},
        Distribution, WeightedError, WeightedIndex,
    },
    Rng,
};

/// Mixture distributions.
///
/// # Examples
///
/// ```rust
/// extern crate rand;
/// extern crate mix_distribution;
///
/// use rand::distributions::{Distribution, Normal};
/// use mix_distribution::Mix;
///
/// let mut rng = rand::thread_rng();
///
/// // Mixture of two distributions
/// let mix = {
///     let dists = vec![Normal::new(0.0, 1.0), Normal::new(1.0, 2.0)];
///     let weights = &[2, 1];
///     Mix::new(dists, weights).unwrap()
/// };
///
/// mix.sample(&mut rng);
///
/// // Mixture of three distributions
/// let mix = {
///     let dists = vec![Normal::new(0.0, 1.0), Normal::new(1.0, 2.0), Normal::new(-1.0, 1.0)];
///     let weights = &[2, 1, 3];
///     Mix::new(dists, weights).unwrap()
/// };
///
/// mix.sample(&mut rng);
/// ```
pub struct Mix<T, U, X>
where
    T: Distribution<U>,
    X: SampleUniform + PartialOrd,
{
    distributions: Vec<T>,
    weights: WeightedIndex<X>,
    _marker: PhantomData<U>,
}

impl<T, U, X> Mix<T, U, X>
where
    T: Distribution<U>,
    X: SampleUniform + PartialOrd,
{
    /// Creates a new `Mix`.
    /// `dists` and `weights` must have the same length.
    ///
    /// Propagates errors from `WeightedIndex::new()`.
    pub fn new<I, J>(dists: I, weights: J) -> Result<Self, WeightedError>
    where
        I: IntoIterator<Item = T>,
        J: IntoIterator,
        J::Item: SampleBorrow<X>,
        X: for<'a> AddAssign<&'a X> + Clone + Default,
    {
        Ok(Self {
            distributions: dists.into_iter().collect(),
            weights: WeightedIndex::new(weights)?,
            _marker: PhantomData,
        })
    }
}

impl<T, U, X> Distribution<U> for Mix<T, U, X>
where
    T: Distribution<U>,
    X: SampleUniform + PartialOrd,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> U {
        let idx = self.weights.sample(rng);
        self.distributions[idx].sample(rng)
    }
}

impl<T, U, X> Clone for Mix<T, U, X>
where
    T: Distribution<U> + Clone,
    X: SampleUniform + PartialOrd + Clone,
    X::Sampler: Clone,
{
    fn clone(&self) -> Self {
        Self {
            distributions: self.distributions.clone(),
            weights: self.weights.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T, U, X> fmt::Debug for Mix<T, U, X>
where
    T: Distribution<U> + fmt::Debug,
    X: SampleUniform + PartialOrd + fmt::Debug,
    X::Sampler: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mix {{ distributions: {:?}, weights: {:?} }}",
            self.distributions, self.weights
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_mix_plot() {
        let mut rng = rand::thread_rng();

        let mix = {
            use rand::distributions::Normal;

            let dists = vec![Normal::new(0.0, 1.0), Normal::new(5.0, 2.0)];
            let weights = &[2, 1];
            Mix::new(dists, weights).unwrap()
        };

        for _ in 0..30000 {
            println!("{} # mix", mix.sample(&mut rng));
        }

        // # cargo test -- --ignored --nocapture | python plot.py
        //
        // from sys import stdin
        //
        // import numpy as np
        // from numpy.random import normal
        // import matplotlib.pyplot as plt
        //
        // BINS = 128
        // ALPHA = 0.5
        //
        // actually = np.array([float(l.split()[0])
        //                      for l in stdin.readlines() if "# mix" in l])
        // plt.hist(actually, bins=BINS, alpha=ALPHA, label="Actual")
        //
        // expected = np.concatenate(
        //     (normal(0.0, 1.0, 20000), normal(5.0, 2.0, 10000)), axis=0)
        // plt.hist(expected, bins=BINS, alpha=ALPHA, label="Expected")
        //
        // plt.legend()
        // plt.grid()
        //
        // plt.show()
    }

    #[test]
    fn test_mix_2() {
        let mut rng = rand::thread_rng();

        let mix = {
            use rand::distributions::Uniform;

            let dists = vec![Uniform::new_inclusive(0, 0), Uniform::new_inclusive(1, 1)];
            let weights = &[2, 1];
            Mix::new(dists, weights).unwrap()
        };

        let x = mix.sample_iter(&mut rng).take(3000).collect::<Vec<_>>();

        let zeros = x.iter().filter(|&&x| x == 0).count();
        let ones = x.iter().filter(|&&x| x == 1).count();

        assert_eq!(zeros + ones, 3000);

        assert_eq!((zeros as f64 / 1000.0).round() as i32, 2);
        assert_eq!((ones as f64 / 1000.0).round() as i32, 1);
    }

    #[test]
    fn test_mix_3() {
        let mut rng = rand::thread_rng();

        let mix = {
            use rand::distributions::Uniform;

            let dists = vec![
                Uniform::new_inclusive(0, 0),
                Uniform::new_inclusive(1, 1),
                Uniform::new_inclusive(2, 2),
            ];
            let weights = &[3, 2, 1];
            Mix::new(dists, weights).unwrap()
        };

        let x = mix.sample_iter(&mut rng).take(6000).collect::<Vec<_>>();

        let zeros = x.iter().filter(|&&x| x == 0).count();
        let ones = x.iter().filter(|&&x| x == 1).count();
        let twos = x.iter().filter(|&&x| x == 2).count();

        assert_eq!(zeros + ones + twos, 6000);

        assert_eq!((zeros as f64 / 1000.0).round() as i32, 3);
        assert_eq!((ones as f64 / 1000.0).round() as i32, 2);
        assert_eq!((twos as f64 / 1000.0).round() as i32, 1);
    }

    #[test]
    fn test_weight_f64() {
        let mut rng = rand::thread_rng();

        let mix = {
            use rand::distributions::Uniform;

            let dists = vec![Uniform::new_inclusive(0, 0), Uniform::new_inclusive(1, 1)];
            let weights = &[0.5, 1.5];
            Mix::new(dists, weights).unwrap()
        };

        let x = mix.sample_iter(&mut rng).take(2000).collect::<Vec<_>>();

        let zeros = x.iter().filter(|&&x| x == 0).count();
        let ones = x.iter().filter(|&&x| x == 1).count();

        assert_eq!(zeros + ones, 2000);

        assert_eq!((zeros as f64 / 100.0).round() as i32, 5);
        assert_eq!((ones as f64 / 100.0).round() as i32, 15);
    }

    #[test]
    fn error_invalid_weights() {
        use rand::distributions::Uniform;

        let dists = vec![Uniform::new_inclusive(0, 0), Uniform::new_inclusive(1, 1)];

        let weights = &[2, 1][0..0];
        assert_eq!(
            Mix::new(dists.clone(), weights).unwrap_err(),
            WeightedError::NoItem,
        );

        let weights = &[2, -1];
        assert_eq!(
            Mix::new(dists.clone(), weights).unwrap_err(),
            WeightedError::NegativeWeight
        );

        let weights = &[0, 0];
        assert_eq!(
            Mix::new(dists, weights).unwrap_err(),
            WeightedError::AllWeightsZero
        );
    }
}
