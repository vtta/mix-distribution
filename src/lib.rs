//! Mixuture distributions.

use std::{fmt, marker::PhantomData, ops::AddAssign};

use rand::Rng;
use rand_distr::{
    uniform::{SampleBorrow, SampleUniform},
    weighted::{WeightedError, WeightedIndex},
    Distribution,
};

/// Mixture distributions.
///
/// # Examples
///
/// ```rust
/// use rand_distr::{Distribution, Normal, Uniform};
/// use mix_distribution::Mix;
///
/// let mut rng = rand::thread_rng();
///
/// // Mixture of two distributions
/// let mix = {
///     let dists = vec![
///         Normal::new(0.0, 1.0).unwrap(),
///         Normal::new(1.0, 2.0).unwrap(),
///     ];
///     let weights = &[2, 1];
///     Mix::new(dists, weights).unwrap()
/// };
/// mix.sample(&mut rng);
///
/// // Mixture of three distributions
/// let mix = {
///     let dists = vec![
///         Uniform::new_inclusive(0.0, 2.0),
///         Uniform::new_inclusive(1.0, 3.0),
///         Uniform::new_inclusive(2.0, 4.0),
///     ];
///     let weights = &[2, 1, 3];
///     Mix::new(dists, weights).unwrap()
/// };
/// mix.sample(&mut rng);
///
/// // From iterator over (distribution, weight) pairs
/// let mix = Mix::with_zip(vec![
///     (Uniform::new_inclusive(0, 2), 2),
///     (Uniform::new_inclusive(1, 3), 1),
/// ])
/// .unwrap();
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
    /// Propagates errors from `rand_dist::weighted::WeightedIndex::new()`.
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

    /// Creats a new `Mix` with the given iterator over (distribution, weight) pairs.
    ///
    /// Propagates errors from `rand_dist::weighted::WeightedIndex::new()`.
    pub fn with_zip<W>(
        dists_weights: impl IntoIterator<Item = (T, W)>,
    ) -> Result<Self, WeightedError>
    where
        W: SampleBorrow<X>,
        X: for<'a> AddAssign<&'a X> + Clone + Default,
    {
        let (distributions, weights): (Vec<_>, Vec<_>) = dists_weights.into_iter().unzip();
        Ok(Self {
            distributions,
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
        f.debug_struct("Mix")
            .field("distributions", &self.distributions)
            .field("weights", &self.weights)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::{Normal, Uniform};

    #[test]
    #[ignore]
    fn test_mix_plot() {
        let mut rng = rand::thread_rng();

        let mix = {
            let dists = vec![
                Normal::new(0.0, 1.0).unwrap(),
                Normal::new(5.0, 2.0).unwrap(),
            ];
            let weights = &[2, 1];
            Mix::new(dists, weights).unwrap()
        };

        for _ in 0..30000 {
            println!("{} # mix", mix.sample(&mut rng));
        }

        // # cargo test test_mix_plot -- --ignored --nocapture | python3 plot.py
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
        // actual = np.array([float(l.split()[0])
        //                      for l in stdin.readlines() if "# mix" in l])
        // plt.hist(actual, bins=BINS, alpha=ALPHA, label="Actual")
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
            let dists = vec![Uniform::new_inclusive(0, 0), Uniform::new_inclusive(1, 1)];
            let weights = &[2, 1];
            Mix::new(dists, weights).unwrap()
        };

        let data = mix.sample_iter(&mut rng).take(300).collect::<Vec<_>>();

        let zeros = data.iter().filter(|&&x| x == 0).count();
        let ones = data.iter().filter(|&&x| x == 1).count();

        assert_eq!(zeros + ones, 300);

        assert_eq!((zeros as f64 / 100.0).round() as i32, 2);
        assert_eq!((ones as f64 / 100.0).round() as i32, 1);
    }

    #[test]
    fn test_mix_3() {
        let mut rng = rand::thread_rng();

        let mix = {
            let dists = vec![
                Uniform::new_inclusive(0, 0),
                Uniform::new_inclusive(1, 1),
                Uniform::new_inclusive(2, 2),
            ];
            let weights = &[3, 2, 1];
            Mix::new(dists, weights).unwrap()
        };

        let data = mix.sample_iter(&mut rng).take(600).collect::<Vec<_>>();

        let zeros = data.iter().filter(|&&x| x == 0).count();
        let ones = data.iter().filter(|&&x| x == 1).count();
        let twos = data.iter().filter(|&&x| x == 2).count();

        assert_eq!(zeros + ones + twos, 600);

        assert_eq!((zeros as f64 / 100.0).round() as i32, 3);
        assert_eq!((ones as f64 / 100.0).round() as i32, 2);
        assert_eq!((twos as f64 / 100.0).round() as i32, 1);
    }

    #[test]
    fn test_weight_f64() {
        let mut rng = rand::thread_rng();

        let mix = {
            let dists = vec![Uniform::new_inclusive(0, 0), Uniform::new_inclusive(1, 1)];
            let weights = &[0.4, 0.6];
            Mix::new(dists, weights).unwrap()
        };

        let data = mix.sample_iter(&mut rng).take(1000).collect::<Vec<_>>();

        let zeros = data.iter().filter(|&&x| x == 0).count();
        let ones = data.iter().filter(|&&x| x == 1).count();

        assert_eq!(zeros + ones, 1000);

        assert_eq!((zeros as f64 / 100.0).round() as i32, 4);
        assert_eq!((ones as f64 / 100.0).round() as i32, 6);
    }

    #[test]
    fn test_zip() {
        let mut rng = rand::thread_rng();

        let mix = Mix::with_zip(vec![
            (Uniform::new_inclusive(0, 0), 2),
            (Uniform::new_inclusive(1, 1), 1),
        ])
        .unwrap();

        let data = mix.sample_iter(&mut rng).take(300).collect::<Vec<_>>();

        let zeros = data.iter().filter(|&&x| x == 0).count();
        let ones = data.iter().filter(|&&x| x == 1).count();

        assert_eq!(zeros + ones, 300);

        assert_eq!((zeros as f64 / 100.0).round() as i32, 2);
        assert_eq!((ones as f64 / 100.0).round() as i32, 1);
    }

    #[test]
    fn error_invalid_weights() {
        let dists = vec![Uniform::new_inclusive(0, 0), Uniform::new_inclusive(1, 1)];

        let weights = &[2, 1][0..0];
        assert_eq!(
            Mix::new(dists.clone(), weights).unwrap_err(),
            WeightedError::NoItem,
        );

        let weights = &[2, -1];
        assert_eq!(
            Mix::new(dists.clone(), weights).unwrap_err(),
            WeightedError::InvalidWeight,
        );

        let weights = &[0, 0];
        assert_eq!(
            Mix::new(dists, weights).unwrap_err(),
            WeightedError::AllWeightsZero,
        );
    }
}
