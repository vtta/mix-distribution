//! Mixuture distribution.

extern crate rand;

use std::marker::PhantomData;
use std::ops::AddAssign;

use rand::{
    distributions::{
        uniform::{SampleBorrow, SampleUniform},
        Distribution, WeightedIndex,
    },
    Rng,
};

/// Mixture distribution.
///
/// # Example
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
/// let mix = {
///     let dists = vec![Normal::new(0.0, 1.0), Normal::new(1.0, 2.0)];
///     let weights = &[2, 1];
///     Mix::new(dists, weights)
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
    pub fn new<I, J>(dists: I, weights: J) -> Self
    where
        I: IntoIterator<Item = T>,
        J: IntoIterator,
        J::Item: SampleBorrow<X>,
        X: for<'a> AddAssign<&'a X> + Clone + Default,
    {
        Self {
            distributions: dists.into_iter().collect(),
            weights: WeightedIndex::new(weights).unwrap(),
            _marker: PhantomData,
        }
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
            Mix::new(dists, weights)
        };

        for _ in 0..1_000_000 {
            println!("{}", mix.sample(&mut rng));
        }
    }

    #[test]
    fn test_mix_2() {
        let mut rng = rand::thread_rng();

        let mix = {
            use rand::distributions::Uniform;

            let dists = vec![Uniform::new_inclusive(0, 0), Uniform::new_inclusive(1, 1)];
            let weights = &[2, 1];
            Mix::new(dists, weights)
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
            Mix::new(dists, weights)
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
}