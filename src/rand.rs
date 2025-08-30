//! Derived rand implementations for the new numeric types + [StandardNormal]

use crate::*;
use ::rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use core::ops::*;

/// Standard normal distribution
pub struct StandardNormal;
/// Unitary distribution, meaning a uniform distribution on the values where the norm is 1.
pub struct StandardUnitary;

impl<T> Distribution<Complex<T>> for StandardUniform
where
    StandardUniform: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<T> {
        Complex::new(rng.sample(self), rng.sample(self))
    }
}

#[cfg(feature = "quaternion")]
impl<T> Distribution<Quaternion<T>> for StandardUniform
where
    StandardUniform: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Quaternion<T> {
        Quaternion::new(
            rng.sample(self),
            rng.sample(self),
            rng.sample(self),
            rng.sample(self),
        )
    }
}

impl<T: NumAlgebraic<Real = T> + One + Add<Output = T> + Sub<Output = T>> Distribution<T> for StandardUnitary
where
    StandardUniform: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        // sign is guaranteed to return something unitary.
        (rng.sample(StandardUniform) * (T::one() + T::one()) - T::one()).sign()
    }
}

impl<T: NumAlgebraic<Real = T> + Zero + One + Sub<Output = T> + Div<Output = T> + PartialOrd>
    Distribution<Complex<T>> for StandardUnitary
where
    StandardUniform: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<T> {
        // There exist multiple algorithms here. The following hit or retry algorithm
        // is the fastest in most situations. It also avoids computing TAU for the
        // generic type T from the acos() function, so the trait bounds are simpler.
        // The downside is, that it requires generating a variable amount of random numbers.
        // With NumAlgebraic one can still take sqrt multiple times to get a high root of unity
        // and do the hit or retry on the remaining circle section (very high success chance!).
        // That is currently not implemeneted here (TODO test a higher success rate method like that for performance)
        let one = T::one();
        let two = one.clone() + one.clone();
        let one_c = Complex::new(one.clone(), one.clone());
        loop {
            let a: Complex<T> = &(rng.sample::<Complex<T>, _>(StandardUniform) * two.clone()) - &one_c;
            let n = a.re.abs_sqr() + a.im.abs_sqr();
            // true with a chance of 79% (99% after 3 tries)
            if !n.is_zero() && n < one {
                return a / n.sqrt();
            }
        }
    }
}

#[cfg(feature = "quaternion")]
impl<T: NumAlgebraic<Real = T> + Zero + One + Sub<Output = T> + Div<Output = T> + PartialOrd>
    Distribution<Quaternion<T>> for StandardUnitary
where
    StandardUniform: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Quaternion<T> {
        // There exist multiple algorithms, however random sampling a cube like for complex numbers
        // is not an option, due to its low success probability. However it can build upon the complex sampling.
        let u = rng.sample::<T, _>(StandardUniform);
        let b = rng.sample::<Complex<T>, _>(StandardUnitary) * u.sqrt();
        let a = rng.sample::<Complex<T>, _>(StandardUnitary) * (T::one() - u).sqrt(); // Note, this is never 0
        Quaternion::new(a.re, a.im, b.re, b.im)
    }
}

impl<T: NumElementary<Real = T> + Neg<Output = T> + Zero + One + Sub<Output = T>>
    Distribution<Complex<T>> for StandardNormal
where
    StandardUniform: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<T> {
        // Box–Muller_transform, see https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        let two = T::one() + T::one();
        // hope the compiler optimizes this away in most cases!
        // TODO check, otherwise use StandardUnitary
        let tau = (-T::one()).acos() * two.clone();
        // random number in range (0, 1]
        let x = T::one() - rng.sample(StandardUniform);
        Complex::from_polar((-two * x.ln()).sqrt(), rng.sample(StandardUniform) * tau)
    }
}

#[cfg(feature = "quaternion")]
impl<T: Num<Real = T>> Distribution<Quaternion<T>> for StandardNormal
where
    StandardNormal: Distribution<Complex<T>>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Quaternion<T> {
        let a = rng.sample::<Complex<T>, _>(StandardNormal);
        let b = rng.sample::<Complex<T>, _>(StandardNormal);
        Quaternion::new(a.re, a.im, b.re, b.im)
    }
}

impl<T: Num<Real = T>> Distribution<T> for StandardNormal
where
    StandardNormal: Distribution<Complex<T>>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        rng.sample::<Complex<T>, _>(StandardNormal).re
    }
}
