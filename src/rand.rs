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
        Quaternion::new(rng.sample(self), rng.sample(self), rng.sample(self), rng.sample(self))
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

impl<T: NumAlgebraic<Real = T> + Zero + One + Sub<Output = T> + Div<Output = T> + PartialOrd> Distribution<Complex<T>> for StandardUnitary
where
    StandardUniform: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<T> {
        // There exist multiple algorithms here. The following hit or retry algorithm
        // is the fastest in most situations. It also has simple trait bounds (not requiring cos).
        // The downside is, that it requires generating a variable amount of random numbers.
        // With NumAlgebraic one can still take sqrt multiple times to get a high root of unity
        // and do the hit or retry on the remaining circle section (with arbitrary high success rate!).
        let one = T::one();
        let two = one.clone() + one.clone();
        let one_c = Complex::new(one.clone(), one.clone());
        loop {
            let a = &(rng.sample::<Complex<T>, _>(StandardUniform) * two.clone()) - &one_c;
            let n = a.re.abs_sqr() + a.im.abs_sqr();
            // true with a chance of 79% (99% after 3 tries)
            if !n.is_zero() && n < one {
                return a / n.sqrt();
            }
        }
        // higher success rate method using an octagon with 97% success chance.
        // Turns out to be a bit slower, even on the slow default RNG.
        // This would only makes sense for RNG's which are really really slow.
        /*let one = T::one();
        let two = one.clone() + one.clone();
        let half = one.clone() / two.clone();
        let sqrt2 = two.sqrt();
        let sqrt_half = half.clone() * sqrt2.clone();
        let height = two.clone() + two.clone() * sqrt2.clone();
        let w1 = two.clone() * sqrt2.clone() - two.clone();
        let w2 = two.clone() - sqrt2.clone();
        let slope = one.clone() + sqrt_half.clone();
        let c = one.clone() + half.clone() + sqrt2.clone();
        loop {
            let a = rng.sample::<Complex<T>, _>(StandardUniform);
            let mut a = Complex::new(a.re - half.clone(), (a.im - half.clone()) * height.clone());
            if a.im.abs() < one {
                a.re = a.re * w1.clone();
            }
            else {
                let s1 = sqrt_half.clone() * (a.re.clone() + slope.clone() * a.im.abs() - c.clone()).sign();
                let s2 = a.im.sign();
                a.re = a.re * w2.clone() - s1.clone();
                a.im = a.im - s2 * (slope.clone() + s1);
            }
            let n = a.re.abs_sqr() + a.im.abs_sqr();
            // true with a chance of 97%
            if !n.is_zero() && n < one {
                return a / n.sqrt();
            }
        }*/
    }
}

#[cfg(feature = "quaternion")]
impl<T: NumAlgebraic<Real = T> + Zero + One + Sub<Output = T> + Div<Output = T> + PartialOrd> Distribution<Quaternion<T>> for StandardUnitary
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

impl<T: NumElementary<Real = T> + Neg<Output = T> + Zero + One + Sub<Output = T> + Div<Output = T> + PartialOrd> Distribution<Complex<T>>
    for StandardNormal
where
    StandardUniform: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<T> {
        // Box–Muller_transform, see https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        let two = T::one() + T::one();
        // random number in range (0, 1]
        let x = T::one() - rng.sample(StandardUniform);

        // Method 1: use from_polar
        // hope the compiler optimizes this away in most cases!
        // For floats it will optimize it away with opt-level >= 1
        //let tau = (-T::one()).acos() * two.clone();
        //Complex::from_polar((-two * x.ln()).sqrt(), rng.sample(StandardUniform) * tau)

        // Method 2: use StandardUnitary (slightly more restricitve trait bounds, but faster)
        rng.sample::<Complex<_>, _>(StandardUnitary) * (-two * x.ln()).sqrt()
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
