//! Field extensions using a lightweight generic approach without default implementations.

use core::cmp::Ordering;
use core::fmt;
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::*;
use take_mut::take;

use crate::complex::Complex;
use crate::rational::Ratio;
use crate::*;

// TODO when doing division, check with is_unit and panic if not (use `debug_assert!()`).

pub trait SqrtConst<T: Num> {
    type Real: SqrtConst<T::Real, Real = Self::Real>;
    /// constant for internal use to be able to check at compile time, if a SqrtConst is allowed.
    const _TEST_SQRT: () = ();
    /// Get the square of this constant. This constant is not allowed to be representable with `T`,
    /// as otherwise equallity checks will be wrong.
    ///
    /// Note, there is nothing enforcing that this is a compile time constant,
    /// however if it changes at runtime, the values of all numbers using it, will change.
    fn sqr() -> T;
    /// returns true if the value of `Self::sqr()` is considered negative and
    /// therefore the sqrt changes sign under complex conjugation.
    ///
    /// Note, implementing it this way, allows for more flexible trait bounds on [Num].
    fn is_negative() -> bool;
    /// Get the rounded down version of this square root by computing `floor(sqrt(Self::sqr()))`
    fn floor() -> T;
}

/// Type for field extentions with square roots of natural numbers.
/// For runtime constants, use the `dynamic_sqrt_u64!(SqrtN, n)` macro.
pub struct Sqrt<T: FromU64, const N: u64>(PhantomData<T>); // not constructible
impl<T: FromU64 + Num, const N: u64> SqrtConst<T> for Sqrt<T, N>
where
    T::Real: FromU64,
{
    type Real = Sqrt<T::Real, N>;
    const _TEST_SQRT: () = {
        assert!(
            N.isqrt() * N.isqrt() != N,
            "N is not allowed to be a perfect square"
        );
        ()
    };
    #[inline(always)]
    fn sqr() -> T {
        let _ = Self::_TEST_SQRT;
        T::from_u64(N)
    }
    #[inline(always)]
    fn is_negative() -> bool {
        false
    }
    #[inline(always)]
    fn floor() -> T {
        T::from_u64(N.isqrt())
    }
}
// TODO maybe make just this one available? Doesn't work with unsigned numbers though!
/// Type for field extentions with square roots of natural numbers.
pub struct SignedSqrt<T: FromU64, const N: i64>(PhantomData<T>); // not constructible
impl<T: FromU64 + Neg<Output = T> + Num, const N: i64> SqrtConst<T> for SignedSqrt<T, N>
where
    T::Real: FromU64 + Neg<Output = T::Real>,
{
    type Real = SignedSqrt<T::Real, N>;
    const _TEST_SQRT: () = {
        assert!(
            N < 0 || N.isqrt() * N.isqrt() != N,
            "N is not allowed to be a perfect square"
        );
        ()
    };
    #[inline(always)]
    fn sqr() -> T {
        let _ = Self::_TEST_SQRT;
        let v = T::from_u64(N.abs() as u64);
        if N < 0 { -v } else { v }
    }
    #[inline(always)]
    fn is_negative() -> bool {
        N < 0
    }
    #[inline(always)]
    fn floor() -> T {
        assert!(N > 1);
        T::from_u64(N.isqrt() as u64)
    }
}

#[macro_export]
#[cfg(feature = "std")]
/// Define a new square root constant using a u64 that can depend on variables in the current scope.
/// Note, this macro creates a global variable, so use sparingly.
macro_rules! dynamic_sqrt_u64 {
    ($name:ident, $value:expr) => {
        std::thread_local! {
            static _SQR_CONST: core::cell::Cell<u64> = core::cell::Cell::new(0);
            static _SQRT_CONST: core::cell::Cell<u64> = core::cell::Cell::new(0);
        }
        {
            let n = $value;
            let nsqrt = n.isqrt();
            assert!(
                nsqrt * nsqrt != n,
                "The square root constant N={n}={nsqrt}^2 is not allowed to be a square number."
            );
            _SQR_CONST.set(n);
            _SQRT_CONST.set(nsqrt);
        }
        struct $name<T: FromU64>(core::marker::PhantomData<T>);
        impl<T: FromU64 + Num> SqrtConst<T> for $name<T>
        where
            T::Real: FromU64,
        {
            type Real = $name<T::Real>;
            #[inline(always)]
            fn sqr() -> T {
                T::from_u64(_SQR_CONST.get())
            }
            #[inline(always)]
            fn is_negative() -> bool {
                false
            }
            #[inline(always)]
            fn floor() -> T {
                T::from_u64(_SQRT_CONST.get())
            }
        }
    };
}

impl<T: Num + Cancel + Neg<Output = T> + PartialOrd> Ratio<T>
where
    for<'a> &'a T: AddMulSub<Output = T>,
{
    /// Exponentially converging approximation of the square root of a constant.
    /// Computed in O(log n) time (n = iterations) and convergent with at least `error < 2^-n`.
    ///
    /// To compute the optimal result with continued fractions use:
    /// ```
    /// use snum::*;
    /// use snum::rational::*;
    /// use snum::extension::*;
    ///
    /// let n = 10;
    /// let x = SqrtExt::<_, Sqrt<i64, 31>>::new(0, 1);
    /// let cfrac = DevelopContinuedFraction::new(x).take(n);
    /// let res = cfrac.continued_fraction(1).last().unwrap();
    /// assert_eq!(res, Ratio::new(33646, 6043));
    /// assert_ne!(res, Ratio::approx_sqrt::<Sqrt<i64, 31>>(n as u64));
    /// ```
    ///
    /// In some cases this function gives the continued fraction results, like for √3
    /// ```
    /// use snum::*;
    /// use snum::rational::*;
    /// use snum::extension::*;
    ///
    /// const N: u64 = 3;
    /// for n in (1..10).step_by(2) {
    ///     let x = SqrtExt::<_, Sqrt<i64, N>>::new(0, 1);
    ///     let cfrac = DevelopContinuedFraction::new(x).take(n);
    ///     let res = cfrac.continued_fraction(1).last().unwrap();
    ///     assert_eq!(res, Ratio::approx_sqrt::<Sqrt<i64, N>>((n / 2) as u64));
    /// }
    /// ```
    pub fn approx_sqrt<E: SqrtConst<T>>(iterations: u64) -> Self {
        // Note, that (√N - floor(√N))^n converges to 0 for increasing n
        // This gives a rational number by x + y√N = 0 -> -x/y = √N
        // round(√N) is better than floor(√N), so use that
        // Note, that (x - round(x))^n converges exponentially to 0 for increasing n
        // This gives a rational number by x + y√N = 0 -> -x/y = √N, which offers many more good approximations depending on x.
        let x1 = SqrtExt::<_, E>::new(-E::floor(), T::one());
        let x2 = SqrtExt::<_, E>::new(E::floor() + T::one(), -T::one());
        let x = if x1 < x2 { x1 } else { x2 };
        let x = x.powu(iterations + 1);
        Self::new(-x.value, x.ext)
    }
}

impl<T: Num + Cancel + Neg<Output = T> + PartialOrd + Div<Output = T>, E: SqrtConst<T>>
    SqrtExt<T, E>
where
    for<'a> &'a T: AddMulSub<Output = T>,
{
    /// Exponentially converging rational approximation of this number.
    pub fn approx_rational(&self, iterations: u64) -> Ratio<T> {
        // just use the sqrt approximation without any additional fancy stuff.
        &(&Ratio::approx_sqrt::<E>(iterations) * &self.ext) + &self.value
    }
}

/// Field/ring extension with square roots like Z[√2] with numbers like 2+3√2.
/// Complex numbers can be constructed like this as well using √-1.
///
/// The type which supplies the number, with which the extention happens, is fixed,
/// so the number is not saved in the struct (saving memory). Mixing extensions doesn't
/// work anyway, as it usually increases the size of the basis.
///
/// Note, the special implementation for fractions (a+b√n)/q isn't available in this library,
/// however a/p+(b/q)√n can be used instead, requiring a bit more memory and computation, but
/// with an extended value range.
#[derive(Hash)]
pub struct SqrtExt<T: Num, E: SqrtConst<T>> {
    pub value: T,
    pub ext: T,
    #[doc(hidden)]
    _e: PhantomData<E>,
}

impl<T: Num, E: SqrtConst<T>> Clone for SqrtExt<T, E> {
    fn clone(&self) -> Self {
        Self::new(self.value.clone(), self.ext.clone())
    }
}

impl<T: Num + Copy, E: SqrtConst<T>> Copy for SqrtExt<T, E> {}

impl<T: Num, E: SqrtConst<T>> SqrtExt<T, E> {
    pub const fn new(value: T, ext: T) -> Self {
        SqrtExt {
            value,
            ext,
            _e: PhantomData,
        }
    }
}

impl<T: Num + Default, E: SqrtConst<T>> Default for SqrtExt<T, E> {
    fn default() -> Self {
        Self::new(T::default(), T::default())
    }
}

impl<T: Zero + Num, E: SqrtConst<T>> SqrtExt<T, E> {
    /// Returns true if the number can be written as a normal number.
    #[inline]
    pub fn is_integral(&self) -> bool {
        self.ext.is_zero()
    }
}
impl<T: Num + Cancel, E: SqrtConst<T>> SqrtExt<T, E>
where
    Self: Cancel + IntoDiscrete<Output = T>,
{
    /// Returns a (positive) unit != 1 derived from [One] in `T`: x+y√N (satisfies `(x^2 - y^2 N).is_unit()`, `x,y > 0`)
    /// The number x-y√N is the inverse.
    ///
    /// This (non const) function is computing it, so avoid calling it multiple times and precompute it, if possible.
    pub fn unit() -> Self {
        if E::sqr().is_unit() {
            return SqrtExt::new(T::zero(), T::one());
        }
        // Based on continued fractions.
        // A cycle in the continued fraction development gives the fundamental unit != 1.
        // https://brilliant.org/wiki/quadratic-diophantine-equations-pells-equation/

        let x = SqrtExt::<T, E>::new(T::zero(), T::one());
        let mut cfrac_iter = DevelopContinuedFraction::new(x).continued_fraction(T::zero());
        for _ in 0..1000 {
            let xy = cfrac_iter.next().unwrap();
            if xy.is_finite() {
                let s = SqrtExt::new(xy.numer, xy.denom);
                if s.abs_sqr_ext().is_unit() {
                    return s;
                }
            }
        }
        panic!("didn't find solution");
    }
}

impl<T: Zero + Num, E: SqrtConst<T>> Zero for SqrtExt<T, E>
where
    for<'a> &'a T: AddMul<Output = T>,
{
    fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }
    fn is_zero(&self) -> bool {
        self.value.is_zero() && self.ext.is_zero()
    }
}

impl<T: Zero + One + Num, E: SqrtConst<T>> One for SqrtExt<T, E>
where
    for<'a> &'a T: Mul<Output = T>,
{
    fn one() -> Self {
        Self::new(T::one(), T::zero())
    }
    fn is_one(&self) -> bool {
        self.value.is_one() && self.ext.is_zero()
    }
}

// need to implement myself, as otherwise E is required to be PartialEq, which is silly.
impl<T: PartialEq + Num, E: SqrtConst<T>> PartialEq for SqrtExt<T, E> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.ext == other.ext
    }
}
impl<T: Eq + Num, E: SqrtConst<T>> Eq for SqrtExt<T, E> {}

impl<T: Num + PartialOrd + Sub<Output = T> + Mul<Output = T>, E: SqrtConst<T>> PartialOrd
    for SqrtExt<T, E>
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // x = a+b√N > 0 <=> a > -b√N <=> b > -a/√N
        // and for a >= 0: <=> a^2 > b^2 N
        // and for b >= 0: <=> b^2 N > a^2
        let a = self.value.partial_cmp(&other.value)?;
        let b = self.ext.partial_cmp(&other.ext)?;
        Some(match (a, b) {
            (Ordering::Equal, _) => b,
            (_, Ordering::Equal) => a,
            (Ordering::Greater, Ordering::Greater) => Ordering::Greater,
            (Ordering::Less, Ordering::Less) => Ordering::Less,
            _ => {
                let x = self - other;
                let c =
                    (x.value.clone() * x.value).partial_cmp(&(x.ext.clone() * x.ext * E::sqr()))?;
                match (a, b) {
                    (Ordering::Less, _) => c.reverse(),
                    (_, Ordering::Less) => c,
                    _ => unreachable!(),
                }
            }
        })
    }
}
impl<T: Num + Zero + Ord + Neg<Output = T> + Sub<Output = T> + Mul<Output = T>, E: SqrtConst<T>> Ord
    for SqrtExt<T, E>
where
    for<'a> &'a T: Mul<Output = T>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap() // no panic
    }
}

impl<
    T: Num + Cancel + IntoDiscrete + Neg<Output = T> + PartialOrd + Div<Output = T>,
    E: SqrtConst<T>,
> IntoDiscrete for SqrtExt<T, E>
where
    <T as IntoDiscrete>::Output: IntoDiscrete<Output = <T as IntoDiscrete>::Output>
        + Add<Output = <T as IntoDiscrete>::Output>
        + Div<Output = <T as IntoDiscrete>::Output>,
{
    type Output = T;
    fn floor(&self) -> Self::Output {
        if self.ext.is_zero() || self.ext.is_one() {
            self.ext.clone() * E::floor() + self.value.floor().into()
        } else if (-self.ext.clone()).is_one() {
            T::from(self.value.floor()) - self.ext.clone() * (E::floor() + T::one())
        } else {
            // 1. split the integral part off
            // 2. floor(x√N) = floor(x)*floor(√N) + n with integer 0 <= n < x (+ floor(√N) for non integer types)
            // -> do a binary search for the integer n in O(log x) steps to find the value n where x√N <> floor(x)*floor(√N) + n switches
            // TODO it would be much better to have integers here, but that requires e.g. TryInto<u64>, as I would need E::floor() as u64
            // TODO the following only works for integers...
            let v = self.ext.clone() * E::floor();
            let w = Self::new(T::zero(), self.ext.clone());
            assert!(w > v.clone().into());
            let mut a = T::zero().floor();
            let mut b = (self.ext.clone() + E::floor() - T::one()).floor();
            let two = (T::one() + T::one()).floor();
            loop {
                let c = ((a.clone() + b.clone()) / two.clone()).floor(); // this way to also handle float
                if c != a && c != b {
                    let add: T = v.clone() + c.clone().into();
                    if w > add.into() {
                        // too small still
                        a = c;
                    } else {
                        b = c;
                    }
                } else {
                    // is only reached for "good" T
                    break;
                }
            }
            // now we have v+a < x√N < v+b, so v+a is the floor
            self.value.clone() + v + a.into()
        }
    }
    fn round(&self) -> Self::Output {
        let x1 = self.floor();
        let x2 = x1.clone() + T::one();
        let d1 = -(self - &x1); // positive
        let d2 = self - &x2; // positive
        if d1 < d2 { x1 } else { x2 }
    }
}

impl<T: Num + Zero, E: SqrtConst<T>> From<T> for SqrtExt<T, E> {
    fn from(value: T) -> Self {
        SqrtExt::new(value, T::zero())
    }
}
impl<T: Num<Real = T>, E: SqrtConst<T>, E2: SqrtConst<Complex<T>>> From<SqrtExt<T, E>>
    for SqrtExt<Complex<T>, E2>
where
    Complex<T>: Num<Real = T>,
{
    fn from(value: SqrtExt<T, E>) -> Self {
        assert_eq!(
            Complex::from(E::sqr()),
            E2::sqr(),
            "the extension number needs to be the same when upcasting to complex type"
        );
        SqrtExt::new(value.value.into(), value.ext.into())
    }
}

impl<T: Num + FromU64 + Zero, E: SqrtConst<T>> FromU64 for SqrtExt<T, E> {
    #[inline(always)]
    fn from_u64(value: u64) -> Self {
        SqrtExt::from(T::from_u64(value))
    }
}

impl<T: Num + Neg<Output = T>, E: SqrtConst<T>> Conjugate for SqrtExt<T, E> {
    fn conj(&self) -> Self {
        let mut ext = self.ext.conj();
        if E::is_negative() {
            ext = -ext;
        }
        Self::new(self.value.conj(), ext)
    }
}

impl<T: Num + Neg<Output = T>, E: SqrtConst<T>> SqrtExt<T, E> {
    /// negate the sqrt part.
    pub fn conj_ext(&self) -> Self {
        Self::new(self.value.clone(), -self.ext.clone())
    }
}

impl<T: Num + One + Add<Output = T> + Sub<Output = T>, E: SqrtConst<T>> SqrtExt<T, E> {
    /// compute `self * self.conj_ext() = value^2 - ext^2 * E::sqr()`
    pub fn abs_sqr_ext(&self) -> T {
        // overflows easily
        //self.value.clone() * self.value.clone() - self.ext.clone() * self.ext.clone() * E::sqr()
        // less overflows (but slower) with rearranged form (value - ext)(value + ext*N) - (N-1)*value*ext
        // TODO make two versions depending on the sign of ext. The current version works well for negative ext
        (self.value.clone() - self.ext.clone()) * (self.value.clone() + self.ext.clone() * E::sqr())
            - (E::sqr() - T::one()) * self.value.clone() * self.ext.clone()
    }
}
impl<T: Num + One, E: SqrtConst<T>> SqrtExt<T, E>
where
    for<'a> &'a T: AddMulSub<Output = T>,
{
    /// see `abs_sqr_ext`
    fn abs_sqr_ext_ref(&self) -> T {
        // overflows easily
        //&(&self.value * &self.value) - &(&(&self.ext * &self.ext) * &E::sqr())
        // less overflows (but slower) with rearranged form (value - ext)(value + ext*N) - (N-1)*value*ext
        &((&self.value - &self.ext) * (&self.value + &(&self.ext * &E::sqr())))
            - &((&E::sqr() - &T::one()) * (&self.value * &self.ext))
    }
}

impl<T: Num + Neg<Output = T>, E: SqrtConst<T>> Neg for SqrtExt<T, E> {
    type Output = SqrtExt<T, E>;
    fn neg(self) -> Self::Output {
        Self::new(-self.value, -self.ext)
    }
}

macro_rules! impl_add {
    ($Add:ident, $add:ident) => {
        impl<T: Num, E: SqrtConst<T>> $Add for SqrtExt<T, E>
        where
            for<'a> &'a T: $Add<Output = T>, // TODO can I move this $Add to the owned version?
        {
            type Output = SqrtExt<T, E>;
            fn $add(self, rhs: Self) -> Self::Output {
                SqrtExt::new(self.value.$add(&rhs.value), self.ext.$add(&rhs.ext))
            }
        }
        impl<'a, T: Num + $Add<T, Output = T>, E: SqrtConst<T>> $Add for &'a SqrtExt<T, E> {
            type Output = SqrtExt<T, E>;
            fn $add(self, rhs: Self) -> Self::Output {
                SqrtExt::new(
                    self.value.clone().$add(rhs.value.clone()),
                    self.ext.clone().$add(rhs.ext.clone()),
                )
            }
        }
        impl<T: Num, E: SqrtConst<T>> $Add<T> for SqrtExt<T, E>
        where
            for<'a> &'a T: $Add<Output = T>,
        {
            type Output = SqrtExt<T, E>;
            fn $add(self, rhs: T) -> Self::Output {
                SqrtExt::new(self.value.$add(&rhs), self.ext)
            }
        }
        impl<'a, T: Num + $Add<T, Output = T>, E: SqrtConst<T>> $Add<&'a T> for &'a SqrtExt<T, E> {
            type Output = SqrtExt<T, E>;
            fn $add(self, rhs: &'a T) -> Self::Output {
                SqrtExt::new(self.value.clone().$add(rhs.clone()), self.ext.clone())
            }
        }
    };
}
impl_add!(Add, add);
impl_add!(Sub, sub);

impl<T: Num + Add<Output = T>, E: SqrtConst<T>> Mul for SqrtExt<T, E>
where
    for<'a> &'a T: Mul<Output = T>,
{
    type Output = SqrtExt<T, E>;
    fn mul(self, rhs: Self) -> Self::Output {
        SqrtExt::new(
            &self.value * &rhs.value + &(&self.ext * &rhs.ext) * &E::sqr(),
            &self.ext * &rhs.value + &self.value * &rhs.ext,
        )
    }
}
impl<'a, T: Num + Add<Output = T> + Mul<Output = T>, E: SqrtConst<T>> Mul for &'a SqrtExt<T, E> {
    type Output = SqrtExt<T, E>;
    fn mul(self, rhs: Self) -> Self::Output {
        SqrtExt::new(
            self.value.clone() * rhs.value.clone() + self.ext.clone() * rhs.ext.clone() * E::sqr(),
            self.ext.clone() * rhs.value.clone() + self.value.clone() * rhs.ext.clone(),
        )
    }
}

impl<T: Num + One, E: SqrtConst<T>> Div for SqrtExt<T, E>
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
    type Output = SqrtExt<T, E>;
    fn div(self, rhs: SqrtExt<T, E>) -> Self::Output {
        let abs_sqr = rhs.abs_sqr_ext_ref();
        // TODO Consider depending on Cancel to have less overflows here, as one can cancel everything with abs_sqr first.
        Self::new(
            &(&(&self.value * &rhs.value) - &(&(&self.ext * &rhs.ext) * &E::sqr())) / &abs_sqr,
            &(&(&self.ext * &rhs.value) - &(&self.value * &rhs.ext)) / &abs_sqr,
        )
    }
}
impl<'a, T: Num + Zero + One + Sub<Output = T> + Div<Output = T>, E: SqrtConst<T>> Div
    for &'a SqrtExt<T, E>
{
    type Output = SqrtExt<T, E>;
    fn div(self, rhs: &'a SqrtExt<T, E>) -> Self::Output {
        let abs_sqr = rhs.abs_sqr_ext();
        SqrtExt::new(
            (self.value.clone() * rhs.value.clone()
                - self.ext.clone() * rhs.ext.clone() * E::sqr())
                / abs_sqr.clone(),
            (self.ext.clone() * rhs.value.clone() - self.value.clone() * rhs.ext.clone()) / abs_sqr,
        )
    }
}

impl<T: Num + Zero + One + Neg<Output = T> + Sub<Output = T> + Div<Output = T>, E: SqrtConst<T>>
    SqrtExt<T, E>
{
    pub fn recip(self) -> Self {
        let abs_sqr = self.abs_sqr_ext();
        Self::new(
            self.value.clone() / abs_sqr.clone(),
            -self.ext.clone() / abs_sqr,
        )
    }
}

// TODO in macro for Div as well:
impl<T: Num, E: SqrtConst<T>> Mul<T> for SqrtExt<T, E>
where
    for<'a> &'a T: Mul<Output = T>,
{
    type Output = SqrtExt<T, E>;
    fn mul(self, rhs: T) -> Self::Output {
        SqrtExt::new(&self.value * &rhs, &self.ext * &rhs)
    }
}
impl<'a, T: Num + Mul<Output = T>, E: SqrtConst<T>> Mul<&'a T> for &'a SqrtExt<T, E> {
    type Output = SqrtExt<T, E>;
    fn mul(self, rhs: &'a T) -> Self::Output {
        SqrtExt::new(
            self.value.clone() * rhs.clone(),
            self.ext.clone() * rhs.clone(),
        )
    }
}

// TODO remainder based on division?

impl<
    T: Num + Zero + One + RemEuclid + PartialOrd + Neg<Output = T> + Sub<Output = T> + Div<Output = T>,
    E: SqrtConst<T>,
> RemEuclid for SqrtExt<T, E>
where
    Self: PartialOrd,
{
    /// Euclidean division for √2 and √3. Otherwise use an extension to √N, which ensures `|r| < |d|`,
    /// but can't be used in a `gcd` or `lcm`. That is because Z[√5] is not a Euclidean domain.
    fn div_rem_euclid(&self, b: &Self) -> (Self, Self) {
        if b.value.is_zero() && b.ext.is_zero() {
            // invalid division -> invalid result, which still holds up the equation self = q * div + r = r;
            return (T::zero().into(), self.clone());
        }
        if self == b {
            return (T::one().into(), T::zero().into());
        }
        // for any E::sqr().abs(), which is not a perfect square,
        // one can arbitrarily closely approx the inverse of any number,
        // as the numbers are dense in the reals (because √N is irrational).
        // However that is not, what should be used here.
        // Instead a solution with as simple as possible numbers is searched.
        let mut denom = b.abs_sqr_ext();
        let mut numer = Self::new(
            self.value.clone() * b.value.clone() - self.ext.clone() * b.ext.clone() * E::sqr(),
            self.ext.clone() * b.value.clone() - self.value.clone() * b.ext.clone(),
        );
        /*if denom.is_unit() {
            // do the exact division. Required by definition of `div_rem_euclid`
            return (&numer * &(T::one() / denom), T::zero().into());
        }*/
        // compute `a/b = q.value + q.ext √N` using rational numbers, then round those.
        // before rounding, decompose √N = floor(√N) + (√N - floor(√N))
        // q = (q.value + q.ext floor(√N)) + q.ext (√N - floor(√N))
        // then round both components and recover the original form.
        // this results in an error < 1/2 in the integer part and < 1/2 * ((√N - floor(√N)) in the rest,
        // which makes the error in the result less than 1, which in turn
        // makes the remainder smaller than the divisor.
        // TODO by correctly combining the two errors, the error can be made
        // smaller than 1/2 and then a positive solution can be returned.

        // TODO handle the complex case √-1 correctly.
        // TODO check if this makes the gcd converge! Just because the remainder is smaller,
        // doesn't mean it converges, as these numbers are dense in the reals!

        numer.value = numer.value + numer.ext.clone() * E::floor();
        if !denom.is_valid_euclid() {
            denom = -denom;
            numer = -numer;
        }
        let ((n, nr), (m, mr), d2) = (
            numer.value.div_rem_euclid(&denom),
            numer.ext.div_rem_euclid(&denom),
            denom.clone() / (T::one() + T::one()),
        );
        // now choose the rounding, such that |nr/denom|^2 + |mr/denom|^2 N < 1
        let mut q = SqrtExt::new(
            if nr > d2 { n + T::one() } else { n },
            if mr > d2 { m + T::one() } else { m },
        );
        q.value = q.value - q.ext.clone() * E::floor();
        let mut r = self - &(b * &q);
        // add/subtract 0 < √N-floor(√N) < 1
        //let mut step = Self::new(-E::floor(), T::one());
        // add/subtract 1, this makes the gcd stable apparently (for N != 5) (TODO proof)
        let mut step = Self::new(T::one(), T::zero());
        if b.is_valid_euclid() {
            step = -step;
        }
        while !r.is_valid_euclid() {
            q += &step;
            r = self - &(b * &q);
        }
        (q, r)
    }
    fn is_valid_euclid(&self) -> bool {
        // a+b√N > 0 <=> a > -b√N <=> b > -a/√N
        // and for a >= 0: <=> a^2 > b^2 N
        // and for b >= 0: <=> b^2 N > a^2
        self >= &T::zero().into()
    }
}

macro_rules! forward_assign_impl {
    ($($AddAssign:ident, $Add:ident, $add_assign:ident, $add:ident);+) => {
        $(impl<T: Num + Add<Output = T>, E: SqrtConst<T>> $AddAssign for SqrtExt<T, E>
            where for<'a> &'a T: $Add<Output = T> {
            fn $add_assign(&mut self, rhs: SqrtExt<T, E>) {
                take(self, |x| x.$add(rhs));
            }
        }
        impl<T: Num, E: SqrtConst<T>> $AddAssign<T> for SqrtExt<T, E>
            where for<'a> &'a T: Add<Output = T> + $Add<Output = T> {
            fn $add_assign(&mut self, rhs: T) {
                take(self, |x| x.$add(rhs));
            }
        }
        impl<'a, T: Num + Add<Output = T> + $Add<Output = T>, E: SqrtConst<T>> $AddAssign<&'a SqrtExt<T, E>> for SqrtExt<T, E> {
            fn $add_assign(&mut self, rhs: &'a SqrtExt<T, E>) {
                take(self, |x| (&x).$add(rhs));
            }
        }
        impl<'a, T: Num + Add<Output = T> + $Add<Output = T>, E: SqrtConst<T>> $AddAssign<&'a T> for SqrtExt<T, E> {
            fn $add_assign(&mut self, rhs: &'a T) {
                take(self, |x| (&x).$add(rhs));
            }
        })+
    };
}
forward_assign_impl!(
    AddAssign, Add, add_assign, add; SubAssign, Sub, sub_assign, sub; MulAssign, Mul, mul_assign,
    mul //, DivAssign, Div, div_assign, div, RemAssign, Div, rem_assign, rem
);

impl<T: Num + Zero + Add<Output = T>, E: SqrtConst<T>> Sum for SqrtExt<T, E>
where
    for<'a> &'a T: Add<Output = T>,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::new(T::zero(), T::zero()), |acc, c| acc + c)
    }
}
impl<'a, T: Num + Zero + Add<Output = T>, E: SqrtConst<T>> Sum<&'a SqrtExt<T, E>>
    for SqrtExt<T, E>
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a SqrtExt<T, E>>,
    {
        iter.fold(Self::new(T::zero(), T::zero()), |acc, c| &acc + c)
    }
}

impl<T: Num + Zero + One + Add<Output = T>, E: SqrtConst<T>> Product for SqrtExt<T, E>
where
    for<'a> &'a T: Mul<Output = T>,
{
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, c| acc * c)
    }
}
impl<'a, T: Num + Zero + One + Add<Output = T>, E: SqrtConst<T>> Product<&'a SqrtExt<T, E>>
    for SqrtExt<T, E>
{
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a SqrtExt<T, E>>,
    {
        iter.fold(Self::new(T::one(), T::zero()), |acc, c| &acc * c)
    }
}

impl<T: Num + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>, E: SqrtConst<T>> Num
    for SqrtExt<T, E>
where
    T::Real: Add<Output = T::Real>
        + Mul<Output = T::Real>
        + Sub<Output = T::Real>
        + Neg<Output = T::Real>,
    Self: From<SqrtExt<T::Real, E::Real>>,
{
    type Real = SqrtExt<T::Real, E::Real>;
    #[inline(always)]
    fn abs_sqr(&self) -> Self::Real {
        // TODO correctly handle negative E::sqr()!!!
        let x = (self.value.clone() * self.ext.conj()).re();
        Self::Real::new(
            self.value.abs_sqr() + self.ext.abs_sqr() * E::sqr().re(),
            x.clone() + x,
        )
    }
    #[inline(always)]
    fn re(&self) -> Self::Real {
        // TODO handle negative E::sqr() correctly!!!
        Self::Real::new(self.value.re(), self.ext.re())
    }
    #[inline(always)]
    fn is_unit(&self) -> bool {
        let abs_sqr = self.value.clone() * self.value.clone()
            - self.ext.clone() * self.ext.clone() * E::sqr();
        abs_sqr.is_unit()
    }
}

impl<
    F: FloatType,
    T: Num + Cancel + PartialOrd + Div<Output = T> + Neg<Output = T>,
    E: SqrtConst<T>,
> ApproxFloat<F> for SqrtExt<T, E>
where
    T: ApproxFloat<F>,
    for<'a> &'a T: AddMulSub<Output = T>,
{
    // Find a close approximation to a float using small integers.
    fn from_approx(mut value: F, mut tol: F) -> Option<Self> {
        if !value.is_finite() || tol < F::zero() {
            return None;
        }
        // The possibly simplest method is binary search.
        // 1. get the integer part right
        // 2. add k*(√N - floor(√N))^n, where k is an integer and n indicates the repetition of this step.
        // this can also be inverted to no longer require binary search, when concrete types are used.
        // This is a kinda radix based number system, just that the radix is 0 < √N - floor(√N) < 1
        // We know how to convert into such a number system using the euclidean algorithm.
        // He we use rounding instead of floor to get smaller integers k.
        let r = Self::new(-E::floor(), T::one());
        let rf = r.to_approx();
        let mut p = Self::new(T::one(), T::zero());
        let mut result = Self::new(T::zero(), T::zero());
        let half = F::one() / (F::one() + F::one());
        loop {
            let xf = value.round();
            let x = T::from_approx(xf.clone(), half.clone())?;
            result += &(&p * &x);
            p *= &r;
            value = (value - xf) / rf.clone();
            tol = tol / rf.clone();
            if value.abs() <= tol.abs() {
                return Some(result);
            }
        }
    }
    #[inline]
    fn to_approx(&self) -> F {
        let c: F = E::sqr().to_approx();
        // evaluate to float, but be very careful to avoid cancellation
        if E::is_negative()
            || self.value.is_zero()
            || self.ext.is_zero()
            || self.value.is_valid_euclid() == self.ext.is_valid_euclid()
        {
            // no cancelation, as both have the same sign.
            let a: F = self.value.to_approx();
            let b: F = self.ext.to_approx();
            a + b * c.sqrt()
        } else {
            // cancel some of the integer part using floor first:
            let mut q = self.clone();
            q.value = q.value + q.ext.clone() * E::floor();
            let rf = c.sqrt() - E::floor().to_approx();
            let mut f = rf * q.ext.to_approx() + q.value.to_approx();
            if self.value.is_zero() || self.value.is_valid_euclid() == self.ext.is_valid_euclid() {
                return f;
            }
            // to remove the remaining cancelation error use Newton iteration:
            // x-y√N = f -> N y^2 = (x - f)^2 = x^2 - 2xf + f^2
            // -> F(f) = f^2 - 2xf + (x^2 - N y^2)
            // -> F'(f) = 2f - 2x
            let x: F = self.value.to_approx();
            // huge exact integer cancelation! Easily overflows...
            // This is what makes this converge to the correct solution.
            let sqr: F = self.abs_sqr_ext().to_approx();
            let two = F::one() + F::one();
            let mut last_v = F::zero();
            // limited to 100 to avoid endless loops at all cost.
            // Theoretically, I think there can be floats, which cycle between two tiny numbers for v.
            for _ in 0..100 {
                let d = f.clone() - x.clone(); // derivative
                let v = f.clone() * (d.clone() - x.clone()) + sqr.clone();
                if v.is_zero() || v == last_v {
                    break;
                }
                last_v = v.clone();
                f = f - v / (two.clone() * d);
            }
            f
        }
    }
}

// Safety: `Complex<T>` is `repr(C)` and contains only instances of `T`, so we
// can guarantee it contains no *added* padding. Thus, if `T: Zeroable`,
// `SqrtExt<T, E>` is also `Zeroable`
#[cfg(feature = "bytemuck")]
unsafe impl<T: Num + bytemuck::Zeroable, E: Copy + SqrtConst<T>> bytemuck::Zeroable
    for SqrtExt<T, E>
{
}

// Safety: `Complex<T>` is `repr(C)` and contains only instances of `T`, so we
// can guarantee it contains no *added* padding. Thus, if `T: Pod`,
// `SqrtExt<T, E>` is also `Pod`
#[cfg(feature = "bytemuck")]
unsafe impl<T: Num + bytemuck::Pod, E: 'static + Copy + SqrtConst<T>> bytemuck::Pod
    for SqrtExt<T, E>
{
}

// String conversions
macro_rules! impl_formatting {
    ($Display:ident, $prefix:expr, $fmt_str:expr, $fmt_alt:expr) => {
        #[cfg(feature = "std")]
        impl<T: Num + fmt::$Display + Clone + Zero + One, E: SqrtConst<T>> fmt::$Display
            for SqrtExt<T, E>
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.ext.is_zero() {
                    return <T as fmt::$Display>::fmt(&self.value, f);
                }
                let value = if f.alternate() {
                    &std::format!($fmt_alt, self.value)
                } else {
                    &std::format!($fmt_str, self.value)
                };
                let ext = if self.ext.is_zero() || self.ext.is_one() {
                    ""
                } else if f.alternate() {
                    &std::format!($fmt_alt, self.ext)
                } else {
                    &std::format!($fmt_str, self.ext)
                };
                let ext_no_sign = ext.strip_prefix("-");
                let is_number = !ext_no_sign
                    .unwrap_or(ext)
                    .chars()
                    .any(|c| c == '+' || c == '-' || c == '/');

                let sqr = if f.alternate() {
                    &std::format!($fmt_alt, E::sqr())
                } else {
                    &std::format!($fmt_str, E::sqr())
                };
                let sqr_no_sign = sqr.strip_prefix("-");
                let sqr_is_number = sqr_no_sign.unwrap_or(sqr).chars().all(|c| c.is_alphanumeric());
                let sqr = if sqr_is_number {
                    std::format!("√{}", sqr)
                } else {
                    std::format!("√({})", sqr)
                };

                let pre_pad = if self.ext.is_one() {
                    &if self.value.is_zero() {
                        sqr
                    } else {
                        std::format!(concat!($fmt_str, "+{}"), self.value, sqr)
                    }
                } else {
                    &if self.value.is_zero() {
                        if is_number {
                            std::format!("{}{}", ext, sqr)
                        } else {
                            std::format!("({}){}", ext, sqr)
                        }
                    } else {
                        if is_number {
                            if ext_no_sign.is_some() {
                                std::format!("{}{}{}", value, ext, sqr)
                            } else {
                                std::format!("{}+{}{}", value, ext, sqr)
                            }
                        } else {
                            std::format!("{}+({}){}", value, ext, sqr)
                        }
                    }
                };
                if let Some(pre_pad) = pre_pad.strip_prefix("-") {
                    if let Some(pre_pad) = pre_pad.strip_prefix($prefix) {
                        f.pad_integral(false, $prefix, pre_pad)
                    } else {
                        f.pad_integral(false, "", pre_pad)
                    }
                } else {
                    if let Some(pre_pad) = pre_pad.strip_prefix($prefix) {
                        f.pad_integral(true, $prefix, pre_pad)
                    } else {
                        f.pad_integral(true, "", &pre_pad)
                    }
                }
            }
            #[cfg(not(feature = "std"))]
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.ext.is_zero() {
                    return <T as fmt::$Display>::fmt(&self.value, f);
                }
                // can't do any of the checking, so just always print with parenthesis to be on the safe side
                if self.value.is_zero() {
                    if f.alternate() {
                        write!(
                            f,
                            concat!("(", $fmt_alt, ")√", $fmt_alt),
                            self.ext,
                            E::sqr()
                        )
                    } else {
                        write!(
                            f,
                            concat!("(", $fmt_str, ")√", $fmt_str),
                            self.ext,
                            E::sqr()
                        )
                    }
                } else if f.alternate() {
                    write!(
                        f,
                        concat!("(", $fmt_alt, ")+(", $fmt_alt, ")√", $fmt_alt),
                        self.value,
                        self.ext,
                        E::sqr()
                    )
                } else {
                    write!(
                        f,
                        concat!("(", $fmt_str, ")+(", $fmt_str, ")√", $fmt_str),
                        self.value,
                        self.ext,
                        E::sqr()
                    )
                }
            }
        }
    };
}

impl_formatting!(Display, "", "{}", "{:#}");
impl_formatting!(Octal, "0o", "{:o}", "{:#o}");
impl_formatting!(Binary, "0b", "{:b}", "{:#b}");
impl_formatting!(LowerHex, "0x", "{:x}", "{:#x}");
impl_formatting!(UpperHex, "0x", "{:X}", "{:#X}");
impl_formatting!(LowerExp, "", "{:e}", "{:#e}");
impl_formatting!(UpperExp, "", "{:E}", "{:#E}");

impl<T: Num + fmt::Debug, E: SqrtConst<T>> fmt::Debug for SqrtExt<T, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        #[derive(Debug)]
        #[allow(dead_code)] // wrong warning, Debug derive reads the fields.
        struct SqrtExt<'a, T> {
            value: &'a T,
            ext: &'a T,
        }
        let Self { value, ext, .. } = self;
        fmt::Debug::fmt(&SqrtExt { value, ext }, f)
    }
}

#[cfg(feature = "serde")]
impl<T: Num + serde::Serialize, E: SqrtConst<T>> serde::Serialize for SqrtExt<T, E> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (&self.value, &self.ext).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Num + serde::Deserialize<'de>, E: SqrtConst<T>> serde::Deserialize<'de>
    for SqrtExt<T, E>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (value, ext) = serde::Deserialize::deserialize(deserializer)?;
        Ok(Self::new(value, ext))
    }
}
