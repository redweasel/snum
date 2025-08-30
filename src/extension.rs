//! Field extensions using a lightweight generic approach without default implementations.

use core::cmp::Ordering;
use core::fmt;
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::*;
use take_mut::take;

use crate::rational::Ratio;
use crate::*;

/// A constant representing `sqrt(Self::sqr())`
pub trait SqrtConst<T: Num> {
    type Real: SqrtConst<T::Real, Real = Self::Real>;
    /// constant for internal use to be able to check at compile time, if a SqrtConst is allowed.
    const _TEST_SQRT: () = ();
    /// Get the square of this sqrt constant, i.e. the content of the sqrt, which is representable in `T`.
    /// This constant is not allowed to be representable with `T`, as otherwise e.g. the equallity check
    /// will be wrong. It is also not allowed to be negative, as that would break comparisons and other
    /// implementations for types which do implement them. The constant does not need to be an integer.
    ///
    /// In theory it can also be a complex number, but note that for complex numbers
    /// `sqrt(z) = sqrt(|z|) * (z/|z| + 1)/2 = z/2 / sqrt(|z|) + 1/2 * sqrt(|z|)`, which means,
    /// that a `Ratio<SqrtExt<_, Sqrt<_, |z|>>>` with real type is sufficient.
    ///
    /// Note, there is nothing enforcing that this is a compile time constant,
    /// however if it changes at runtime, the values of all numbers using it, will change.
    fn sqr() -> T;
    /// Get the rounded down version of this square root by computing `floor(sqrt(Self::sqr()))`.
    /// For non integers, this returns the closest approximation, which is smaller than the real constant.
    fn floor() -> T;
}

/// Type for field extentions with square roots of natural numbers.
/// For runtime constants, use the `dynamic_sqrt_const!` macro.
pub struct Sqrt<T: FromU64, const N: u64>(PhantomData<T>); // not constructible
impl<T: FromU64 + Num, const N: u64> SqrtConst<T> for Sqrt<T, N>
where
    T::Real: Num<Real = T::Real> + FromU64,
{
    type Real = Sqrt<T::Real, N>;
    const _TEST_SQRT: () = {
        assert!(
            N.isqrt() * N.isqrt() != N,
            "N is not allowed to be a perfect square"
        );
        assert!(
            T::CHAR == 0 || N <= T::CHAR,
            "N is not allowed to be larger than the ring characteristic"
        );
        ()
    };
    #[inline(always)]
    fn sqr() -> T {
        let _ = Self::_TEST_SQRT;
        T::from_u64(N)
    }
    #[inline(always)]
    fn floor() -> T {
        T::from_u64(N.isqrt())
    }
}

#[macro_export]
#[cfg(feature = "std")]
/// Define a new square root constant using a `u64` that can depend on variables in the current scope.
/// The resulting type can only be used on the thread, on which this macro has been executed.
/// Note, this macro creates a global/static thread local variable, so use sparingly.
///
/// ### Examples
///
/// ```rust
/// use snum::{*, extension::*, rational::*};
/// for n in 5..9 {
///     dynamic_sqrt_const!(N, n);
///     println!("{}", Ratio::approx_sqrt::<N<i64>>(n));
/// }
/// ```
/// When using threads, one has to be careful. One can not move the type `N` into a different thread.
/// ```rust
/// use snum::{*, extension::*};
/// for n in 5..9 {
///     dynamic_sqrt_const!(N, n);
///     std::thread::spawn(move || {
///         // N isn't "sent" to the new thread, so using it, panics.
///         let _ = N::<i64>::sqr();
///     }).join().expect_err("thread should panic");
/// }
/// ```
/// To fix this, use it in a function like this:
/// ```rust
/// use snum::{*, extension::*};
/// // define a function, which works for any thread (this is always safe)
/// fn f(n: u64) -> i64 {
///     dynamic_sqrt_const!(N, n);
///     N::<i64>::sqr()
/// }
///
/// let threads: Vec<_> = (5..9).map(|n|
///     std::thread::spawn(move || {
///         assert_eq!(f(n), n as i64);
///     })
/// ).collect();
/// // threads running concurrently here
/// for t in threads {
///     t.join().expect("no thread should panic");
/// }
/// ```
/// Any misuse like this results in a panic or compile time error.
macro_rules! dynamic_sqrt_const {
    ($name:ident, $value:expr) => {
        std::thread_local! {
            static _SQR_CONST: core::cell::Cell<u64> = panic!("uninitialized constant");
            static _SQRT_CONST: core::cell::Cell<u64> = panic!("uninitialized constant");
        }
        {
            let n: u64 = $value;
            let nsqrt = n.isqrt();
            assert!(
                nsqrt * nsqrt != n,
                "The square root constant N={n}={nsqrt}^2 is not allowed to be a square number."
            );
            _SQR_CONST.set(n);
            _SQRT_CONST.set(nsqrt);
        }
        #[allow(dead_code)]
        struct $name<T: FromU64>(*mut T); // no Send or Sync
        impl<T: FromU64 + Num> SqrtConst<T> for $name<T>
        where
            T::Real: Num<Real = T::Real> + FromU64,
        {
            type Real = $name<T::Real>;
            #[inline(always)]
            fn sqr() -> T {
                T::from_u64(_SQR_CONST.get())
            }
            #[inline(always)]
            fn floor() -> T {
                T::from_u64(_SQRT_CONST.get())
            }
        }
    };
}

// TODO implement alternative macro to dynamic_sqrt_const for no_std

impl<T: Num + Cancel + Neg<Output = T> + PartialOrd> Ratio<T>
where
    for<'a> &'a T: AddMulSub<Output = T>,
{
    /// Linearly converging approximation of the square root of a constant.
    /// Computed in O(log n) time (n = iterations) and convergent with at least error `<= 2^-n`.
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
    /// Linearly converging rational approximation of this number in O(log n) time.
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
#[repr(C)]
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
    #[inline(always)]
    pub fn is_integral(&self) -> bool {
        self.ext.is_zero()
    }
}

impl<T: SafeDiv, E: SqrtConst<T>> SqrtExt<T, E>
where
    Self: SafeDiv + IntoDiscrete<Output = T>,
{
    /// Returns the (positive) fundamental unit != 1 derived from [One] in `T`: x+y√N (satisfies `(x^2 - y^2 N).is_unit()`, `x,y > 0`)
    /// The number x-y√N (the generalized conjugate) is the inverse and all other units are integer powers of this one.
    /// See <https://en.wikipedia.org/wiki/Dirichlet%27s_unit_theorem> for more information.
    ///
    /// This (non const) function is computing it, so avoid calling it multiple times and precompute it, if possible.
    pub fn unit() -> Self {
        if E::sqr().is_unit() {
            return SqrtExt::new(T::zero(), T::one());
        }
        // Based on continued fractions.
        // A cycle in the continued fraction development gives the fundamental unit != 1.
        // https://en.wikipedia.org/wiki/Pell%27s_equation
        // To compute this faster, find the square free factorisation first and split off the square part.
        // (this is the computationally difficult part with O(cbrt(N)))
        // then compute the unit of the reduced problem. The solution of the complete problem is a power of
        // the reduced problems unit, which can be computed efficiently using modulo arithmetic.
        // However this reduction is left to the user of the library, as it's only beneficial for some numbers
        // and a slowdown for others (e.g. semiprimes).

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

impl<T: Num + Cancel + IntoDiscrete + Neg<Output = T> + PartialOrd, E: SqrtConst<T>> IntoDiscrete
    for SqrtExt<T, E>
where
    <T as IntoDiscrete>::Output: fmt::Debug
        + IntoDiscrete<Output = <T as IntoDiscrete>::Output>
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
            // Note, it would be much better to have integers here, but that requires e.g. TryInto<u64>, as I would need E::floor() as u64
            let v = self.ext.clone() * E::floor();
            let w = Self::new(T::zero(), self.ext.clone());
            assert!(w > v.clone().into());
            let mut a = T::zero().floor();
            // Note, if ext is a non finite float, this will result in an endless loop!
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

impl<T: Num, E: SqrtConst<T>> SqrtExt<T, E> {
    pub fn try_from<Q: Num + TryInto<T>, E2: SqrtConst<Q>>(value: SqrtExt<Q, E2>) -> Option<Self> {
        if E::sqr() != E2::sqr().try_into().ok()? {
            return None;
        }
        Some(Self::new(
            value.value.try_into().ok()?,
            value.ext.try_into().ok()?,
        ))
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
        // equallity of the constants can not be checked at compile time here.
        assert_eq!(
            E2::sqr(),
            E::sqr().into(),
            "can not convert between SqrtExt with different constants"
        );
        SqrtExt::new(value.value.into(), value.ext.into())
    }
}
impl<T: Num<Real = T>, E: SqrtConst<T>, E2: SqrtConst<Complex<T>>> From<Complex<SqrtExt<T, E>>>
    for SqrtExt<Complex<T>, E2>
where
    Complex<T>: Num<Real = T>,
{
    fn from(value: Complex<SqrtExt<T, E>>) -> Self {
        // equallity of the constants can not be checked at compile time here.
        assert_eq!(
            E2::sqr(),
            E::sqr().into(),
            "can not convert between SqrtExt with different constants"
        );
        SqrtExt::new(
            Complex::new(value.re.value, value.im.value),
            Complex::new(value.re.ext, value.im.ext),
        )
    }
}
impl<T: Num<Real = T>, E: SqrtConst<T>, E2: SqrtConst<Complex<T>>> From<SqrtExt<Complex<T>, E2>>
    for Complex<SqrtExt<T, E>>
where
    Complex<T>: Num<Real = T>,
{
    fn from(value: SqrtExt<Complex<T>, E2>) -> Self {
        // equallity of the constants can not be checked at compile time here.
        assert_eq!(
            E2::sqr(),
            E::sqr().into(),
            "can not convert between SqrtExt with different constants"
        );
        Complex::new(
            SqrtExt::new(value.value.re, value.ext.re),
            SqrtExt::new(value.value.im, value.ext.im),
        )
    }
}
impl<T: Num + Cancel, E: SqrtConst<T>, E2: SqrtConst<Ratio<T>>> From<SqrtExt<Ratio<T>, E2>>
    for Ratio<SqrtExt<T, E>>
where
    Ratio<T>: Num,
{
    fn from(value: SqrtExt<Ratio<T>, E2>) -> Self {
        // equallity of the constants can not be checked at compile time here.
        assert_eq!(
            E2::sqr(),
            E::sqr().into(),
            "can not convert between SqrtExt with different constants"
        );
        let (a, b) = value.value.denom.clone().cancel(value.ext.denom.clone());
        // TODO check behavior with non finite values
        Ratio {
            numer: SqrtExt::new(value.value.numer * b, value.ext.numer * a.clone()),
            denom: SqrtExt::from(a * value.ext.denom.clone()),
        }
    }
}
impl<T: Num + Cancel, E: SqrtConst<T>, E2: SqrtConst<Ratio<T>>> From<Ratio<SqrtExt<T, E>>>
    for SqrtExt<Ratio<T>, E2>
where
    Ratio<T>: Num,
    for<'a> &'a T: Mul<&'a T, Output = T>,
{
    fn from(value: Ratio<SqrtExt<T, E>>) -> Self {
        // equallity of the constants can not be checked at compile time here.
        assert_eq!(
            E2::sqr(),
            E::sqr().into(),
            "can not convert between SqrtExt with different constants"
        );
        // this cast ist also possible since all (finite, non zero) rational numbers have an inverse.
        &SqrtExt::new(
            Ratio::new_raw(value.numer.value, T::one()),
            Ratio::new_raw(value.numer.ext, T::one()),
        ) / &SqrtExt::new(
            Ratio::new_raw(value.denom.value, T::one()),
            Ratio::new_raw(value.denom.ext, T::one()),
        )
    }
}

impl<T: Num + FromU64 + Zero, E: SqrtConst<T>> FromU64 for SqrtExt<T, E> {
    #[inline(always)]
    fn from_u64(value: u64) -> Self {
        SqrtExt::from(T::from_u64(value))
    }
}

impl<T: Num + Neg<Output = T>, E: SqrtConst<T>> Conjugate for SqrtExt<T, E> {
    #[inline(always)]
    fn conj(&self) -> Self {
        Self::new(self.value.conj(), self.ext.conj())
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
        // or base it on floor?!?
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
        impl<T: Num + $Add<Output = T>, E: SqrtConst<T>> $Add for SqrtExt<T, E> {
            type Output = SqrtExt<T, E>;
            fn $add(self, rhs: Self) -> Self::Output {
                SqrtExt::new(self.value.$add(rhs.value), self.ext.$add(rhs.ext))
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
        Self::new(
            &(&(&self.value * &rhs.value) - &(&(&self.ext * &rhs.ext) * &E::sqr())) / &abs_sqr,
            &(&(&self.ext * &rhs.value) - &(&self.value * &rhs.ext)) / &abs_sqr,
        )
    }
}
impl<'a, T: Num + One + Add<Output = T> + Sub<Output = T> + Div<Output = T>, E: SqrtConst<T>> Div
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

impl<T: Num + One + Add<Output = T> + Sub<Output = T>, E: SqrtConst<T>> Rem for SqrtExt<T, E>
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
    type Output = SqrtExt<T, E>;
    fn rem(self, rhs: SqrtExt<T, E>) -> Self::Output {
        let d = self.clone() / rhs.clone();
        self - d * rhs
    }
}
impl<'a, T: Num + One + Add<Output = T> + Sub<Output = T> + Div<Output = T>, E: SqrtConst<T>> Rem
    for &'a SqrtExt<T, E>
{
    type Output = SqrtExt<T, E>;
    fn rem(self, rhs: &'a SqrtExt<T, E>) -> Self::Output {
        self - &(&(self / rhs) * rhs)
    }
}

macro_rules! impl_mul {
    ($($Mul:ident, $mul:ident);+) => {
        $(impl<T: Num, E: SqrtConst<T>> $Mul<T> for SqrtExt<T, E>
        where
            for<'a> &'a T: $Mul<Output = T>,
        {
            type Output = SqrtExt<T, E>;
            fn $mul(self, rhs: T) -> Self::Output {
                SqrtExt::new((&self.value).$mul(&rhs), (&self.ext).$mul(&rhs))
            }
        }
        impl<'a, T: Num + $Mul<Output = T>, E: SqrtConst<T>> $Mul<&'a T> for &'a SqrtExt<T, E> {
            type Output = SqrtExt<T, E>;
            fn $mul(self, rhs: &'a T) -> Self::Output {
                SqrtExt::new(
                    self.value.clone().$mul(rhs.clone()),
                    self.ext.clone().$mul(rhs.clone()),
                )
            }
        })+
    };
}
impl_mul!(Mul, mul; Div, div; Rem, rem);

impl<
    T: Num + Zero + One + Euclid + PartialOrd + Neg<Output = T> + Sub<Output = T> + Div<Output = T>,
    E: SqrtConst<T>,
> Euclid for SqrtExt<T, E>
where
    Self: PartialOrd,
{
    /// Euclidean division for √2 and √3. Otherwise use an extension to √N, which ensures `|r| < |d|`,
    /// but can't be used in a `gcd` or `lcm` as the set of |r| is not finite. That is because Z[√5] is not a Euclidean domain.
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
        // compute `a/b = q.value + q.ext √N` using rational numbers, then round those.
        // before rounding, decompose √N = floor(√N) + (√N - floor(√N))
        // q = (q.value + q.ext floor(√N)) + q.ext (√N - floor(√N))
        // then round both components and recover the original form.
        // this results in an error < 1/2 in the integer part and < 1/2 * ((√N - floor(√N)) in the rest,
        // which makes the error in the result less than 1, which in turn
        // makes the remainder abs_sqr smaller than the divisor abs_sqr.

        // if we consider the field-norm ||x|^2 - N|y|^2| = |x+y√N|*|x-y√N|, we can prove, that
        // there is more than one solution, as all solutions, where r is multiplied with a unit are also valid,
        // as they have the same norm (the norm is multiplicative). (there is infinitely many units)

        // If we consider the norm |z=x+y√N|^2=|x|^2+N|y|^2, there is a finite amount or remainders.
        // (e.x + (e.y)√N)*(b.x + (b.y)√N) = e.x*b.x + N e.y*b.y + (e.x*b.y + e.y*b.x)√N
        // -> |e.x*b.x + N e.y*b.y|^2 + N|e.x*b.y + e.y*b.x|^2 ≤ |b.x + N b.y|^2 / 4 + N|b.y + b.x|^2 / 4

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
        // round correctly
        let mut q = SqrtExt::new(
            if nr > d2 { n + T::one() } else { n },
            if mr > d2 { m + T::one() } else { m },
        );
        q.value = q.value - q.ext.clone() * E::floor();
        let mut r = self - &(b * &q);
        // The following breaks sqrt(2) and sqrt(3) as Euclidean domains, but it's needed to make the results positive.
        // add/subtract 1, this makes the gcd stable apparently (for N != 5) (TODO proof)
        let mut step = T::one();
        if b.is_valid_euclid() {
            step = -step;
        }
        // with "while" this can lead to infinite loops for floats, if r is non finite.
        while !r.is_valid_euclid() && r.value == r.value {
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

// can't use the global forward_assign_impl, as there is additional generic parameters here.
macro_rules! forward_assign_impl {
    ($($AddAssign:ident, ($($Owned:ident),*), ($($Add:ident),+), $add_assign:ident, $add:ident;)+) => {
        $(impl<T: Num + Add<Output = T> + Sub<Output = T> $(+ $Owned)* $(+ $Add<Output = T>)+, E: SqrtConst<T>> $AddAssign for SqrtExt<T, E>
            where for<'a> &'a T: Sized $(+ $Add<Output = T>)+ {
            fn $add_assign(&mut self, rhs: SqrtExt<T, E>) {
                take(self, |x| x.$add(rhs));
            }
        }
        impl<T: Num + Add<Output = T> + Sub<Output = T>, E: SqrtConst<T>> $AddAssign<T> for SqrtExt<T, E>
            where for<'a> &'a T: Add<Output = T> $(+ $Add<Output = T>)+ {
            fn $add_assign(&mut self, rhs: T) {
                take(self, |x| x.$add(rhs));
            }
        }
        impl<'a, T: Num + Add<Output = T> + Sub<Output = T> $(+ $Owned)* $(+ $Add<Output = T>)+, E: SqrtConst<T>> $AddAssign<&'a SqrtExt<T, E>> for SqrtExt<T, E> {
            fn $add_assign(&mut self, rhs: &'a SqrtExt<T, E>) {
                take(self, |x| (&x).$add(rhs));
            }
        }
        impl<'a, T: Num + Add<Output = T> + Sub<Output = T> $(+ $Owned)* $(+ $Add<Output = T>)+, E: SqrtConst<T>> $AddAssign<&'a T> for SqrtExt<T, E> {
            fn $add_assign(&mut self, rhs: &'a T) {
                take(self, |x| (&x).$add(rhs));
            }
        })+
    };
}
forward_assign_impl!(
    AddAssign, (), (Add), add_assign, add;
    SubAssign, (), (Sub), sub_assign, sub;
    MulAssign, (), (Mul), mul_assign, mul;
    DivAssign, (One), (Add, Sub, Mul, Div), div_assign, div;
    RemAssign, (One), (Add, Sub, Mul, Div, Rem), rem_assign, rem;
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

// the following is the most intuitive implementation of Num.
// However, interpreting the extension as "imaginary unit" and implementing
// conj etc wrt that would also work and have desirable properties wrt Euclid.
impl<T: Num + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>, E: SqrtConst<T>> Num
    for SqrtExt<T, E>
where
    T::Real: Num<Real = T::Real>
        + Add<Output = T::Real>
        + Mul<Output = T::Real>
        + Sub<Output = T::Real>
        + Neg<Output = T::Real>,
    Self: From<SqrtExt<T::Real, E::Real>>,
{
    type Real = SqrtExt<T::Real, E::Real>;
    const CHAR: u64 = T::CHAR;
    #[inline(always)]
    fn abs_sqr(&self) -> Self::Real {
        let x = (self.value.clone() * self.ext.conj()).re();
        Self::Real::new(
            self.value.abs_sqr() + self.ext.abs_sqr() * E::sqr().re(),
            x.clone() + x,
        )
    }
    #[inline(always)]
    fn re(&self) -> Self::Real {
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
    F: FloatType + NumAlgebraic,
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
    fn to_approx(&self) -> F {
        let c: F = E::sqr().to_approx();
        // evaluate to float, but be very careful to avoid cancellation
        if self.value.is_zero()
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

// Safety: `SqrtExt<T, _>` is `repr(C)` and contains only instances of `T`, so we
// can guarantee it contains no *added* padding. Thus, if `T: Zeroable`,
// `SqrtExt<T, E>` is also `Zeroable`
#[cfg(feature = "bytemuck")]
unsafe impl<T: Num + bytemuck::Zeroable, E: Copy + SqrtConst<T>> bytemuck::Zeroable
    for SqrtExt<T, E>
{
}

// Safety: `SqrtExt<T, _>` is `repr(C)` and contains only instances of `T`, so we
// can guarantee it contains no *added* padding. Thus, if `T: Pod`,
// `SqrtExt<T, E>` is also `Pod`
#[cfg(feature = "bytemuck")]
unsafe impl<T: Num + bytemuck::Pod, E: 'static + Copy + SqrtConst<T>> bytemuck::Pod
    for SqrtExt<T, E>
{
}

#[cfg(feature = "std")]
#[inline(never)]
fn fmt_sqrtext(
    f: &mut fmt::Formatter<'_>,
    value_zero: bool,
    ext_one: bool,
    value_args: fmt::Arguments<'_>,
    ext_args: fmt::Arguments<'_>,
    sqr_args: fmt::Arguments<'_>,
    prefix: &str,
) -> fmt::Result {
    let value = &std::fmt::format(value_args);
    let ext = if ext_one {
        ""
    } else {
        &std::fmt::format(ext_args)
    };
    let ext_no_sign = ext.strip_prefix("-");
    let is_number = !ext_no_sign
        .unwrap_or(ext)
        .chars()
        .any(|c| c == '+' || c == '-' || c == '/');

    let sqr = &std::fmt::format(sqr_args);
    let sqr_is_number = sqr.chars().all(|c| c.is_alphanumeric());
    let sqr = if sqr_is_number {
        std::format!("√{}", sqr)
    } else {
        std::format!("√({})", sqr)
    };

    let pre_pad = if ext_one {
        &if value_zero {
            sqr
        } else {
            std::format!("{}+{}", value, sqr)
        }
    } else {
        &if value_zero {
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
    // this can add zeros before parenthesis, however that is a user problem
    crate::rational::pad_expr(f, prefix, pre_pad)
}
#[cfg(not(feature = "std"))]
#[inline(never)]
fn fmt_sqrtext(
    f: &mut fmt::Formatter<'_>,
    value_zero: bool,
    ext_one: bool,
    value_args: fmt::Arguments<'_>,
    ext_args: fmt::Arguments<'_>,
    sqr_args: fmt::Arguments<'_>,
    _prefix: &str,
) -> fmt::Result {
    // can't do any of the checking, so just always print with parenthesis to be on the safe side
    if ext_one {
        if value_zero {
            write!(f, "√{}", sqr_args)
        } else {
            write!(f, "({})+√{}", value_args, sqr_args)
        }
    } else {
        if value_zero {
            write!(f, "({})√{}", ext_args, sqr_args)
        } else {
            write!(f, "({})+({})√{}", value_args, ext_args, sqr_args)
        }
    }
}

// String conversions
macro_rules! impl_formatting {
    ($Display:ident, $prefix:expr, $fmt_str:expr) => {
        impl<T: Num + fmt::$Display + Clone + Zero + One, E: SqrtConst<T>> fmt::$Display
            for SqrtExt<T, E>
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.ext.is_zero() {
                    return <T as fmt::$Display>::fmt(&self.value, f);
                }
                if let Some(prec) = f.precision() {
                    if f.alternate() {
                        fmt_sqrtext(
                            f,
                            self.value.is_zero(),
                            self.ext.is_one(),
                            format_args!(
                                concat!("{:#.prec$", $fmt_str, "}"),
                                self.value,
                                prec = prec
                            ),
                            format_args!(
                                concat!("{:#.prec$", $fmt_str, "}"),
                                self.ext,
                                prec = prec
                            ),
                            format_args!(concat!("{:#", $fmt_str, "}"), E::sqr()),
                            $prefix,
                        )
                    } else {
                        fmt_sqrtext(
                            f,
                            self.value.is_zero(),
                            self.ext.is_one(),
                            format_args!(
                                concat!("{:.prec$", $fmt_str, "}"),
                                self.value,
                                prec = prec
                            ),
                            format_args!(concat!("{:.prec$", $fmt_str, "}"), self.ext, prec = prec),
                            format_args!(concat!("{:", $fmt_str, "}"), E::sqr()),
                            $prefix,
                        )
                    }
                } else {
                    if f.alternate() {
                        fmt_sqrtext(
                            f,
                            self.value.is_zero(),
                            self.ext.is_one(),
                            format_args!(concat!("{:#", $fmt_str, "}"), self.value),
                            format_args!(concat!("{:#", $fmt_str, "}"), self.ext),
                            format_args!(concat!("{:#", $fmt_str, "}"), E::sqr()),
                            $prefix,
                        )
                    } else {
                        fmt_sqrtext(
                            f,
                            self.value.is_zero(),
                            self.ext.is_one(),
                            format_args!(concat!("{:", $fmt_str, "}"), self.value),
                            format_args!(concat!("{:", $fmt_str, "}"), self.ext),
                            format_args!(concat!("{:", $fmt_str, "}"), E::sqr()),
                            $prefix,
                        )
                    }
                }
            }
        }
    };
}

impl_formatting!(Display, "", "");
impl_formatting!(Octal, "0o", "o");
impl_formatting!(Binary, "0b", "b");
impl_formatting!(LowerHex, "0x", "x");
impl_formatting!(UpperHex, "0x", "X");
impl_formatting!(LowerExp, "", "e");
impl_formatting!(UpperExp, "", "E");

impl<T: Num + fmt::Debug, E: SqrtConst<T>> fmt::Debug for SqrtExt<T, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
