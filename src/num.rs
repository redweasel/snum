use core::{fmt::Debug, num::Wrapping, ops::*};

// TODO consider `CommutativeAdd` and `CommutativeMul` traits,
// which are just markers to make clear, that commutativity is given,
// so algorithms don't have to say in their description,
// whether they work for non commutative rings as well.

/// Defines an additive identity element for `Self`.
///
/// # Laws
///
/// ```text
/// a + 0 = a       ∀ a ∈ Self
/// 0 + a = a       ∀ a ∈ Self
/// ```
pub trait Zero: Sized + Add<Self, Output = Self> {
    /// Returns the additive identity element of `Self`, `0`.
    /// This function call is usually optimized away, so a constant isn't needed.
    fn zero() -> Self;

    /// Sets `self` to the additive identity element of `Self`, `0`.
    #[inline]
    fn set_zero(&mut self) {
        *self = Zero::zero();
    }

    /// Returns `true` if `self` is equal to the additive identity.
    fn is_zero(&self) -> bool;
}

/// Defines a multiplicative identity element for `Self`.
///
/// # Laws
///
/// ```text
/// a * 1 = a       ∀ a ∈ Self
/// 1 * a = a       ∀ a ∈ Self
/// ```
pub trait One: Sized + Mul<Self, Output = Self> {
    /// Returns the multiplicative identity element of `Self`, `1`.
    /// This function call is usually optimized away, so a constant isn't needed.
    fn one() -> Self;

    /// Sets `self` to the multiplicative identity element of `Self`, `1`.
    #[inline(always)]
    fn set_one(&mut self) {
        *self = One::one();
    }

    /// Returns `true` if `self` is equal to the multiplicative identity.
    fn is_one(&self) -> bool;
}

macro_rules! zero_one_impl {
    ($($t:ty),+; $z:expr, $o:expr) => {
        $(impl Zero for $t {
            #[inline(always)]
            fn zero() -> Self {
                $z
            }
            #[inline(always)]
            fn is_zero(&self) -> bool {
                *self == $z
            }
        }
        impl One for $t {
            #[inline(always)]
            fn one() -> Self {
                $o
            }
            #[inline(always)]
            fn is_one(&self) -> bool {
                *self == $o
            }
        })+
    };
}
// standard types need to get default implementations here
zero_one_impl!(usize, u8, u16, u32, u64, u128; 0, 1);
zero_one_impl!(isize, i8, i16, i32, i64, i128; 0, 1);
zero_one_impl!(f32, f64; 0.0, 1.0);
#[cfg(feature = "num-bigint")]
zero_one_impl!(num_bigint::BigInt; num_bigint::BigInt::from(0i8), num_bigint::BigInt::from(1i8));
#[cfg(feature = "num-bigint")]
zero_one_impl!(num_bigint::BigUint; num_bigint::BigUint::from(0u8), num_bigint::BigUint::from(1u8));

impl<T: Zero> Zero for Wrapping<T>
where
    Self: Add<Output = Self>,
{
    fn zero() -> Self {
        Wrapping(T::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}
impl<T: One> One for Wrapping<T>
where
    Self: Mul<Output = Self>,
{
    fn one() -> Self {
        Wrapping(T::one())
    }
    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

pub trait AddMul:
    Sized + Add<Output = <Self as AddMul>::Output> + Mul<Output = <Self as AddMul>::Output>
{
    type Output;
}
impl<'a, T: Sized> AddMul for &'a T
where
    &'a T: Add<Output = T> + Mul<Output = T>,
{
    type Output = T;
}

pub trait AddMulSub:
    AddMul<Output = <Self as AddMulSub>::Output> + Sub<Output = <Self as AddMulSub>::Output>
{
    type Output;
}
impl<'a, T: Sized> AddMulSub for &'a T
where
    &'a T: AddMul<Output = T> + Sub<Output = T>,
{
    type Output = T;
}

pub trait AddMulSubDiv:
    AddMulSub<Output = <Self as AddMulSubDiv>::Output> + Div<Output = <Self as AddMulSubDiv>::Output>
{
    type Output;
}
impl<'a, T: Sized> AddMulSubDiv for &'a T
where
    &'a T: AddMulSub<Output = T> + Div<Output = T>,
{
    type Output = T;
}

/// Any [Num] with [Zero] and [One] that can be negated, added, subtracted and multiplied
pub trait Ring: Num + Zero + Neg<Output = Self> + Sub<Output = Self>
where
    for<'a> &'a Self: AddMulSub<Output = Self>,
    for<'a> &'a Self::Real: AddMulSub<Output = Self::Real>,
    Self::Real: Num<Real = Self::Real> + Zero + Neg<Output = Self::Real>,
{
}
impl<'b, T: Num + Zero + Neg<Output = T> + Sub<Output = T>> Ring for T
where
    for<'a> &'a T: AddMulSub<Output = T>,
    for<'a> &'a T::Real: AddMulSub<Output = T::Real>,
    T::Real: Num<Real = T::Real> + Zero + Neg<Output = T::Real>,
{
}

/// Any [Ring] with [One] that can be divided
pub trait Field: Ring + One + Div<Output = Self>
where
    for<'a> &'a Self: AddMulSubDiv<Output = Self>,
    for<'a> &'a Self::Real: AddMulSubDiv<Output = Self::Real>,
    Self::Real: Ring<Real = Self::Real> + One,
{
}
impl<T: Ring + One + Div<Output = T>> Field for T
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
    for<'a> &'a T::Real: AddMulSubDiv<Output = T::Real>,
    T::Real: Ring<Real = T::Real> + One,
{
}

/// Any [Field] that also implements [NumAlgebraic]
pub trait AlgebraicField: Field + NumAlgebraic
where
    for<'a> &'a Self: AddMulSubDiv<Output = Self>,
    for<'a> &'a Self::Real: AddMulSubDiv<Output = Self::Real>,
    Self::Real: Field<Real = Self::Real> + NumAlgebraic<Real = Self::Real>,
{
}
impl<T: Field + NumAlgebraic> AlgebraicField for T
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
    for<'a> &'a T::Real: AddMulSubDiv<Output = T::Real>,
    T::Real: Field<Real = T::Real> + NumAlgebraic<Real = T::Real>,
{
}

/// General complex conjugate trait. To implement this for real types, use the macro `impl_conjugate_real!(<type>)`.
pub trait Conjugate {
    /// complex conjugate of the value
    #[must_use]
    fn conj(&self) -> Self;
}
#[macro_export]
macro_rules! impl_conjugate_real {
    ($($type:ty),+) => {
        $(impl Conjugate for $type {
            #[inline(always)]
            fn conj(&self) -> Self {
                self.clone()
            }
        })+
    };
}
impl_conjugate_real!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64
);
#[cfg(feature = "num-bigint")]
impl_conjugate_real!(num_bigint::BigInt, num_bigint::BigUint);

impl<T: Clone> Conjugate for Wrapping<T> {
    #[inline(always)]
    fn conj(&self) -> Self {
        self.clone() // assume wrapping types are real integers.
    }
}

/// Represents a number, which can be real or complex, floating point or integer, signed or unsigned.
/// To differentiate between float and int, use `Num::Real: Ord`, as that is not implemented for floats.
/// To differentiate between signed and unsigned use `Num::Real: Neg<Output = Num::Real>`.
/// To differentiate between real and complex, use `T: Num<Real = T>` and/or `Num: PartialOrd`.
pub trait Num: Clone + Debug + From<Self::Real> + PartialEq + Conjugate {
    type Real: Num<Real = Self::Real>;
    /// characteristic of a number ring. Limited to u64. If it is bigger, or not known at compile time, it's considered 0.
    const CHAR: u64;
    /// real part of the number
    #[must_use]
    fn re(&self) -> Self::Real;
    /// absolute value squared
    #[must_use]
    fn abs_sqr(&self) -> Self::Real;
    /// Check if the element has a multiplicative inverse `x * 1/x * x = x` (1/x might not be unique).
    ///
    /// For fields it's recommended to use this instead of zero checks,
    /// as non finite values are also not units and often indicate failure of an algorithm.
    #[must_use]
    fn is_unit(&self) -> bool;
}

/// A number that supports sqrt and cbrt (E.g. a float).
/// Integers should not implement this, as their sqrt and cbrt are approx.
// NOTE: general powf could be added here, but strictly speaking algebraic just means pow(p/q) with p, q integers.
// In any case, sqrt and cbrt are enough to get all analytic polynomial solutions working, which was the goal with this type.
pub trait NumAlgebraic: Num {
    #[must_use]
    fn sqrt(&self) -> Self;
    #[must_use]
    fn cbrt(&self) -> Self;
    /// absolute value/magnitude of the number.
    /// Note that it's possible that `x.abs_sqr().sqrt() != x.abs()`, but `x.abs() * x.abs() = x.abs_sqr()` should always hold.
    #[must_use]
    fn abs(&self) -> Self::Real;
    /// the sign/phase of a number, defined as `x.sign() = x / x.abs()` or one if `x.abs() == 0`.
    /// This is defined globally to allow for more efficient (e.g. division free) implementations.
    #[must_use]
    fn sign(&self) -> Self;
    /// copy the sign/phase of a number, defined as `x.copysign(y) = x.abs() * y.sign()`.
    /// This is defined globally to allow for more efficient (e.g. division free) implementations.
    #[must_use]
    fn copysign(&self, sign: &Self) -> Self;
}

/// A number that supports trigonometric and exponential functions.
pub trait NumAnalytic: NumAlgebraic {
    #[must_use]
    fn sin(&self) -> Self;
    #[must_use]
    fn cos(&self) -> Self;
    #[must_use]
    fn tan(&self) -> Self;
    #[must_use]
    fn asin(&self) -> Self;
    #[must_use]
    fn acos(&self) -> Self;
    #[must_use]
    fn atan(&self) -> Self;
    #[must_use]
    fn atan2(&self, x: &Self) -> Self;
    #[must_use]
    fn exp(&self) -> Self;
    #[must_use]
    fn exp_m1(&self) -> Self;
    #[must_use]
    fn ln(&self) -> Self;
    #[must_use]
    fn ln_1p(&self) -> Self;
    #[must_use]
    fn sinh(&self) -> Self;
    #[must_use]
    fn cosh(&self) -> Self;
    #[must_use]
    fn tanh(&self) -> Self;
    #[must_use]
    fn asinh(&self) -> Self;
    #[must_use]
    fn acosh(&self) -> Self;
    #[must_use]
    fn atanh(&self) -> Self;
    #[must_use]
    fn pow(&self, exp: &Self) -> Self;
}

/// like [Rem<T, Output = T>] but with euclidean division.
pub trait Euclid: Sized {
    /// Compute the euclidean division `q` and remainder `r` of `self / d`, such that `self = q * d + r`.
    ///
    /// There is two valid ways to interpret Euclidean division:
    /// 1. Just `0 ≤ |r| < |d|` with some norm `|.|`, usually choosing the minimal |r| solution. Makes sense in rings.
    /// 2. `q` is an integer, such that `0 ≤ |r| < |d|`. Makes sense in fields, where otherwise `r` can always be zero.
    ///
    /// The hard rules (Axioms) are:
    /// - There must exist a norm (in the strict mathematical sense!), such that `|r| < |d|`
    ///   and such that the elements in |r| are finite and computable without panics. This makes the gcd converge.
    /// - Division by zero returns the (potentially) invalid value `(q=0, r=self)`.
    /// - `is_valid_euclid` must be true for all resulting remainders.
    ///
    /// Most implementations should follow the additional rules:
    /// - If [Num] is implemented for `self`, the remainder is bounded by the denominator `d` with `|r| < |d|`. (e.g. complex numbers: `|r|^2 <= |div|^2/2`).
    /// - If the number implements [PartialOrd], the positive sign solution is returned.
    ///
    /// The implementation to always return `(q=0, r=self)` for non unit divisors IS NOT valid for number types, as the chosen norm `|r|` might be larger than `|div|`.
    /// For fields on the other hand, the implementation `(q=self/div, r=0)` IS valid, however fields may also implement a
    /// different notion of division here based on modulo, as that also fulfills the condition, just not with minimal `r`.
    fn div_rem_euclid(&self, div: &Self) -> (Self, Self);
    /// Return, whether the number could be the result of an Euclidean remainder.
    ///
    /// Axiom:
    /// - For all x: x or -x need to be "valid euclid". Therefore zero is always "valid euclid"
    fn is_valid_euclid(&self) -> bool;
}

macro_rules! impl_rem_euclid {
    ($($T:ty),+) => {
        $(impl Euclid for $T {
            #[inline(always)]
            fn div_rem_euclid(&self, div: &Self) -> (Self, Self) {
                (self.div_euclid(*div), self.rem_euclid(*div))
            }
            #[inline(always)]
            fn is_valid_euclid(&self) -> bool {
                self >= &0
            }
        })+
    };
}
impl_rem_euclid!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

macro_rules! impl_rem_euclid_float {
    ($($T:ty),+) => {
        $(impl Euclid for $T {
            #[inline(always)]
            fn div_rem_euclid(&self, div: &Self) -> (Self, Self) {
                if !div.is_finite() {
                    return (0.0, 0.0); // for safety in gcd
                }
                (self.div_euclid(*div), self.rem_euclid(*div))
            }
            #[inline(always)]
            fn is_valid_euclid(&self) -> bool {
                self >= &0.0 && self.is_finite()
            }
        })+
    };
}
impl_rem_euclid_float!(f32, f64);

impl<T: Clone + Zero + Euclid> Euclid for Wrapping<T> {
    /// Don't cancel Wrapping types, there is no useful norm on these and canceling doesn't extend the range.
    fn div_rem_euclid(&self, _div: &Self) -> (Self, Self) {
        (Wrapping(T::zero()), self.clone())
    }
    fn is_valid_euclid(&self) -> bool {
        self.0.is_valid_euclid()
    }
}

#[cfg(feature = "num-bigint")]
mod bigint {
    use crate::*;
    // num_bigint is written based on num_traits
    use num_traits::Signed;

    impl Euclid for num_bigint::BigInt {
        fn div_rem_euclid(&self, div: &Self) -> (Self, Self) {
            <Self as num_traits::Euclid>::div_rem_euclid(self, div)
        }
        fn is_valid_euclid(&self) -> bool {
            !self.is_negative()
        }
    }
    impl Euclid for num_bigint::BigUint {
        fn div_rem_euclid(&self, div: &Self) -> (Self, Self) {
            <Self as num_traits::Euclid>::div_rem_euclid(self, div)
        }
        fn is_valid_euclid(&self) -> bool {
            true
        }
    }
}

/// Calculates the Greatest Common Divisor (GCD) of the number and
/// `other`. The result is always positive (> 0).
#[must_use]
pub fn gcd<T: Zero + One + Sub<Output = T> + Euclid>(mut a: T, mut b: T) -> T {
    if a.is_zero() {
        return if b.is_zero() {
            T::one()
        } else {
            if b.is_valid_euclid() {
                b
            } else {
                T::zero() - b
            }
        };
    }
    while !b.is_zero() {
        (b, a) = (a.div_rem_euclid(&b).1, b);
    }
    if a.is_valid_euclid() {
        a
    } else {
        T::zero() - a
    }
}

/// Calculates the Least Common Multiple (LCM) of the number and
/// `other`. The result is always `result.is_valid_euclid() == true` (usually that means >= 0).
#[must_use]
pub fn lcm<T: Clone + Zero + One + Sub<Output = T> + Div<T, Output = T> + Euclid>(
    mut a: T,
    mut b: T,
) -> T {
    if !a.is_valid_euclid() {
        a = T::zero() - a;
    }
    if !b.is_valid_euclid() {
        b = T::zero() - b;
    }
    a.clone() * (b.clone() / gcd(a, b))
}

/// Extended Euclidean algorithm to solve Bézout's identity: `ax + by = d`.
/// Returns `((x, y), d)`.
/// Note that `d` differs from the gcd if both arguments are zero.
#[must_use]
pub fn bezout<T: Clone + Zero + One + Sub<Output = T> + Mul<Output = T> + Euclid>(
    mut a: T,
    mut b: T,
) -> ((T, T), T) {
    if a.is_zero() {
        return if b.is_zero() {
            ((T::zero(), T::zero()), T::zero())
        } else {
            if b.is_valid_euclid() {
                ((T::zero(), T::one()), b)
            } else {
                ((T::zero(), T::zero() - T::one()), T::zero() - b)
            }
        };
    }
    let mut x0 = T::zero();
    let mut x1 = T::one();
    let mut y0 = T::one();
    let mut y1 = T::zero();
    while !a.is_zero() {
        let q;
        ((q, a), b) = (b.div_rem_euclid(&a), a);
        (y0, y1) = (y1.clone(), y0 - q.clone() * y1);
        (x0, x1) = (x1.clone(), x0 - q * x1);
    }
    if b.is_valid_euclid() {
        ((x0, y0), b)
    } else {
        ((T::zero() - x0, T::zero() - y0), T::zero() - b)
    }
}

pub trait Cancel: Sized + Clone + Zero + One + Sub<Output = Self> + PartialEq + Euclid {
    /// Cancel two numbers by dividing by their (signed) greatest common divisor.
    /// This will keep the signs intact if one of the numbers is zero.
    /// 
    /// This is based on the assumption, that `Div` is exact if `Euclid::div_rem_euclid` has zero remainder.
    #[must_use]
    fn cancel(self, b: Self) -> (Self, Self);
}
impl<T: Clone + Zero + PartialEq + One + Sub<Output = T> + Div<Output = T> + Euclid> Cancel for T {
    fn cancel(self, b: Self) -> (Self, Self) {
        if self == b {
            if self.is_zero() {
                return (T::zero(), T::zero());
            } else {
                // Note, the sign is lost here!
                return (T::one(), T::one());
            }
        }
        let gcd = gcd(self.clone(), b.clone());
        //(self.div_rem_euclid(&gcd).0, b.div_rem_euclid(&gcd).0) // this messes with the signs...
        (self / gcd.clone(), b / gcd)
    }
}

pub trait SafeDiv: Num + Cancel + Div<Output = Self> {
    /// Compute the division, but only divide as much as possible.
    ///
    /// - If the type is a field, this is guaranteed to be the same as normal division.
    /// - If the type is an Euclidean ring, the result is the same as `cancel`, but with guaranteed positive sign on `q`.
    /// - On division by zero, return `(self, 0)`
    ///
    /// This is useful, as cancel might be defined for some fields, even if it doesn't work.
    ///
    /// Returns `(p, q)` such that `self/rhs = p/q` with "best" `q`, which is "valid euclid".
    #[must_use]
    fn safe_div(self, rhs: Self) -> (Self, Self);
}
impl<T: Num + Cancel + Div<Output = T>> SafeDiv for T {
    fn safe_div(mut self, mut rhs: Self) -> (Self, Self) {
        if rhs.is_zero() {
            (self, rhs)
        } else if rhs.is_unit() {
            (self / rhs, T::one())
        } else {
            (self, rhs) = self.cancel(rhs);
            // ensure q is positive
            if !rhs.is_valid_euclid() {
                rhs = T::zero() - rhs;
                self = T::zero() - self;
            }
            (self, rhs)
        }
    }
}

pub trait IntoDiscrete: PartialEq + From<Self::Output> {
    type Output: Clone + Zero + One;
    fn floor(&self) -> Self::Output;
    fn ceil(&self) -> Self::Output {
        let x = self.floor();
        if self == &Self::from(x.clone()) {
            x
        } else {
            x + One::one()
        }
    }
    fn round(&self) -> Self::Output;
}

impl IntoDiscrete for f32 {
    type Output = f32; // has to be f32, as impl From<i128> for f32 doesn't exist (and can't exist).
    #[inline(always)]
    fn floor(&self) -> Self::Output {
        f32::floor(*self)
    }
    #[inline(always)]
    fn ceil(&self) -> Self::Output {
        f32::ceil(*self)
    }
    #[inline(always)]
    fn round(&self) -> Self::Output {
        f32::round(*self)
    }
}
impl IntoDiscrete for f64 {
    type Output = f64;
    #[inline(always)]
    fn floor(&self) -> Self::Output {
        f64::floor(*self)
    }
    #[inline(always)]
    fn ceil(&self) -> Self::Output {
        f64::ceil(*self)
    }
    #[inline(always)]
    fn round(&self) -> Self::Output {
        f64::round(*self)
    }
}
macro_rules! impl_into_discrete_int {
    ($($t:ty),+) => {
        $(impl IntoDiscrete for $t {
            type Output = $t;
            #[inline(always)]
            fn floor(&self) -> Self::Output {
                self.clone()
            }
            #[inline(always)]
            fn ceil(&self) -> Self::Output {
                self.clone()
            }
            #[inline(always)]
            fn round(&self) -> Self::Output {
                self.clone()
            }
        })+
    };
}
impl_into_discrete_int!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);
#[cfg(feature = "num-bigint")]
impl_into_discrete_int!(num_bigint::BigInt, num_bigint::BigUint);

impl<T: Clone + PartialEq> IntoDiscrete for Wrapping<T>
where
    Self: Zero + One,
{
    type Output = Self;
    #[inline(always)]
    fn floor(&self) -> Self::Output {
        self.clone()
    }
    #[inline(always)]
    fn ceil(&self) -> Self::Output {
        self.clone()
    }
    #[inline(always)]
    fn round(&self) -> Self::Output {
        self.clone()
    }
}

macro_rules! int_num_type {
    ($($type:ty),+;$unit:expr) => {
        int_num_type!($($type, {<$type>::MAX as u64}),+;$unit);
    };
    ($($type:ty, $char:block),+;$unit:expr) => {
        $(impl Num for $type {
            type Real = Self;
            const CHAR: u64 = $char;
            #[inline(always)]
            fn abs_sqr(&self) -> Self::Real {
                self * self
            }
            #[inline(always)]
            fn re(&self) -> Self::Real {
                self.clone()
            }
            #[inline(always)]
            fn is_unit(&self) -> bool {
                $unit(self)
            }
        })+
    };
}
#[cfg(all(feature = "libm", not(feature = "std")))]
macro_rules! libm_func {
    (f32::$f64:ident/$f32:ident($a:ident$(, $b:ident)?)) => {
        libm::$f32(*$a$(, *$b)?)
    };
    (f64::$f64:ident/$f32:ident($a:ident$(, $b:ident)?)) => {
        libm::$f64(*$a$(, *$b)?)
    };
}
#[cfg(any(feature = "std", feature = "libm"))]
#[rustfmt::skip]
macro_rules! forward_math_impl {
    ($type:ident, $f:ident, $f32:ident) => {
        forward_math_impl!($type, $f, $f, $f32);
    };
    ($type:ident, $f:ident, $f64_libm:ident, $f32_libm:ident$(, $x:ident)?) => {
        #[inline(always)]
        fn $f(&self$(, $x: &Self)?) -> Self {
            #[cfg(feature = "std")]
            { <$type>::$f(*self$(, *$x)?) }
            #[cfg(not(feature = "std"))]
            { libm_func!($type::$f64_libm/$f32_libm(self$(, $x)?)) }
        }
    };
}
macro_rules! num_float_type {
    ($($type:ident),+) => {
        $(impl Num for $type {
            type Real = Self;
            const CHAR: u64 = 0;
            #[inline(always)]
            fn abs_sqr(&self) -> Self::Real {
                self * self
            }
            #[inline(always)]
            fn re(&self) -> Self::Real {
                self.clone()
            }
            #[inline(always)]
            fn is_unit(&self) -> bool {
                // every number has an (approx) inverse, except for 0 and non finite values.
                // inverse in the sense, that x * (1.0 / x) = 1.0
                self != &0.0 && self.is_finite()
            }
        }
        #[cfg(any(feature = "std", feature = "libm"))]
        impl NumAlgebraic for $type {
            forward_math_impl!($type, sqrt, sqrtf);
            forward_math_impl!($type, cbrt, cbrtf);
            #[inline(always)]
            fn abs(&self) -> Self::Real {
                <$type>::abs(*self)
            }
            #[inline(always)]
            fn sign(&self) -> Self {
                <$type>::one().copysign(*self)
            }
            forward_math_impl!($type, copysign, copysign, copysignf, sign);
        }
        #[cfg(any(feature = "std", feature = "libm"))]
        impl NumAnalytic for $type {
            forward_math_impl!($type, sin, sinf);
            forward_math_impl!($type, cos, cosf);
            forward_math_impl!($type, tan, tanf);
            forward_math_impl!($type, asin, asinf);
            forward_math_impl!($type, acos, acosf);
            forward_math_impl!($type, atan, atanf);
            forward_math_impl!($type, exp, expf);
            forward_math_impl!($type, exp_m1, expm1, expm1f);
            forward_math_impl!($type, ln, log, logf);
            forward_math_impl!($type, ln_1p, log1p, log1pf);
            forward_math_impl!($type, sinh, sinhf);
            forward_math_impl!($type, cosh, coshf);
            forward_math_impl!($type, tanh, tanhf);
            forward_math_impl!($type, asinh, asinhf);
            forward_math_impl!($type, acosh, acoshf);
            forward_math_impl!($type, atanh, atanhf);
            forward_math_impl!($type, atan2, atan2, atan2f, x);
            #[inline(always)]
            fn pow(&self, exp: &Self) -> Self {
                #[cfg(feature = "std")]
                { <$type>::powf(*self, *exp) }
                #[cfg(not(feature = "std"))]
                { libm_func!($type::pow/powf(self, exp)) }
            }
        })+
    };
}
int_num_type!(i8, i16, i32, i64, i128, isize; (|x| x == &1 || x == &-1));
int_num_type!(u8, u16, u32, u64, u128, usize; (|x| x == &1));
num_float_type!(f32, f64);
#[cfg(feature = "num-bigint")]
int_num_type!(num_bigint::BigInt, {0}; |x: &num_bigint::BigInt| x.is_one() || (-x).is_one());
#[cfg(feature = "num-bigint")]
int_num_type!(num_bigint::BigUint, {0}; One::is_one);

impl<T: Num> Num for Wrapping<T>
where
    Self: Mul<Output = Self>,
{
    type Real = Self;
    const CHAR: u64 = T::CHAR;
    fn abs_sqr(&self) -> Self::Real {
        self.clone() * self.clone()
    }
    fn re(&self) -> Self::Real {
        self.clone()
    }
    fn is_unit(&self) -> bool {
        self.0.is_unit()
    }
}
