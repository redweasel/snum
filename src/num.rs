use core::{fmt::Debug, ops::*};

// TODO Zero and Conjugate could have derive macros just like Clone
// TODO add powi to the definition of a Field.
// TODO consider `CommutativeAdd` and `CommutativeMul` traits,
// which are just markers to make clear, that commutativity is given,
// so algorithms don't have to say in their description,
// whether they work for non commutative rings as well.
// TODO similarly add a `TrueDiv` trait, so integers can no longer be used with `Field`.
// anything that previously worked with `Field` and now doesn't, needs to use `Ring` with `RemEuclid`.
// Note, that is_unit can already provide a way to check if a division is a true division, however those checks should be in `debug_assert!()`.
// TODO consider moving `Real` to Conjugate.

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
            fn zero() -> $t {
                $z
            }
            #[inline(always)]
            fn is_zero(&self) -> bool {
                *self == $z
            }
        }
        impl One for $t {
            #[inline(always)]
            fn one() -> $t {
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
zero_one_impl!(num_bigint::BigInt; num_bigint::BigInt::ZERO, num_bigint::BigInt::from(1i8));
#[cfg(feature = "num-bigint")]
zero_one_impl!(num_bigint::BigUint; num_bigint::BigUint::ZERO, num_bigint::BigUint::from(1u8));

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

/// Represents a number, which can be real or complex, floating point or integer, signed or unsigned.
/// To differentiate between float and int, use `Num::Real: Ord`, as that is not implemented for floats.
/// To differentiate between signed and unsigned use `Num::Real: Neg<Output = Num::Real>`.
/// To differentiate between real and complex, use `T: Num<Real = T>` and/or `Num: PartialOrd`.
pub trait Num: Clone + Debug + From<Self::Real> + PartialEq + Conjugate {
    type Real: Num<Real = Self::Real>;
    /// real part of the number
    #[must_use]
    fn re(&self) -> Self::Real;
    /// absolute value squared
    #[must_use]
    fn abs_sqr(&self) -> Self::Real;
    /// check if the element has a multiplicative inverse `x * 1/x * x = x` (1/x might not be unique).
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

/// Squared norm of a vector or matrix based on [Num::abs_sqr]
pub trait NormSqr {
    type Output;
    #[must_use]
    fn norm_sqr(&self) -> Self::Output;
}
/// Norm based on [NormSqr]
pub trait Norm: NormSqr {
    #[must_use]
    fn norm(&self) -> Self::Output;
}
impl<S: ?Sized + NormSqr> Norm for S
where
    S::Output: NumAlgebraic,
{
    #[inline(always)]
    fn norm(&self) -> Self::Output {
        self.norm_sqr().sqrt()
    }
}

/// like [Rem<T, Output = T>] but with euclidean division.
pub trait RemEuclid: Sized {
    /// Compute the euclidean division `q` and remainder `r` of `self`, such that `self = q * div + r`.
    ///
    /// - If [Num] is implemented for `self`, the remainder is bounded by the denominator `div` with `|r| < |div|`. (e.g. complex numbers: `|r|^2 <= |div|^2/2`).
    /// - If the divisor is a unit, the remainder is also required to be 0.
    /// - If the number is signed, the positive sign solution is returned. In all cases `is_valid_euclid` needs to be true on the result remainder,
    /// - The result must make the Euclidean algorithm to compute the gcd convergent.
    /// - Division by zero may panic, or return the invalid value `(q=0, r=self)`.
    ///
    /// The implementation to always return `(q=0, r=self)` for non unit divisors IS NOT valid for number types, as `|r|` might be larger than `|div|`.
    /// For fields on the other hand, the implementation `(q=self/div, r=0)` IS valid, however fields may also implement a
    /// different notion of division here based on modulo, as that also fulfills the condition, but not with minimal `r`.
    /// Note, that even if only `r=0` is returned, `is_valid_euclid` still has to consider all positive numbers to be valid,
    /// due to it's condition that x or -x need to be valid.
    fn div_rem_euclid(&self, div: &Self) -> (Self, Self);
    /// Return, whether the number could be the result of a euclidean remainder.
    /// Note, that zero is always "valid euclid".
    ///
    /// For all x: x or -x need to be "valid euclid".
    fn is_valid_euclid(&self) -> bool;
}

macro_rules! impl_rem_euclid {
    ($($T:ty),+) => {
        $(impl RemEuclid for $T {
            #[inline(always)]
            fn div_rem_euclid(&self, div: &Self) -> (Self, Self) {
                (self.div_euclid(*div), self.rem_euclid(*div))
            }
            #[inline(always)]
            fn is_valid_euclid(&self) -> bool {
                // the compiler optimizes this away for integers, but this works for non finite floats.
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
        $(impl RemEuclid for $T {
            #[inline(always)]
            fn div_rem_euclid(&self, div: &Self) -> (Self, Self) {
                if !div.is_finite() {
                    return (0.0, 0.0); // for safety in gcd
                }
                (self.div_euclid(*div), self.rem_euclid(*div))
            }
            #[inline(always)]
            fn is_valid_euclid(&self) -> bool {
                // the compiler optimizes this away for integers, but this works for non finite floats.
                self >= &0.0 && self.is_finite()
            }
        })+
    };
}
impl_rem_euclid_float!(
    f32, f64
);

#[cfg(feature = "num-bigint")]
mod bigint {
    use crate::*;
    // TODO to my strong disliking I need this here... copying the implementation from num_bigint isn't possible because it relies on `is_positive()`, which is again from `num_traits`.
    use num_traits::{Signed, Euclid};

    impl RemEuclid for num_bigint::BigInt {
        fn div_rem_euclid(&self, div: &Self) -> (Self, Self) {
            <Self as Euclid>::div_rem_euclid(self, div)
        }
        fn is_valid_euclid(&self) -> bool {
            !self.is_negative()
        }
    }
    impl RemEuclid for num_bigint::BigUint {
        fn div_rem_euclid(&self, div: &Self) -> (Self, Self) {
            <Self as Euclid>::div_rem_euclid(self, div)
        }
        fn is_valid_euclid(&self) -> bool {
            true
        }
    }
}

/// Calculates the Greatest Common Divisor (GCD) of the number and
/// `other`. The result is always positive (> 0).
#[must_use]
pub fn gcd<T: Zero + One + Sub<Output = T> + RemEuclid>(mut a: T, mut b: T) -> T {
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
pub fn lcm<T: Clone + Zero + One + Sub<Output = T> + Div<T, Output = T> + RemEuclid>(
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
pub fn bezout<T: Clone + Zero + One + Sub<Output = T> + Mul<Output = T> + RemEuclid>(mut a: T, mut b: T) -> ((T, T), T) {
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
        (y0, y1) = (y1.clone(), y0 - q.clone() * y1); // TODO for b >= 2^63 this q * y1 can be outside of i64 range, but still in u64
        (x0, x1) = (x1.clone(), x0 - q * x1);
    }
    if b.is_valid_euclid() {
        ((x0, y0), b)
    } else {
        ((T::zero() - x0, T::zero() - y0), T::zero() - b)
    }
    
}

pub trait Cancel: Sized + Clone + Zero + One + Sub<Output = Self> + PartialEq + RemEuclid {
    #[must_use]
    fn cancel(self, b: Self) -> (Self, Self);
}
impl<T: Clone + Zero + PartialEq + One + Sub<Output = T> + RemEuclid> Cancel for T
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
    fn cancel(self, b: Self) -> (Self, Self) {
        if self == b {
            if self.is_zero() {
                return (T::zero(), T::zero());
            } else {
                return (T::one(), T::one());
            }
        }
        let gcd = gcd(self.clone(), b.clone());
        (&self / &gcd, &b / &gcd)
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
    fn floor(&self) -> Self::Output {
        f32::floor(*self)
    }
    fn ceil(&self) -> Self::Output {
        f32::ceil(*self)
    }
    fn round(&self) -> Self::Output {
        f32::round(*self)
    }
}
impl IntoDiscrete for f64 {
    type Output = f64;
    fn floor(&self) -> Self::Output {
        f64::floor(*self)
    }
    fn ceil(&self) -> Self::Output {
        f64::ceil(*self)
    }
    fn round(&self) -> Self::Output {
        f64::round(*self)
    }
}
macro_rules! impl_into_discrete_int {
    ($($t:ty),+) => {
        $(impl IntoDiscrete for $t {
            type Output = $t;
            fn floor(&self) -> Self::Output {
                *self
            }
            fn ceil(&self) -> Self::Output {
                *self
            }
            fn round(&self) -> Self::Output {
                *self
            }
        })+
    };
}
impl_into_discrete_int!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

macro_rules! signed_num_type {
    ($($type:ty),+) => {
        $(impl Num for $type {
            type Real = Self;
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
                self.is_one() || (-self).is_one()
            }
        })+
    };
}
macro_rules! unsigned_num_type {
    ($($type:ty),+) => {
        $(impl Num for $type {
            type Real = Self;
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
                self.is_one()
            }
        })+
    };
}
macro_rules! forward_math_impl {
    ($type:ty, $f: ident) => {
        #[inline(always)]
        fn $f(&self) -> Self {
            #[cfg(feature = "std")]
            {
                <$type>::$f(*self)
            }
            #[cfg(not(feature = "std"))]
            {
                libm::$f(*self)
            }
        }
    };
}
macro_rules! num_float_type {
    ($($type:ty),+) => {
        $(impl Num for $type {
            type Real = Self;
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
            forward_math_impl!($type, sqrt);
            forward_math_impl!($type, cbrt);
            #[inline(always)]
            fn abs(&self) -> Self::Real {
                <$type>::abs(*self)
            }
            #[inline(always)]
            fn sign(&self) -> Self {
                <$type>::one().copysign(*self)
            }
            #[inline(always)]
            fn copysign(&self, sign: &Self) -> Self {
                #[cfg(feature = "std")]
                { <$type>::copysign(*self, *sign) }
                #[cfg(not(feature = "std"))]
                { libm::copysign(*self, *sign) }
            }
        }
        impl NumAnalytic for $type {
            forward_math_impl!($type, sin);
            forward_math_impl!($type, cos);
            forward_math_impl!($type, tan);
            forward_math_impl!($type, asin);
            forward_math_impl!($type, acos);
            forward_math_impl!($type, atan);
            forward_math_impl!($type, exp);
            forward_math_impl!($type, exp_m1);
            forward_math_impl!($type, ln);
            forward_math_impl!($type, ln_1p);
            forward_math_impl!($type, sinh);
            forward_math_impl!($type, cosh);
            forward_math_impl!($type, tanh);
            forward_math_impl!($type, asinh);
            forward_math_impl!($type, acosh);
            forward_math_impl!($type, atanh);
            #[inline(always)]
            fn pow(&self, exp: &Self) -> Self {
                #[cfg(feature = "std")]
                { <$type>::powf(*self, *exp) }
                #[cfg(not(feature = "std"))]
                { libm::powf(*self, *exp) }
            }
            #[inline(always)]
            fn atan2(&self, x: &Self) -> Self {
                #[cfg(feature = "std")]
                { <$type>::atan2(*self, *x) }
                #[cfg(not(feature = "std"))]
                { libm::atan2(*self, *x) }
            }
        })+
    };
}
signed_num_type!(i8, i16, i32, i64, i128, isize);
unsigned_num_type!(u8, u16, u32, u64, u128, usize);
num_float_type!(f32, f64); // can't just do it for Float as that conflicts with the complex types
#[cfg(feature = "num-bigint")]
signed_num_type!(num_bigint::BigInt);
#[cfg(feature = "num-bigint")]
unsigned_num_type!(num_bigint::BigUint);

// the num_bigfloat library is incompatible with the operation constraints used in this library.
// TODO check again!
