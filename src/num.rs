use core::{fmt::Debug, ops::*};

// TODO Zero and Conjugate could have derive macros just like Clone
// TODO add powi to the definition of a Field.
// TODO consider a `CommutativeAdd` and `CommutativeMul` trait,
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
    /// check if the element has a multiplicative inverse
    #[must_use]
    fn is_unit(&self) -> bool;
}

/// A number that supports sqrt and cbrt (E.g. a float).
/// Integers should not implement this, as their sqrt and cbrt are approximate.
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
    fn asinh(&self) -> Self;
    #[must_use]
    fn acosh(&self) -> Self;
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

/// like [Rem<T, Output = T>] but with euclidean division, so the remainder `r` is bounded by the denominator `d` e.g. with `|r| <= |d|`.
/// A valid (but trivial) implementation would be, to always return zero.
pub trait RemEuclid: Sized {
    /// Compute the euclidean division and remainder.
    fn div_rem_euclid(&self, div: &Self) -> (Self, Self);
    /// Return, whether the number could be the result of a euclidean remainder.
    /// Note, that zero is always "valid euclid".
    /// For all x: x or -x need to be "valid euclid".
    fn is_valid_euclid(&self) -> bool;
}

macro_rules! impl_rem_euclid {
    ($($T:ty),+) => {
        $(impl RemEuclid for $T {
            fn div_rem_euclid(&self, div: &Self) -> (Self, Self) {
                (self.div_euclid(*div), self.rem_euclid(*div))
            }
            fn is_valid_euclid(&self) -> bool {
                self >= &0
            }
        })+
    };
}
impl_rem_euclid!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);
// TODO add a careful implementation for floats, such that division is lossless? Addition and multiplication would still be lossy though... is it worth it?

#[cfg(feature = "num-bigint")]
mod bigint {
    use crate::*;
    // TODO to my strong disliking I need this here... copying the implementation from num_bigint isn't possible because it relies on `is_positive()`, which is again from `num_traits`.
    use num_traits::Euclid;

    impl RemEuclid for num_bigint::BigInt {
        fn div_rem_euclid(&self, div: &Self) -> (Self, Self) {
            <Self as Euclid>::div_rem_euclid(self, div)
        }
        fn is_valid_euclid(&self) -> bool {
            self >= &Zero::zero()
        }
    }
    impl RemEuclid for num_bigint::BigUint {
        fn div_rem_euclid(&self, div: &Self) -> (Self, Self) {
            (self / div, self % div)
        }
        fn is_valid_euclid(&self) -> bool {
            true
        }
    }
}

/// Calculates the Greatest Common Divisor (GCD) of the number and
/// `other`. The result is always positive (> 0).
/// If the number type is a floating point number, it returns
/// the absolute smallest non zero number, or 1 if both are 0.
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
            }
            else {
                return (T::one(), T::one());
            }
        }
        // negative equality check (overflows on i32::MIN, TODO see if there is a performance difference with this)
        /*if !b.is_valid_euclid() && self == &T::zero() - &b {
            return (T::one(), &T::zero() - &T::one());
        }
        if !self.is_valid_euclid() && b == &T::zero() - &self {
            return (&T::zero() - &T::one(), T::one());
        }*/
        // TODO what do signs do here?
        let gcd = gcd(self.clone(), b.clone());
        (&self / &gcd, &b / &gcd)
    }
}

/// Calculates the Least Common Multiple (LCM) of the number and
/// `other`. The result is always `result.is_valid_euclid() == true` (usually that means >= 0).
#[must_use]
pub fn lcm<T: Clone + Zero + One + Sub<Output = T> + Div<T, Output = T> + RemEuclid>(mut a: T, mut b: T) -> T {
    if !a.is_valid_euclid() {
        a = T::zero() - a;
    }
    if !b.is_valid_euclid() {
        b = T::zero() - b;
    }
    a.clone() * (b.clone() / gcd(a, b))
}

// TODO add extended gcd

pub trait IntPow {
    /// take the power of an integer and panic, if anything is out of range.
    fn ipow(&self, base: i32) -> Self;
}
macro_rules! impl_int_pow {
    ($($T:ty),+) => {
        $(impl IntPow for $T {
            fn ipow(&self, base: i32) -> Self {
                let base: $T = base.try_into().unwrap();
                base.pow((*self).try_into().unwrap())
            }
        })+
    };
}
impl_int_pow!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);
macro_rules! impl_int_pow_float {
    ($($T:ty),+) => {
        $(impl IntPow for $T {
            fn ipow(&self, base: i32) -> Self {
                let base: $T = base as $T; // accept small errors, as the result will be probably be NaN anyway in those cases.
                base.powf(*self)
            }
        })+
    };
}
impl_int_pow_float!(f32, f64);


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
                self != &0.0
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
            forward_math_impl!($type, asinh);
            forward_math_impl!($type, acosh);
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
