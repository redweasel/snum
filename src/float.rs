use core::ops::*;

use crate::{rational::Ratio, Cancel, IntoDiscrete, Num, NumAnalytic, One, Zero};

// adapted from num_traits
// returns f = mantissa * 2^exponent
fn integer_decode_f32(f: f32) -> (i32, i16, bool, bool) {
    let bits: u32 = f.to_bits();
    let mut exponent: i16 = ((bits >> 23) & 0xff) as i16;
    let finite = exponent != 0xff;
    let mut mantissa = bits & 0x7fffff;
    let zero_mantissa = mantissa == 0;
    if exponent == 0 {
        mantissa <<= 1;
    } else {
        mantissa |= 0x800000;
    }
    // Exponent bias + mantissa shift
    exponent -= 127 + 23;
    let mut mantissa = if bits >> 31 == 0 {
        mantissa as i32
    } else {
        -(mantissa as i32)
    };
    // cancel the mantissa
    let c = mantissa.trailing_zeros();
    mantissa >>= c;
    exponent += c as i16;
    (mantissa, exponent, finite, zero_mantissa)
}
// adapted from num_traits
// returns f = mantissa * 2^exponent
fn integer_decode_f64(f: f64) -> (i64, i16, bool, bool) {
    let bits: u64 = f.to_bits();
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let finite = exponent != 0x7ff;
    let mut mantissa = bits & 0xfffffffffffff;
    let zero_mantissa = mantissa == 0;
    if exponent == 0 {
        mantissa <<= 1;
    } else {
        mantissa |= 0x10000000000000;
    }
    // Exponent bias + mantissa shift
    exponent -= 1023 + 52;
    let mut mantissa = if bits >> 63 == 0 {
        mantissa as i64
    } else {
        -(mantissa as i64)
    };
    // cancel the mantissa
    let c = mantissa.trailing_zeros();
    mantissa >>= c;
    exponent += c as i16;
    (mantissa, exponent, finite, zero_mantissa)
}

/// Type for floats used with [ApproxFloat]. Otherwise it is never explicitly required.
pub trait FloatType: Clone
+ Zero
+ One
+ PartialOrd
+ Cancel
+ Num<Real = Self>
+ NumAnalytic
+ IntoDiscrete<Output = Self>
+ Div<Output = Self>
+ Neg<Output = Self> {
    fn is_finite(&self) -> bool;
    fn is_infinite(&self) -> bool;
}

/// A float type that can approximate `T`, e.g. `f32` can approximate `i64`.
///
/// Never panicing, but the results might be wrong if something
/// is out of range, like `f32::MAX as i32`, so
/// check the results if exact conversion is required.
pub trait ApproxFloat<F: FloatType>: Sized {
    /// from `T` to approximation with cutoff. E.g. i64 -> f32
    fn to_approx(&self) -> F;
    /// get `T` from approximation with some kind of rounding. E.g. f32 -> i64.
    /// Returns None, if the tolerance `error <= tol` can not be satisfied.
    /// The tolerance is the absolute tolerance.
    fn from_approx(value: F, tol: F) -> Option<Self>;
}

macro_rules! impl_approx_float {
    ($float:ty; $($int:ty),+) => {
        impl FloatType for $float {
            #[inline(always)]
            fn is_finite(&self) -> bool {
                <$float>::is_finite(*self)
            }
            #[inline(always)]
            fn is_infinite(&self) -> bool {
                <$float>::is_infinite(*self)
            }
        }
        // self approximation
        impl ApproxFloat<$float> for $float {
            #[inline(always)]
            fn to_approx(&self) -> $float {
                *self
            }
            #[inline(always)]
            fn from_approx(value: $float, _tol: $float) -> Option<Self> {
                Some(value)
            }
        }
        $(impl ApproxFloat<$float> for $int {
            #[inline(always)]
            fn to_approx(&self) -> $float {
                *self as $float
            }
            #[inline(always)]
            fn from_approx(value: $float, tol: $float) -> Option<Self> {
                if !value.is_finite() {
                    return None;
                }
                let v = value.round() as $int;
                ((v as $float - value).abs() <= tol).then_some(v)
            }
        })+
    };
}
impl_approx_float!(f32; i32, i64, i128);
impl_approx_float!(f64; i32, i64, i128);

// specific implementations of TryFrom for Ratio.
// note, f32::MAX can be represented using u128, but not using i128.
macro_rules! impl_try_from {
    ($float:ident, $integer_decode:ident, $($int:ident),+) => {
        $(impl TryFrom<$float> for Ratio<$int> {
            type Error = ();
            fn try_from(value: $float) -> Result<Self, Self::Error> {
                let (mantissa, exponent, finite, zero_mantissa) = $integer_decode(value);
                let numer = mantissa as $int; // this is why no smaller integers are implemented.
                if !finite && zero_mantissa {
                    Ok(Ratio { numer: numer.signum(), denom: 0 })
                }
                else if !finite {
                    Ok(Ratio { numer: 0, denom: 0 })
                }
                else if exponent <= 0 {
                    match (1 as $int).checked_shl((-exponent) as u32) {
                        Some(denom) => Ok(Ratio { numer, denom }),
                        None => Err(()),
                    }
                } else {
                    match numer.checked_shl(exponent as u32) {
                        Some(x) => if x >> exponent == numer {
                            Ok(Ratio { numer: x, denom: 1 })
                        } else {
                            Err(())
                        },
                        None => Err(()),
                    }
                }
            }
        })+
    };
}
impl_try_from!(f32, integer_decode_f32, i32, i64, i128);
impl_try_from!(f64, integer_decode_f64, i64, i128);

macro_rules! impl_bigint {
    ($float:ident, $integer_decode:ident, $from:ident) => {
        impl From<$float> for Ratio<num_bigint::BigInt> {
            fn from(value: $float) -> Self {
                let (mantissa, exponent, finite, zero_mantissa) = $integer_decode(value);
                let numer = num_bigint::BigInt::from(mantissa);
                let zero = num_bigint::BigInt::from(0i64);
                let one = num_bigint::BigInt::from(1i64);
                if !finite && zero_mantissa {
                    Ratio {
                        numer: one * mantissa.signum(),
                        denom: zero,
                    }
                } else if !finite {
                    Ratio {
                        numer: zero.clone(),
                        denom: zero,
                    }
                } else if exponent <= 0 {
                    Ratio {
                        numer,
                        denom: one << (-exponent) as u32,
                    }
                } else {
                    Ratio {
                        numer: numer << exponent as u32,
                        denom: one,
                    }
                }
            }
        }

        impl ApproxFloat<$float> for num_bigint::BigInt {
            #[inline(always)]
            fn to_approx(&self) -> $float {
                use num_traits::ToPrimitive;
                self.to_f64().unwrap() as $float
            }
            #[inline(always)]
            fn from_approx(value: $float, tol: $float) -> Option<Self> {
                use num_traits::FromPrimitive;
                let vf = value.round();
                let v = num_bigint::BigInt::$from(vf)?;
                ((vf - value).abs() <= tol).then_some(v)
            }
        }
    };
}
#[cfg(feature = "num-bigint")]
impl_bigint!(f32, integer_decode_f32, from_f32);
#[cfg(feature = "num-bigint")]
impl_bigint!(f64, integer_decode_f64, from_f64);
