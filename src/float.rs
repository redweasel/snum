use core::ops::*;

use crate::*;
#[cfg(feature = "rational")]
use crate::rational::Ratio;


/// Type for floats used with [ApproxFloat]. Otherwise it is never explicitly required.
#[cfg(any(feature = "std", feature = "libm"))]
pub trait FloatType: Clone
+ Zero
+ One
+ FromU64
+ PartialOrd
+ Cancel
+ Num<Real = Self>
+ IntoDiscrete<Output = Self>
+ Div<Output = Self>
+ Neg<Output = Self>
+ NumElementary {
    /// check if the number is finite. This can often also be done by checking `self == self`.
    fn is_finite(&self) -> bool;
    /// check if the number is finite. This can often also be done by checking `self == self && !(self - self).is_zero()`.
    fn is_infinite(&self) -> bool;
}
#[cfg(not(any(feature = "std", feature = "libm")))]
pub trait FloatType: Clone
+ Zero
+ One
+ FromU64
+ PartialOrd
+ Cancel
+ Num<Real = Self>
+ IntoDiscrete<Output = Self>
+ Div<Output = Self>
+ Neg<Output = Self> {
    /// check if the number is finite. This can often also be done by checking `self == self`.
    fn is_finite(&self) -> bool;
    /// check if the number is finite. This can often also be done by checking `self == self && !(self - self).is_zero()`.
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

impl ApproxFloat<f32> for f64 {
    #[inline(always)]
    fn to_approx(&self) -> f32 {
        *self as f32
    }
    #[inline(always)]
    fn from_approx(value: f32, _tol: f32) -> Option<Self> {
        Some(value as f64)
    }
}
impl ApproxFloat<f64> for f32 {
    #[inline(always)]
    fn to_approx(&self) -> f64 {
        *self as f64
    }
    #[inline(always)]
    fn from_approx(value: f64, tol: f64) -> Option<Self> {
        let v = value as f32;
        ((v as f64 - value).abs() <= tol).then_some(v)
    }
}

// adapted from num_traits
// returns f = mantissa * 2^exponent
#[cfg(any(feature = "rational", feature = "ibig"))]
fn integer_decode_f32(f: f32) -> (i32, i16, bool, bool) {
    let bits: u32 = f.to_bits();
    let mut exponent: i16 = ((bits >> 23) & 0xff) as i16;
    let finite = exponent != 0xff;
    let mut mantissa = bits & 0x7f_ffff;
    let zero_mantissa = mantissa == 0;
    if exponent == 0 {
        mantissa <<= 1;
    } else {
        mantissa |= 0x80_0000;
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
#[cfg(any(feature = "rational", feature = "ibig"))]
fn integer_decode_f64(f: f64) -> (i64, i16, bool, bool) {
    let bits: u64 = f.to_bits();
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let finite = exponent != 0x7ff;
    let mut mantissa = bits & 0xf_ffff_ffff_ffff;
    let zero_mantissa = mantissa == 0;
    if exponent == 0 {
        mantissa <<= 1;
    } else {
        mantissa |= 0x10_0000_0000_0000;
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

#[cfg(feature = "rational")]
mod rational {
    use super::*;
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
}

#[cfg(feature = "ibig")]
mod bigint {
    use super::*;
    macro_rules! impl_bigint {
        ($float:ident, $integer_decode:ident) => {
            #[cfg(feature = "rational")]
            impl From<$float> for Ratio<ibig::IBig> {
                fn from(value: $float) -> Self {
                    let (mantissa, exponent, finite, zero_mantissa) = $integer_decode(value);
                    let numer = ibig::IBig::from(mantissa);
                    let zero = ibig::IBig::from(0i64);
                    let one = ibig::IBig::from(1i64);
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
                            denom: one << (-exponent) as usize,
                        }
                    } else {
                        Ratio {
                            numer: numer << exponent as usize,
                            denom: one,
                        }
                    }
                }
            }

            impl ApproxFloat<$float> for ibig::IBig {
                #[inline(always)]
                fn to_approx(&self) -> $float {
                    self.to_f64() as $float
                }
                #[inline(always)]
                fn from_approx(value: $float, tol: $float) -> Option<Self> {
                    // in contrast to num_bigint, ibig doesn't have this functionallity build in.
                    let vf = value.round();
                    let (mantissa, exponent, finite, _) = $integer_decode(vf);
                    if !finite || exponent < 0 || (vf - value).abs() > tol {
                        return None;
                    }
                    Some(ibig::IBig::from(mantissa) << exponent as usize)
                }
            }
        };
    }
    impl_bigint!(f32, integer_decode_f32);
    impl_bigint!(f64, integer_decode_f64);
}