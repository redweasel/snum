use crate::rational::Ratio;

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

macro_rules! impl_from {
    ($float:ident, $integer_decode:ident) => {
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
    };
}
#[cfg(feature = "num-bigint")]
impl_from!(f32, integer_decode_f32);
#[cfg(feature = "num-bigint")]
impl_from!(f64, integer_decode_f64);
