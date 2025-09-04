//! Implements a trait for converting integers to the desired numeric type.

use core::ops::Neg;

pub trait FromU64 {
    /// Convert an unsigned integer into `Self`. This is meant for small efficient constants.
    /// 
    /// # Panics
    /// If this integer is considered out of range.
    fn from_u64(value: u64) -> Self;
}

macro_rules! impl_from {
    ($($type:ty),+) => {
        $(impl FromU64 for $type {
            #[inline(always)]
            fn from_u64(value: u64) -> Self {
                value.try_into().unwrap()
            }
        })+
    };
}
impl_from!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

macro_rules! impl_from_float {
    ($($type:ty),+) => {
        $(impl FromU64 for $type {
            #[inline(always)]
            #[allow(clippy::cast_precision_loss)]
            fn from_u64(value: u64) -> Self {
                value as $type // accept errors here!
            }
        })+
    };
}
impl_from_float!(f32, f64);

pub trait FromI64: FromU64 + Neg<Output = Self> {
    /// Convert a signed integer into `Self`. This is meant for small efficient constants.
    /// 
    /// # Panics
    /// If this integer is considered out of range.
    fn from_i64(value: i64) -> Self;
}

impl<T: FromU64 + Neg<Output = T>> FromI64 for T {
    fn from_i64(value: i64) -> Self {
        let a = value.unsigned_abs();
        if value < 0 {
            -T::from_u64(a)
        }
        else {
            T::from_u64(a)
        }
    }
}

#[cfg(feature = "ibig")]
mod bigint {
    use super::*;
    use ibig::*;

    impl FromU64 for IBig {
        #[inline(always)]
        fn from_u64(value: u64) -> Self {
            Self::from(value)
        }
    }
    impl FromU64 for UBig {
        #[inline(always)]
        fn from_u64(value: u64) -> Self {
            Self::from(value)
        }
    }
}