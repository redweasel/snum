//! Implements a trait for converting integers to the desired numeric type.

pub trait FromU64 {
    /// Convert an unsigned integer into `Self`.
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
            fn from_u64(value: u64) -> Self {
                value as $type // accept errors here!
            }
        })+
    };
}
impl_from_float!(f32, f64);

#[cfg(feature = "num-bigint")]
mod bigint {
    use super::*;
    use num_bigint::*;

    impl FromU64 for BigInt {
        #[inline(always)]
        fn from_u64(value: u64) -> Self {
            value.to_bigint().unwrap()
        }
    }
    impl FromU64 for BigUint {
        #[inline(always)]
        fn from_u64(value: u64) -> Self {
            value.to_biguint().unwrap()
        }
    }
}