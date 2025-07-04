use crate::{One, Zero, Num};
use core::ops::{Add, Mul, Div};

pub trait PowerU {
    /// Compute the power with an unsigned integer exponent.
    ///
    /// Runtime complexity: O(log n) calls of `mul`.
    fn powu(&self, n: u64) -> Self;
}

impl<T: Clone + One> PowerU for T
where
    for<'a> &'a Self: Mul<&'a Self, Output = Self>,
{
    fn powu(&self, n: u64) -> Self {
        // overflow-free power algorithm for u64.
        // It's u64 instead of u32 because many types never overflow.
        // TODO add powi to the definition of a Field.
        if n == 0 {
            return Self::one();
        }
        let mut p = self.clone();
        let mut mask = n.midpoint(2).next_power_of_two() >> 1; // mask highest set bit
        while mask != 0 {
            p = &p * &p;
            if (n & mask) != 0 {
                p = &p * self;
            }
            mask >>= 1;
        }
        p
    }
}
pub trait PowerI {
    /// Compute the power with an unsigned integer exponent.
    ///
    /// Runtime complexity: O(log n) calls of `mul`.
    fn powi(&self, n: i64) -> Self;
}

impl<T: Num + One + Div<Output = T>> PowerI for T
where
    for<'a> &'a Self: Mul<&'a Self, Output = Self>,
{
    fn powi(&self, n: i64) -> Self {
        let x = if n < 0 {
            assert!(self.is_unit());
            T::one() / self.clone()
        } else {
            self.clone()
        };
        x.powu(n.unsigned_abs())
    }
}

pub trait IntMul {
    /// Multiply with an unsigned integer.
    ///
    /// Runtime complexity: O(log n) calls of `add`.
    ///
    /// This is rarely the best solution.
    fn mulu(&self, n: u64) -> Self;
}

impl<T: Clone + Zero> IntMul for T
where
    for<'a> &'a Self: Add<&'a Self, Output = Self>,
{
    fn mulu(&self, n: u64) -> Self {
        if n == 0 {
            return Self::zero();
        }
        let mut p = self.clone();
        let mut mask = n.midpoint(2).next_power_of_two() >> 1; // mask highest set bit
        while mask != 0 {
            p = &p + &p;
            if (n & mask) != 0 {
                p = &p + self;
            }
            mask >>= 1;
        }
        p
    }
}
