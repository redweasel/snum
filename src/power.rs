use crate::{Num, One, Zero};
use core::ops::Div;

pub trait PowerU {
    /// Compute the power with an unsigned integer exponent.
    ///
    /// Runtime complexity: O(log n) calls of `mul`.
    #[must_use]
    fn powu(&self, n: u64) -> Self;
}

impl<T: Clone + One> PowerU for T {
    fn powu(&self, n: u64) -> Self {
        // overflow-free power algorithm for u64.
        // It's u64 instead of u32 because many types never overflow.
        if n == 0 {
            return Self::one();
        }
        let mut p = self.clone();
        let mut mask = n.midpoint(2).next_power_of_two() >> 1; // mask highest set bit
        while mask != 0 {
            p = p.clone() * p;
            if (n & mask) != 0 {
                p = p * self.clone();
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
    #[must_use]
    fn powi(&self, n: i64) -> Self;
}

impl<T: Num + One + Div<Output = T>> PowerI for T {
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
    #[must_use]
    fn mulu(&self, n: u64) -> Self;
}

impl<T: Clone + Zero> IntMul for T {
    fn mulu(&self, n: u64) -> Self {
        if n == 0 {
            return Self::zero();
        }
        let mut p = self.clone();
        let mut mask = n.midpoint(2).next_power_of_two() >> 1; // mask highest set bit
        while mask != 0 {
            p = p.clone() + p;
            if (n & mask) != 0 {
                p = p + self.clone();
            }
            mask >>= 1;
        }
        p
    }
}
