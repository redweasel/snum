//! Custom rational type, which allows much more generic types based on [Euclid] and handles division by zero like floats without panic.

use core::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::*,
};
use take_mut::take;

use crate::{
    DevelopContinuedFraction, FloatType, IntoContinuedFraction, IntoDiscrete, float::ApproxFloat,
    num::*,
};
use crate::{FromU64, complex::Complex};

/// A fraction, or rational number `p/q`.
/// For anything useful, it requires the [Zero], [One] and [Euclid] traits
/// and some arithmetic operations, depending on the usecase.
///
/// Dividing by zero is handled without panics with Inf, -Inf and NaN just like for floats.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Ratio<T> {
    pub numer: T,
    pub denom: T,
}

impl<T> Ratio<T> {
    pub const fn new_raw(numer: T, denom: T) -> Self {
        Ratio { numer, denom }
    }
    pub fn recip(self) -> Self {
        Self {
            numer: self.denom,
            denom: self.numer,
        }
    }
}
impl<T: Zero> Ratio<T> {
    pub fn is_finite(&self) -> bool {
        !self.denom.is_zero()
    }
    pub fn is_nan(&self) -> bool {
        self.denom.is_zero() && self.numer.is_zero()
    }
}
impl<T: Zero + One> Ratio<T>
where
    for<'a> &'a T: Div<&'a T, Output = T>,
{
    /// rounds to the next integer towards zero by dividing the numerator
    /// by the denominator and setting the new denominator to one.
    #[inline]
    pub fn trunc(self) -> Self {
        if self.is_finite() {
            Self::from(&self.numer / &self.denom)
        } else {
            self
        }
    }
}
impl<T: Zero + One> Ratio<T>
where
    for<'a> &'a T: Rem<&'a T, Output = T>,
{
    /// Returns the fractional part of a number, with division rounded towards zero. (based on [Rem])
    ///
    /// Satisfies `self == self.trunc() + self.fract()`.
    #[inline]
    pub fn fract(self) -> Ratio<T> {
        if self.is_finite() {
            Ratio::new_raw(&self.numer % &self.denom, self.denom)
        } else {
            Ratio::new_raw(T::zero(), T::one())
        }
    }
}
impl<T: Cancel + PartialOrd> IntoDiscrete for Ratio<T> {
    type Output = T;
    /// rounds to the next integer towards -oo, using [Euclid].
    ///
    /// Panics if the rational is not finite.
    fn floor(&self) -> T {
        if !self.is_finite() {
            panic!("Called floor on non finite rational");
        }
        // TODO there is no guarantee, that this is the correct floor...
        // e.g. for Complex<i32> this will NOT do the floor in both components.
        if self.denom >= T::zero() {
            self.numer.div_rem_euclid(&self.denom)
        } else {
            (T::zero() - self.numer.clone()).div_rem_euclid(&(T::zero() - self.denom.clone()))
        }
        .0
    }
    /// rounds to the closest integer, breaking ties by rounding away from zero.
    ///
    /// Panics if the rational is not finite.
    fn round(&self) -> T {
        if !self.is_finite() {
            panic!("Called round on non finite rational");
        }
        let mut t1 = self.floor();
        let mut t2 = t1.clone() + T::one();
        if self.denom < T::zero() {
            (t1, t2) = (t2, t1);
        }
        // TODO avoid this multiplication using continued fractions
        let d1 = self.numer.clone() - t1.clone() * self.denom.clone();
        let d2 = t2.clone() * self.denom.clone() - self.numer.clone();
        if t1 >= T::zero() {
            // round up
            if d1 < d2 { t1 } else { t2 }
        } else {
            // round down
            if d1 <= d2 { t1 } else { t2 }
        }
    }
}
impl<T: Zero + Euclid + PartialEq> Ratio<T> {
    /// Returns true if the rational number can be written as a normal number (i.e. `self == self.trunc()`).
    #[inline]
    pub fn is_integral(&self) -> bool {
        self.is_finite() && self.numer.div_rem_euclid(&self.denom).1.is_zero()
    }
}
impl<T: Cancel> Ratio<T> {
    pub fn new(numer: T, denom: T) -> Self {
        let (mut numer, mut denom) = numer.cancel(denom);
        if !numer.is_zero() && !denom.is_valid_euclid() {
            // also != 0, as 0 is always valid
            numer = T::zero() - numer;
            denom = T::zero() - denom;
        }
        Ratio { numer, denom }
    }

    pub fn reduced(self) -> Self {
        Self::new(self.numer, self.denom)
    }
}

impl<T: Num + Cancel + Div<Output = T>> Ratio<T> {
    /// reduce, not only by canceling, but also by dividing by the denominator, if it has a representable inverse.
    pub fn reduced_full(mut self) -> Self {
        (self.numer, self.denom) = self.numer.cancel(self.denom);
        if self.denom.is_unit() {
            self.numer = self.numer / self.denom;
            self.denom = One::one();
        }
        self
    }
}

impl<T: Zero + One> Default for Ratio<T> {
    fn default() -> Self {
        Self {
            numer: Zero::zero(),
            denom: One::one(),
        }
    }
}

impl<T: Cancel> Zero for Ratio<T>
where
    for<'a> &'a T: AddMul<Output = T>,
{
    fn zero() -> Self {
        Self {
            numer: T::zero(),
            denom: T::one(),
        }
    }
    fn is_zero(&self) -> bool {
        self.numer.is_zero()
    }
}

impl<T: PartialEq + Cancel> One for Ratio<T>
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
{
    fn one() -> Self {
        Self {
            numer: T::one(),
            denom: T::one(),
        }
    }
    fn is_one(&self) -> bool {
        self.numer == self.denom
    }
}

impl<T: Clone + Zero + PartialEq + Sub<Output = T> + Euclid> PartialEq for Ratio<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.denom == other.denom {
            return if self.numer == other.numer {
                true
            } else if self.denom.is_zero() {
                // classify the 3 states +oo, 0, -oo for both
                let s = self.numer.is_valid_euclid() as u8 + self.numer.is_zero() as u8;
                let o = other.numer.is_valid_euclid() as u8 + other.numer.is_zero() as u8;
                s == o
            } else {
                false
            };
        }
        if self.denom.is_zero() || other.denom.is_zero() {
            return false;
        }
        if self.numer == other.numer {
            // +0 and -0 are considered equal
            return self.numer.is_zero() || self.denom == other.denom;
        }
        // Compare by comparing the continued fraction expansion.
        let mut s = self.clone();
        let mut o = other.clone();
        loop {
            // Note, that div_rem_euclid doesn't have the same behavior as div_mod_floor,
            // when negative denominators are used, so the denominators have to be checked.
            let a = if s.denom.is_valid_euclid() {
                s
            } else {
                Ratio::new_raw(T::zero() - s.numer, T::zero() - s.denom)
            };
            let b = if o.denom.is_valid_euclid() {
                o
            } else {
                Ratio::new_raw(T::zero() - o.numer, T::zero() - o.denom)
            };
            // after flipping signs, the denominators might have become equal.
            if a.denom == b.denom {
                return a.numer == b.numer;
            }
            let (a_int, a_rem) = a.numer.div_rem_euclid(&a.denom);
            let (b_int, b_rem) = b.numer.div_rem_euclid(&b.denom);
            if a_int == b_int {
                let a_zero = a_rem.is_zero();
                let b_zero = b_rem.is_zero();
                if a_zero || b_zero {
                    return a_zero && b_zero;
                }
                // Compare the reciprocals of the remaining fractions in reverse
                // Note, the denominators are smaller, so this can't lead to infinite recursion.
                s = Ratio::new_raw(a.denom, a_rem);
                o = Ratio::new_raw(b.denom, b_rem);
            } else {
                return false;
            }
        }
    }
}
impl<T: Clone + Zero + Eq + Sub<T, Output = T> + Euclid> Eq for Ratio<T> {}

impl<T: Cancel + PartialOrd> PartialOrd for Ratio<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let zero = T::zero();
        if !self.is_finite() || !other.is_finite() {
            if self.is_nan() || other.is_nan() {
                return None;
            }
            let mut s = self.numer.partial_cmp(&zero)?;
            let mut o = other.numer.partial_cmp(&zero)?;
            if self.denom < zero {
                s = s.reverse();
            }
            if other.denom < zero {
                o = o.reverse();
            }
            // compare infinities
            return Some(s.cmp(&o));
        }
        let mut s = self.clone();
        let mut o = other.clone();
        // Compare by comparing the continued fraction expansion.
        // based on `num_rational`, but fixed critical stackoverflow.
        loop {
            if s.denom == o.denom {
                // With equal denominators, the numerators can be compared
                let ord = s.numer.partial_cmp(&o.numer);
                return if s.denom < zero {
                    ord.map(Ordering::reverse)
                } else {
                    ord
                };
            } else if s.numer == o.numer {
                // With equal numerators, the denominators can be inversely compared
                if s.numer.is_zero() {
                    // -0 and 0 are equal
                    return Some(Ordering::Equal);
                }
                let ord = s.denom.partial_cmp(&o.denom);
                return if s.numer < zero {
                    ord
                } else {
                    ord.map(Ordering::reverse)
                };
            } else {
                // Compare as floored integers and remainders
                // Note, that div_rem_euclid doesn't have the same behavior as div_mod_floor,
                // when negative denominators are used, so the denominators have to be checked.
                let a = if s.denom >= zero {
                    s
                } else {
                    Ratio::new_raw(T::zero() - s.numer, T::zero() - s.denom)
                };
                let b = if o.denom >= zero {
                    o
                } else {
                    Ratio::new_raw(T::zero() - o.numer, T::zero() - o.denom)
                };
                // after flipping signs, the denominators might have become equal.
                if a.denom == b.denom {
                    return a.numer.partial_cmp(&b.numer);
                }
                let (a_int, a_rem) = a.numer.div_rem_euclid(&a.denom);
                let (b_int, b_rem) = b.numer.div_rem_euclid(&b.denom);
                return match a_int.partial_cmp(&b_int)? {
                    Ordering::Greater => Some(Ordering::Greater),
                    Ordering::Less => Some(Ordering::Less),
                    Ordering::Equal => {
                        match (a_rem.is_zero(), b_rem.is_zero()) {
                            (true, true) => Some(Ordering::Equal),
                            (true, false) => Some(Ordering::Less),
                            (false, true) => Some(Ordering::Greater),
                            (false, false) => {
                                // Compare the reciprocals of the remaining fractions in reverse
                                // Note, the denominators are smaller, so this can't lead to infinite recursion if the gcd doesn't.
                                o = Ratio::new_raw(a.denom, a_rem);
                                s = Ratio::new_raw(b.denom, b_rem);
                                continue;
                            }
                        }
                    }
                };
            }
        }
    }
}
impl<T: Cancel + Ord> Ord for Ratio<T>
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        // TODO should this implement Ord? Where is it useful? Keep in mind, that integers also panic on divide by zero.
        self.partial_cmp(other).unwrap() // panic for NaN < NaN!
    }
}

impl<T: One> From<T> for Ratio<T> {
    fn from(value: T) -> Self {
        Ratio {
            numer: value,
            denom: T::one(),
        }
    }
}
impl<T: Zero> From<Ratio<T>> for Ratio<Complex<T>> {
    fn from(value: Ratio<T>) -> Self {
        Ratio {
            numer: value.numer.into(),
            denom: value.denom.into(),
        }
    }
}
impl<T> From<Ratio<T>> for (T, T) {
    fn from(value: Ratio<T>) -> Self {
        (value.numer, value.denom)
    }
}
impl<T> From<(T, T)> for Ratio<T> {
    fn from(value: (T, T)) -> Self {
        Ratio::new_raw(value.0, value.1)
    }
}

impl<T: FromU64 + One> FromU64 for Ratio<T> {
    #[inline(always)]
    fn from_u64(value: u64) -> Self {
        Ratio::from(T::from_u64(value))
    }
}

impl<T: Conjugate> Conjugate for Ratio<T> {
    fn conj(&self) -> Self {
        Self {
            numer: self.numer.conj(),
            denom: self.denom.conj(),
        }
    }
}

impl<T: Zero + Neg<Output = T>> Neg for Ratio<T> {
    type Output = Ratio<T>;
    fn neg(self) -> Self::Output {
        if !self.numer.is_zero() {
            Self {
                numer: -self.numer,
                denom: self.denom,
            }
        } else {
            // negative zero
            Self {
                numer: self.numer,
                denom: -self.denom,
            }
        }
    }
}

macro_rules! impl_add {
    ($Add:ident, $add:ident) => {
        impl<T: Cancel> $Add for Ratio<T> {
            type Output = Ratio<T>;
            fn $add(self, rhs: Self) -> Self::Output {
                if self.denom.is_zero() && rhs.denom.is_zero() {
                    if self.numer.is_zero() || rhs.numer.is_zero() {
                        return Ratio::new_raw(T::zero(), T::zero());
                    }
                    let a = self.reduced();
                    let b = rhs.reduced();
                    return Ratio::new_raw(a.numer.$add(b.numer), T::zero());
                }
                // avoid overflows by computing the gcd early (in each operation)
                let (a, b) = self.denom.clone().cancel(rhs.denom.clone());
                Ratio {
                    numer: (self.numer * b).$add(rhs.numer * a.clone()),
                    denom: a * rhs.denom,
                }
            }
        }
        impl<'a, T: Cancel + $Add<T, Output = T>> $Add for &'a Ratio<T> {
            type Output = Ratio<T>;
            fn $add(self, rhs: Self) -> Self::Output {
                if self.denom.is_zero() && rhs.denom.is_zero() {
                    if self.numer.is_zero() || rhs.numer.is_zero() {
                        return Ratio::new_raw(T::zero(), T::zero());
                    }
                    let a = self.clone().reduced();
                    let b = rhs.clone().reduced();
                    return Ratio::new_raw(a.numer.$add(b.numer), T::zero());
                }
                // avoid overflows by computing the gcd early (in each operation)
                let (a, b) = self.denom.clone().cancel(rhs.denom.clone());
                Ratio {
                    numer: (self.numer.clone() * b).$add(rhs.numer.clone() * a.clone()),
                    denom: a * rhs.denom.clone(),
                }
            }
        }
        impl<T: Cancel> $Add<T> for Ratio<T> {
            type Output = Ratio<T>;
            fn $add(self, rhs: T) -> Self::Output {
                // avoid overflows by computing the gcd in each operation
                let numer = self.numer.$add(self.denom.clone() * rhs);
                Ratio::new(numer, self.denom)
            }
        }
        impl<'a, T: Cancel + $Add<T, Output = T>> $Add<&'a T> for &'a Ratio<T> {
            type Output = Ratio<T>;
            fn $add(self, rhs: &'a T) -> Self::Output {
                // avoid overflows by computing the gcd in each operation
                let numer = (self.numer.clone()).$add(self.denom.clone() * rhs.clone());
                Ratio::new(numer, self.denom.clone())
            }
        }
    };
}
impl_add!(Add, add);
impl_add!(Sub, sub);

impl<T: Cancel> Mul for Ratio<T>
where
    for<'a> &'a T: Mul<Output = T>,
{
    type Output = Ratio<T>;
    fn mul(mut self, mut rhs: Self) -> Self::Output {
        // avoid overflows by computing the gcd early (in each operation)
        (self.denom, rhs.numer) = self.denom.cancel(rhs.numer);
        (self.numer, rhs.denom) = self.numer.cancel(rhs.denom);
        // make sure no signs get lost by multiplying with zero
        let mut modified = false;
        if self.numer.is_zero() && !rhs.numer.is_zero() {
            rhs.denom = &rhs.denom * &rhs.numer;
            modified = true;
        } else if rhs.numer.is_zero() && !self.numer.is_zero() {
            self.denom = &self.denom * &self.numer;
            modified = true;
        }
        if self.denom.is_zero() && !rhs.denom.is_zero() {
            rhs.numer = &rhs.numer * &rhs.denom;
            modified = true;
        } else if rhs.denom.is_zero() && !self.denom.is_zero() {
            self.numer = &self.numer * &self.denom;
            modified = true;
        }
        // now compute the result. No more cancelation needed, as the initial fractions are assumed to be cancelled.
        let mut r = Ratio {
            numer: &self.numer * &rhs.numer,
            denom: &self.denom * &rhs.denom,
        };
        if modified {
            r = r.reduced();
        }
        r
    }
}
impl<'a, T: Cancel> Mul for &'a Ratio<T> {
    type Output = Ratio<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        // avoid overflows by computing the gcd early (in each operation)
        let (mut sd, mut rn) = self.denom.clone().cancel(rhs.numer.clone());
        let (mut sn, mut rd) = self.numer.clone().cancel(rhs.denom.clone());
        // make sure no signs get lost by multiplying with zero
        let mut modified = false;
        if sn.is_zero() && !rn.is_zero() {
            rd = rd * rn.clone();
            modified = true;
        } else if rn.is_zero() && !sn.is_zero() {
            sd = sd * sn.clone();
            modified = true;
        }
        if sd.is_zero() && !rd.is_zero() {
            rn = rn * rd.clone();
            modified = true;
        } else if rd.is_zero() && !sd.is_zero() {
            sn = sn * sd.clone();
            modified = true;
        }
        // now compute the result. No more cancelation needed, as the initial fractions are assumed to be cancelled.
        let mut r = Ratio {
            numer: sn * rn,
            denom: sd * rd,
        };
        if modified {
            r = r.reduced();
        }
        r
    }
}
impl<T: Cancel> Mul<T> for Ratio<T>
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
{
    type Output = Ratio<T>;
    fn mul(self, rhs: T) -> Self::Output {
        // avoid overflows by computing the gcd early (in each operation)
        let (mut rhs, mut denom) = rhs.cancel(self.denom.clone());
        if self.numer.is_zero() && !rhs.is_valid_euclid() {
            rhs = T::zero() - rhs; // floats can have zero signs!
            denom = T::zero() - denom;
        }
        Ratio {
            numer: &self.numer * &rhs,
            denom,
        }
    }
}
impl<'a, T: Cancel> Mul<&'a T> for &'a Ratio<T> {
    type Output = Ratio<T>;
    fn mul(self, rhs: &'a T) -> Self::Output {
        // avoid overflows by computing the gcd early (in each operation)
        let (mut rhs, mut denom) = rhs.clone().cancel(self.denom.clone());
        if self.numer.is_zero() && !rhs.is_valid_euclid() {
            rhs = T::zero() - rhs; // floats can have zero signs!
            denom = T::zero() - denom;
        }
        Ratio {
            numer: self.numer.clone() * rhs,
            denom,
        }
    }
}

impl<T: Cancel> Div for Ratio<T>
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
{
    type Output = Ratio<T>;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.recip()
    }
}
impl<'a, T: Cancel> Div for &'a Ratio<T> {
    type Output = Ratio<T>;
    fn div(self, rhs: Self) -> Self::Output {
        self * &rhs.clone().recip()
    }
}
impl<T: Cancel> Div<T> for Ratio<T> {
    type Output = Ratio<T>;
    fn div(self, rhs: T) -> Self::Output {
        // avoid overflows by computing the gcd early (in each operation)
        let (mut numer, rhs) = self.numer.cancel(rhs);
        if self.denom.is_zero() && !rhs.is_valid_euclid() {
            numer = T::zero() - numer;
        }
        Ratio {
            numer,
            denom: self.denom * rhs,
        }
    }
}
impl<'a, T: Cancel> Div<&'a T> for &'a Ratio<T> {
    type Output = Ratio<T>;
    fn div(self, rhs: &'a T) -> Self::Output {
        // avoid overflows by computing the gcd early (in each operation)
        let (mut numer, rhs) = self.numer.clone().cancel(rhs.clone());
        if self.denom.is_zero() && !rhs.is_valid_euclid() {
            numer = T::zero() - numer;
        }
        Ratio {
            numer,
            denom: self.denom.clone() * rhs,
        }
    }
}

impl<T: Cancel> Rem for Ratio<T>
where
    for<'a> &'a T: Div<Output = T>,
{
    type Output = Ratio<T>;
    /// remainder like for floats, not the division remainder, but rather a signed modulo function.
    fn rem(self, rhs: Self) -> Self::Output {
        let f = &self / &rhs;
        &self - &(&rhs * &(&f.numer / &f.denom))
    }
}
impl<'a, T: Cancel + Div<Output = T>> Rem for &'a Ratio<T> {
    type Output = Ratio<T>;
    /// remainder like for floats, not the division remainder, but rather a signed modulo function.
    fn rem(self, rhs: Self) -> Self::Output {
        let f = self / rhs;
        self - &(rhs * &(f.numer / f.denom))
    }
}
impl<T: Cancel> Rem<T> for Ratio<T>
where
    for<'a> &'a T: Div<Output = T>,
{
    type Output = Ratio<T>;
    /// remainder like for floats, not the division remainder, but rather a signed modulo function.
    fn rem(self, rhs: T) -> Self::Output {
        let f = &self / &rhs;
        &self - &(rhs * (&f.numer / &f.denom))
    }
}
impl<'a, T: Cancel + Div<Output = T>> Rem<&'a T> for &'a Ratio<T> {
    type Output = Ratio<T>;
    /// remainder like for floats, not the division remainder, but rather a signed modulo function.
    fn rem(self, rhs: &'a T) -> Self::Output {
        let f = self / rhs;
        self - &(rhs.clone() * (f.numer / f.denom))
    }
}

impl<T: Cancel> Euclid for Ratio<T> {
    fn div_rem_euclid(&self, div: &Self) -> (Self, Self) {
        // always exactly divisible.
        let f = self / div;
        (f, T::zero().into())
        // alternative: rational division as integers
        // -> bring them to the same denominator, then do the remainder
        // problem: overflows easily.
    }
    fn is_valid_euclid(&self) -> bool {
        // only the finite 0 is a possible output of `div_rem_euclid`, however either x or -x need to be valid,
        // so use rem_euclid on the numerator and call some more numbers valid euclid.
        self.numer.is_valid_euclid() && !self.denom.is_zero()
    }
}

macro_rules! forward_assign_impl {
    ($($AddAssign:ident, $Add:ident, $add_assign:ident, $add:ident),+) => {
        $(impl<T: Cancel> $AddAssign for Ratio<T>
            where for<'a> &'a T: AddMul<Output = T> + $Add<Output = T> {
            fn $add_assign(&mut self, rhs: Ratio<T>) {
                take(self, |x| x.$add(rhs));
            }
        }
        impl<T: Cancel> $AddAssign<T> for Ratio<T>
            where for<'a> &'a T: AddMul<Output = T> + $Add<Output = T> {
            fn $add_assign(&mut self, rhs: T) {
                take(self, |x| x.$add(rhs));
            }
        }
        impl<'a, T: Cancel + $Add<Output = T>> $AddAssign<&'a Ratio<T>> for Ratio<T> {
            fn $add_assign(&mut self, rhs: &'a Ratio<T>) {
                take(self, |x| (&x).$add(rhs));
            }
        }
        impl<'a, T: Cancel + $Add<Output = T>> $AddAssign<&'a T> for Ratio<T> {
            fn $add_assign(&mut self, rhs: &'a T) {
                take(self, |x| (&x).$add(rhs));
            }
        })+
    };
}
forward_assign_impl!(
    AddAssign, Add, add_assign, add, SubAssign, Sub, sub_assign, sub, MulAssign, Mul, mul_assign,
    mul, DivAssign, Mul, div_assign, div, RemAssign, Div, rem_assign, rem
);

impl<T: Cancel> Sum for Ratio<T>
where
    for<'a> &'a T: AddMul<Output = T>,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::zero(), |acc, c| acc + c)
    }
}
impl<'a, T: Cancel> Sum<&'a Ratio<T>> for Ratio<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Ratio<T>>,
    {
        iter.fold(Self::new_raw(T::zero(), T::one()), |acc, c| &acc + c)
    }
}

impl<T: Cancel> Product for Ratio<T>
where
    for<'a> &'a T: Mul<Output = T>,
{
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, c| acc * c)
    }
}
impl<'a, T: Cancel> Product<&'a Ratio<T>> for Ratio<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Ratio<T>>,
    {
        iter.fold(Self::new_raw(T::one(), T::one()), |acc, c| &acc * c)
    }
}

impl<T: Num + Cancel> Num for Ratio<T>
where
    T::Real: Num + Cancel,
    Ratio<T>: From<Ratio<T::Real>>,
{
    type Real = Ratio<T::Real>;
    #[inline(always)]
    fn abs_sqr(&self) -> Self::Real {
        Ratio {
            numer: self.numer.abs_sqr(),
            denom: self.denom.abs_sqr(),
        }
    }
    #[inline(always)]
    fn re(&self) -> Self::Real {
        let d = self.denom.conj();
        if d != self.denom {
            // do the trick of multiplying with denom.conj()
            // the result is not cancelled! This is BAD for real numbers, hence the check above.
            // (all operations also work without canceling and canceling doesn't extend the range of this operation)
            Ratio {
                numer: (self.numer.clone() * d).re(),
                denom: self.denom.abs_sqr(),
            }
        } else {
            Ratio {
                numer: self.numer.re(),
                denom: self.denom.re(),
            }
        }
    }
    #[inline(always)]
    fn is_unit(&self) -> bool {
        // +oo and -oo have inverse 0, but 0 has no unique inverse, since there is two infinities,
        // so exclude all 3 values.
        !self.numer.is_zero() && !self.denom.is_zero()
    }
}

impl<F: FloatType, T: Cancel + Neg<Output = T>> ApproxFloat<F> for Ratio<T>
where
    T: ApproxFloat<F>,
{
    fn from_approx(value: F, tol: F) -> Option<Self> {
        if !value.is_finite() {
            if value.is_infinite() {
                return Some(Ratio::new_raw(
                    if value > F::zero() {
                        T::one()
                    } else {
                        -T::one()
                    },
                    T::zero(),
                ));
            }
            return Some(Ratio::new_raw(T::zero(), T::zero()));
        }
        if value.round() == value {
            let v = T::from_approx(value.clone(), tol)?;
            return Some(Ratio::new_raw(v, T::one()));
        }
        // zero is already covered above. These non finite values don't come from division by zero.
        if !(F::one() / value.clone()).is_finite() {
            // same as in to_approx, handle subnormal inputs by multiplying.
            // all of this needs to happen without integer overflows.
            let mut _16 = T::one() + T::one();
            _16 = _16.clone() * _16;
            _16 = _16.clone() * _16;
            let _16f: F = _16.to_approx();
            // Note, this can lead to stack overflow, but not to an endless loop.
            return Some(&Self::from_approx(value * _16f.clone(), tol * _16f.clone())? / &_16);
        }
        let iter = DevelopContinuedFraction::new(value.clone()).continued_fraction(F::one());
        for x in iter {
            let err = x.numer.clone() / x.denom.clone() - value.clone();
            if !x.numer.is_finite() || !x.denom.is_finite() || !err.is_finite() {
                return None;
            }
            if err <= tol && -err <= tol {
                let numer = T::from_approx(x.numer.clone(), F::one())?;
                let denom = T::from_approx(x.denom.clone(), F::one())?;
                return Some(Ratio::new(numer, denom));
            }
        }
        None
    }
    #[inline]
    fn to_approx(&self) -> F {
        // not always safe to just compute it directly.
        // The unsafe case is, where the denominator is so big,
        // that an infinite result is returned, but a
        // subnormal float result would be possible.
        // To fix that, divided by 16 until the result is finite.
        let mut n = self.numer.to_approx();
        let mut denom = self.denom.clone();
        let mut d = denom.to_approx();
        if !n.is_zero() && !d.is_finite() {
            let mut _16 = T::one() + T::one();
            _16 = _16.clone() * _16;
            _16 = _16.clone() * _16;
            let _16f: F = _16.to_approx();
            loop {
                denom = denom.div_rem_euclid(&_16).0;
                n = n / _16f.clone();
                d = denom.to_approx();
                if n.is_zero() || d.is_finite() {
                    break;
                }
            }
        }
        n / d
    }
}

// Safety: `Complex<T>` is `repr(C)` and contains only instances of `T`, so we
// can guarantee it contains no *added* padding. Thus, if `T: Zeroable`,
// `Ratio<T>` is also `Zeroable`
#[cfg(feature = "bytemuck")]
unsafe impl<T: bytemuck::Zeroable> bytemuck::Zeroable for Ratio<T> {}

// Safety: `Complex<T>` is `repr(C)` and contains only instances of `T`, so we
// can guarantee it contains no *added* padding. Thus, if `T: Pod`,
// `Ratio<T>` is also `Pod`
#[cfg(feature = "bytemuck")]
unsafe impl<T: bytemuck::Pod> bytemuck::Pod for Ratio<T> {}

// NB: We can't just `#[derive(Hash)]`, because it needs to agree
// with `Eq` even for non-reduced ratios.
impl<T: Clone + Zero + Euclid + Hash> Hash for Ratio<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut r = self.clone();
        loop {
            if !r.denom.is_zero() {
                let (int, rem) = r.numer.div_rem_euclid(&r.denom);
                (r.denom, r.numer) = (r.numer, rem);
                int.hash(state);
            } else {
                return r.denom.hash(state);
            }
        }
    }
}

#[cfg(feature = "std")]
fn fmt_ratio(
    f: &mut fmt::Formatter<'_>,
    numer_zero: bool,
    denom_zero: bool,
    numer_args: fmt::Arguments<'_>,
    denom_args: fmt::Arguments<'_>,
    prefix: &str,
) -> fmt::Result {
    let numer = &std::fmt::format(numer_args);
    let numer_no_sign = numer.strip_prefix("-");
    let numer_is_number = !numer_no_sign
        .unwrap_or(numer)
        .chars()
        .any(|c| c == '+' || c == '-' || c == '/');
    let pre_pad = if denom_zero {
        if numer_zero {
            "NaN"
        } else if numer_no_sign.unwrap_or(numer) == "1" {
            &std::format!("{}∞", &numer[..numer.len() - 1])
        } else if numer_is_number {
            &std::format!("{}∞", numer)
        } else {
            &std::format!("({})∞", numer)
        }
    } else {
        // note, this is missing a lot of annotations like the precision in case of floats.
        let denom = &std::fmt::format(denom_args);
        let denom_no_sign = denom.strip_prefix("-");
        let denom_is_number = !denom_no_sign
            .unwrap_or(denom)
            .chars()
            .any(|c| c == '+' || c == '-' || c == '/' || c == '*');
        // Note, the signs can not be processed as strings, as the expression might be e.g. -1+i
        &if numer_is_number {
            if denom_is_number {
                std::format!("{}/{}", numer, denom)
            } else {
                std::format!("{}/({})", numer, denom)
            }
        } else {
            if denom_is_number {
                std::format!("({})/{}", numer, denom)
            } else {
                std::format!("({})/({})", numer, denom)
            }
        }
    };
    // TODO this padding can add zeros before parenthesis...
    if let Some(pre_pad) = pre_pad.strip_prefix("-") {
        if let Some(pre_pad) = pre_pad.strip_prefix(prefix) {
            f.pad_integral(false, prefix, pre_pad)
        } else {
            f.pad_integral(false, "", pre_pad)
        }
    } else {
        if let Some(pre_pad) = pre_pad.strip_prefix(prefix) {
            f.pad_integral(true, prefix, pre_pad)
        } else {
            f.pad_integral(true, "", &pre_pad)
        }
    }
}
// TODO extract the no_std case as well and use the same function in the macro part.

// String conversions
macro_rules! impl_formatting {
    ($Display:ident, $prefix:expr, $fmt_str:expr, $fmt_alt:expr) => {
        impl<T: fmt::$Display + Clone + Zero + One> fmt::$Display for Ratio<T> {
            #[cfg(feature = "std")]
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.denom.is_one() {
                    return self.numer.fmt(f);
                }
                if f.alternate() {
                    fmt_ratio(
                        f,
                        self.numer.is_zero(),
                        self.denom.is_zero(),
                        format_args!($fmt_alt, self.numer),
                        format_args!($fmt_alt, self.denom),
                        $prefix,
                    )
                } else {
                    fmt_ratio(
                        f,
                        self.numer.is_zero(),
                        self.denom.is_zero(),
                        format_args!($fmt_str, self.numer),
                        format_args!($fmt_str, self.denom),
                        $prefix,
                    )
                }
            }
            #[cfg(not(feature = "std"))]
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.denom.is_one() {
                    return self.numer.fmt(f);
                }
                // can't do any of the checking, so just always print with parenthesis to be on the safe side
                if self.denom.is_zero() {
                    if self.numer.is_zero() {
                        write!(f, "NaN")
                    } else {
                        if f.alternate() {
                            write!(f, concat!("(", $fmt_alt, ")∞"), self.numer)
                        } else {
                            write!(f, concat!("(", $fmt_str, ")∞"), self.numer)
                        }
                    }
                } else if f.alternate() {
                    write!(
                        f,
                        concat!("(", $fmt_alt, ")/(", $fmt_alt, ")"),
                        self.numer, self.denom
                    )
                } else {
                    write!(
                        f,
                        concat!("(", $fmt_str, ")/(", $fmt_str, ")"),
                        self.numer, self.denom
                    )
                }
            }
        }
    };
}

impl_formatting!(Display, "", "{}", "{:#}");
impl_formatting!(Octal, "0o", "{:o}", "{:#o}");
impl_formatting!(Binary, "0b", "{:b}", "{:#b}");
impl_formatting!(LowerHex, "0x", "{:x}", "{:#x}");
impl_formatting!(UpperHex, "0x", "{:X}", "{:#X}");
impl_formatting!(LowerExp, "", "{:e}", "{:#e}");
impl_formatting!(UpperExp, "", "{:E}", "{:#E}");

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for Ratio<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (&self.numer, &self.denom).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for Ratio<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (numer, denom) = serde::Deserialize::deserialize(deserializer)?;
        Ok(Self::new_raw(numer, denom))
    }
}
