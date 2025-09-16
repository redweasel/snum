//! Custom rational type, which allows much more generic types based on [Euclid] and handles division by zero like floats without panic.

use core::{
    cmp::Ordering,
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::*,
};
use take_mut::take;

use crate::{Complex, FromU64};
use crate::{
    DevelopContinuedFraction, FloatType, IntoContinuedFraction, IntoDiscrete, float::ApproxFloat,
    num::*,
};

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
    /// create a rational number without cancelling.
    #[must_use]
    pub const fn new_raw(numer: T, denom: T) -> Self {
        Ratio { numer, denom }
    }
    #[must_use]
    pub fn recip(self) -> Self {
        Self {
            numer: self.denom,
            denom: self.numer,
        }
    }
}
impl<T: One> Ratio<T> {
    /// create a rational number from the numerator. The denominator is set to one.
    /// Use `.into()` if the intention was to use the `From` trait. This function fixes
    /// the type inference when used as `Ratio::from(x)`.
    #[must_use]
    pub fn from(numer: T) -> Self {
        Ratio { numer, denom: T::one() }
    }
}
impl<T: Zero + PartialEq> Ratio<T> {
    #[must_use]
    pub fn is_finite(&self) -> bool {
        !self.denom.is_zero() && self.numer == self.numer && self.denom == self.denom
    }
    #[must_use]
    pub fn is_nan(&self) -> bool {
        (self.denom.is_zero() && self.numer.is_zero())
            || self.numer != self.numer
            || self.denom != self.denom
    }
}
impl<T: Zero + One + PartialEq> Ratio<T>
where
    for<'a> &'a T: Div<&'a T, Output = T>,
{
    /// rounds to the next integer towards zero by dividing the numerator
    /// by the denominator and setting the new denominator to one.
    #[must_use]
    pub fn trunc(self) -> Self {
        if self.is_finite() {
            Self::from(&self.numer / &self.denom)
        } else {
            self
        }
    }
}
impl<T: Zero + One + PartialEq> Ratio<T>
where
    for<'a> &'a T: Rem<&'a T, Output = T>,
{
    /// Returns the fractional part of a number, with division rounded towards zero. (based on [Rem])
    ///
    /// Satisfies `self == self.trunc() + self.fract()`.
    #[must_use]
    pub fn fract(self) -> Ratio<T> {
        if self.is_finite() {
            Ratio::new_raw(&self.numer % &self.denom, self.denom)
        } else {
            Ratio::new_raw(T::zero(), T::one())
        }
    }
}
impl<T: Cancel + IntoDiscrete + PartialOrd> IntoDiscrete for Ratio<T>
where <T as IntoDiscrete>::Output: Add<Output = <T as IntoDiscrete>::Output> + Div<Output = <T as IntoDiscrete>::Output> {
    type Output = T;
    /// rounds to an integer by rounding towards -∞
    ///
    /// Panics if the rational is non finite.
    fn div_floor(&self, div: &Self) -> T {
        if !self.is_finite() {
            if self.numer != self.numer {
                return self.numer.clone(); // NaN
            }
            if self.denom != self.denom {
                return self.denom.clone(); // NaN
            }
            panic!("called div_floor on non finite rational");
        }
        if !div.is_finite() {
            if div.numer != div.numer {
                return div.numer.clone(); // NaN
            }
            if div.denom != div.denom {
                return div.denom.clone(); // NaN
            }
            panic!("called div_floor on non finite rational");
        }
        let x = self / div;
        x.numer.div_floor(&x.denom).into()
    }
    /// rounds to the closest integer, breaking ties by rounding away from zero.
    ///
    /// Panics if the rational is not finite.
    fn round(&self) -> T {
        assert!(self.is_finite(), "called round on non finite rational");
        let mut t1 = self.floor();
        let mut t2 = t1.clone() + T::one();
        let mut denom_abs = self.denom.clone();
        if self.denom < T::zero() {
            (t1, t2) = (t2, t1);
            denom_abs = T::zero() - denom_abs;
        }
        let (d1, d2) = if self.numer < T::zero() {
            let d2 = self.denom.clone() * t2.clone() - self.numer.clone();
            (denom_abs - d2.clone(), d2)
        } else {
            let d1 = self.numer.clone() - self.denom.clone() * t1.clone();
            (d1.clone(), denom_abs - d1)
        };
        let ord = d1.partial_cmp(&d2).unwrap();
        if t1 >= T::zero() {
            // round up
            if ord == Ordering::Less { t1 } else { t2 }
        } else {
            // round down
            if ord == Ordering::Greater { t2 } else { t1 }
        }
    }
}
impl<T: Zero + Euclid + PartialEq> Ratio<T> {
    /// Returns true if the rational number can be written as a normal number (i.e. `self == self.trunc()`).
    #[must_use]
    pub fn is_integral(&self) -> bool {
        self.is_finite() && self.numer.div_rem_euclid(&self.denom).1.is_zero()
    }
}
impl<T: Cancel> Ratio<T> {
    /// Create a rational number with cancelling.
    /// To create a `Ratio` in a const function, use `new_raw`.
    #[must_use]
    pub fn new(numer: T, denom: T) -> Self {
        let (mut numer, mut denom) = numer.cancel(denom);
        if !numer.is_zero() && !denom.is_valid_euclid() {
            // also != 0, as 0 is always valid
            numer = T::zero() - numer;
            denom = T::zero() - denom;
        }
        Ratio { numer, denom }
    }
    #[must_use]
    pub fn reduced(self) -> Self {
        Self::new(self.numer, self.denom)
    }
}

impl<T: SafeDiv> Ratio<T> {
    /// reduce, not only by canceling, but also by dividing by the denominator, if it has a representable inverse.
    #[must_use]
    pub fn reduced_full(mut self) -> Self {
        // not using safe_div directly though, as this should also cancel x/0 to ±1/0 with correct sign.
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
        self.numer.is_zero() && !self.denom.is_zero()
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
        // detect nan to avoid endless loops
        if self.numer != self.numer
            || self.denom != self.denom
            || other.numer != other.numer
            || other.denom != other.denom
        {
            return false;
        }
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
        // based on `num_rational`, but fixed stackoverflow.
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
            }
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
impl<T: One + Zero> From<T> for Ratio<Complex<T>> {
    fn from(value: T) -> Self {
        Ratio {
            numer: Complex::real(value),
            denom: Complex::real(T::one()),
        }
    }
}

impl<T: FromU64 + One> FromU64 for Ratio<T> {
    #[inline(always)]
    fn from_u64(value: u64) -> Self {
        Ratio::from(T::from_u64(value))
    }
}

impl<T: Conjugate> Conjugate for Ratio<T> {
    #[inline(always)]
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
        if self.numer.is_zero() {
            // negative zero
            Self {
                numer: self.numer,
                denom: -self.denom,
            }
        } else {
            Self {
                numer: -self.numer,
                denom: self.denom,
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
        impl<'a, T: Cancel + $Add<Output = T>> $Add for &'a Ratio<T> {
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
        impl<T: Cancel + $Add<Output = T>> $Add<T> for Ratio<T> {
            type Output = Ratio<T>;
            fn $add(self, rhs: T) -> Self::Output {
                // avoid overflows by computing the gcd in each operation
                let numer = self.numer.$add(self.denom.clone() * rhs);
                Ratio::new(numer, self.denom)
            }
        }
        impl<'a, T: Cancel> $Add<&'a T> for &'a Ratio<T>
        where
            for<'b> &'b T: $Add<Output = T>,
        {
            type Output = Ratio<T>;
            fn $add(self, rhs: &'a T) -> Self::Output {
                // avoid overflows by computing the gcd in each operation
                let numer = (&self.numer).$add(&(self.denom.clone() * rhs.clone()));
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
impl<T: Cancel> Mul for &Ratio<T> {
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
impl<T: Cancel> Mul<T> for Ratio<T> {
    type Output = Ratio<T>;
    fn mul(self, rhs: T) -> Self::Output {
        // avoid overflows by computing the gcd early (in each operation)
        let (mut rhs, mut denom) = rhs.cancel(self.denom.clone());
        if self.numer.is_zero() && !rhs.is_valid_euclid() {
            rhs = T::zero() - rhs; // floats can have zero signs!
            denom = T::zero() - denom;
        }
        Ratio {
            numer: self.numer * rhs,
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
impl<T: Cancel> Div for &Ratio<T> {
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

impl<T: Cancel + Div<Output = T>> Rem for Ratio<T> {
    type Output = Ratio<T>;
    /// remainder like for floats, not the division remainder, but rather a signed modulo function.
    fn rem(self, rhs: Self) -> Self::Output {
        let f = &self / &rhs;
        &self - &(&rhs * &(f.numer / f.denom))
    }
}
impl<T: Cancel + Div<Output = T>> Rem for &Ratio<T> {
    type Output = Ratio<T>;
    /// remainder like for floats, not the division remainder, but rather a signed modulo function.
    fn rem(self, rhs: Self) -> Self::Output {
        let f = self / rhs;
        self - &(rhs * &(f.numer / f.denom))
    }
}
impl<T: Cancel + Div<Output = T>> Rem<T> for Ratio<T> {
    type Output = Ratio<T>;
    /// remainder like for floats, not the division remainder, but rather a signed modulo function.
    fn rem(self, rhs: T) -> Self::Output {
        let f = &self / &rhs;
        self - (rhs * (f.numer / f.denom))
    }
}
impl<'a, T: Cancel + Div<Output = T>> Rem<&'a T> for &'a Ratio<T> {
    type Output = Ratio<T>;
    /// remainder like for floats, not the division remainder, but rather a signed modulo function.
    fn rem(self, rhs: &'a T) -> Self::Output {
        let f = self / rhs;
        self.clone() - (rhs.clone() * (f.numer / f.denom))
    }
}

// alternative: rational division as integers
// -> bring them to the same denominator, then do the remainder
// problem: overflows easily.
impl<T: Cancel> Euclid for Ratio<T> {
    fn div_rem_euclid(&self, div: &Self) -> (Self, Self) {
        if div.numer.is_zero() {
            (T::zero().into(), self.clone())
        } else {
            // always exactly divisible.
            (self / div, T::zero().into())
        }
    }
    fn is_valid_euclid(&self) -> bool {
        // only the finite 0 is a possible output of `div_rem_euclid`, however either x or -x need to be valid,
        // so use rem_euclid on the numerator and call some more numbers valid euclid.
        (self.numer.is_valid_euclid() == self.denom.is_valid_euclid()) && !self.denom.is_zero()
    }
}

use crate::forward_assign_impl;
forward_assign_impl!(
    Ratio;
    AddAssign, (Add), (), {Cancel}, [AddMul], add_assign, add;
    SubAssign, (Sub), (), {Cancel}, [AddMul], sub_assign, sub;
    MulAssign, (Mul), (), {Cancel}, [AddMul], mul_assign, mul;
    DivAssign, (Div), (), {Cancel}, [AddMul], div_assign, div;
    RemAssign, (Rem), (Div), {Cancel}, [AddMul], rem_assign, rem;
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
    T::Real: Num<Real = T::Real> + Cancel,
    Ratio<T>: From<Ratio<T::Real>>,
{
    type Real = Ratio<T::Real>;
    const CHAR: u64 = T::CHAR;
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
        if d == self.denom {
            Ratio {
                numer: self.numer.re(),
                denom: self.denom.re(),
            }
        } else {
            // do the trick of multiplying with denom.conj()
            // the result is not cancelled! This is BAD for real numbers, hence the check above.
            // (all operations also work without canceling and canceling doesn't extend the range of this operation)
            Ratio {
                numer: (self.numer.clone() * d).re(),
                denom: self.denom.abs_sqr(),
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
impl<T: NumAlgebraic + Cancel> NumAlgebraic for Ratio<T>
where
    T::Real: Num<Real = T::Real> + Cancel,
    Ratio<T>: From<Ratio<T::Real>>,
{
    fn abs(&self) -> Self::Real {
        Ratio::new_raw(self.numer.abs(), self.denom.abs())
    }
    fn sqrt(&self) -> Self {
        if self.denom.is_valid_euclid() {
            Ratio::new_raw(self.numer.sqrt(), self.denom.sqrt())
        } else {
            Ratio::new_raw(
                (T::zero() - self.numer.clone()).sqrt(),
                (T::zero() - self.denom.clone()).sqrt(),
            )
        }
    }
    fn cbrt(&self) -> Self {
        Ratio::new_raw(self.numer.cbrt(), self.denom.cbrt())
    }
    fn sign(&self) -> Self {
        Ratio::new(self.numer.sign(), self.denom.sign())
    }
    fn copysign(&self, sign: &Self) -> Self {
        let mut r: Self = self.abs().into();
        r.numer = r.numer.copysign(&sign.numer);
        r.denom = r.denom.copysign(&sign.denom);
        r
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
            let mut v16 = T::one() + T::one();
            v16 = v16.clone() * v16;
            v16 = v16.clone() * v16;
            let v16f: F = v16.to_approx();
            // do a loop instead of recursion to avoid stack overflows at all cost.
            let mut normalized = value * v16f.clone();
            let mut norm_tol = tol * v16f.clone();
            let mut fac = v16.clone();
            while !(F::one() / normalized.clone()).is_finite() {
                normalized = normalized * v16f.clone();
                norm_tol = norm_tol * v16f.clone();
                fac = fac * v16.clone();
            }
            return Some(&Self::from_approx(normalized, norm_tol)? / &fac);
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
    fn to_approx(&self) -> F {
        // not always safe to just compute it directly.
        // The unsafe case is, where the denominator is so big,
        // that an infinite result is returned, but a
        // subnormal float result would be possible.
        // To fix that, divided by 16 until the result is finite.
        let mut n = self.numer.to_approx();
        let mut denom = self.denom.clone();
        if denom != denom {
            return F::zero() / F::zero();
        }
        let mut d = denom.to_approx();
        if !n.is_zero() && !d.is_finite() {
            let mut v16 = T::one() + T::one();
            v16 = v16.clone() * v16;
            v16 = v16.clone() * v16;
            let v16f: F = v16.to_approx();
            loop {
                denom = denom.div_rem_euclid(&v16).0;
                n = n / v16f.clone();
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
            if r.denom.is_zero() {
                return r.denom.hash(state);
            }
            let (int, rem) = r.numer.div_rem_euclid(&r.denom);
            (r.denom, r.numer) = (r.numer, rem);
            int.hash(state);
        }
    }
}

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
