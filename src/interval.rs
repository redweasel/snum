//! Interval arithmetic without correct floating point rounding.

use core::ops::*;

use crate::*;

/// Type for exact interval arithmetic.
///
/// Note, that floating point rounding rules are not
/// easily available in Rust and are therefore omitted.
/// That means the bounds will have rounding errors and should
/// not be taken as fully reliable! This type is meant for
/// algorithms like [minimize_interval] and [roots_interval],
/// which depend on interval arithmetic.
///
/// When defining functions with this, remember that it
/// (approximately) associative and commutative, but not distributive.
/// Also consider, that the interval may overshoot outside of the defined
/// region of a function like `sqrt`. In that case it is automatically
/// clamped to the defined region before evaluating it.
///
/// # Examples
/// ```rust
/// use snum::interval::*;
/// let b = Bounded::from(1).extend(2);
/// println!("{}", b);
/// assert_eq!(b, Bounded::from(2).extend(1));
/// assert!(b.contains(&1)); // both bounds are inclusive
/// assert!(b.contains(&2));
/// assert!(!b.contains(&3));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Bounded<T> {
    min: T,
    max: T,
}

impl<T: RealNum> From<T> for Bounded<T> {
    fn from(value: T) -> Self {
        assert!(value == value, "Converting NaN to bounded is not allowed");
        Self {
            min: value.clone(),
            max: value,
        }
    }
}

impl<T: RealNum> From<&[T]> for Bounded<T> {
    fn from(value: &[T]) -> Self {
        let mut res = Self::from(value[0].clone());
        for v in &value[1..] {
            res = res.extend(v.clone());
        }
        res
    }
}
impl<T: RealNum, const N: usize> From<&[T; N]> for Bounded<T> {
    fn from(value: &[T; N]) -> Self {
        let mut res = Self::from(value[0].clone());
        for v in &value[1..] {
            res = res.extend(v.clone());
        }
        res
    }
}

impl<T> Bounded<T> {
    pub fn lower(&self) -> &T {
        &self.min
    }
    pub fn upper(&self) -> &T {
        &self.max
    }
}

impl<T: RealNum> Bounded<T> {
    pub unsafe fn raw(lower: T, upper: T) -> Self {
        Self { min: lower, max: upper }
    }
    pub fn extend(self, other: T) -> Self {
        if self.min > other {
            Self { min: other, max: self.max }
        } else if self.max < other {
            Self { min: self.min, max: other }
        } else {
            self // NaNs are ignored!
        }
    }
    pub fn contains(&self, value: &T) -> bool {
        &self.min <= value && value <= &self.max
    }
    pub fn contains_bounded(&self, value: &Bounded<T>) -> bool {
        self.min <= value.min && value.max <= self.max
    }
    pub fn union(self, other: Self) -> Self {
        let min = if self.min < other.min { self.min } else { other.min };
        let max = if self.max > other.max { self.max } else { other.max };
        Self { min, max }
    }
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let min = if self.min > other.min { self.min.clone() } else { other.min.clone() };
        let max = if self.max < other.max { self.max.clone() } else { other.max.clone() };
        (min <= max).then_some(Self { min, max })
    }
    pub fn max(self, other: Self) -> Self {
        Self {
            min: if self.min > other.min { self.min } else { other.min },
            max: if self.max > other.max { self.max } else { other.max },
        }
    }
    pub fn min(self, other: Self) -> Self {
        Self {
            min: if self.min < other.min { self.min } else { other.min },
            max: if self.max < other.max { self.max } else { other.max },
        }
    }
}
impl<T: RealNum> Bounded<T> {
    pub fn max_const(self, other: T) -> Self {
        Self {
            min: if self.min > other { self.min } else { other.clone() },
            max: if self.max > other { self.max } else { other.clone() },
        }
    }
    pub fn min_const(self, other: T) -> Self {
        Self {
            min: if self.min < other { self.min } else { other.clone() },
            max: if self.max < other { self.max } else { other.clone() },
        }
    }
}
impl<T: RealNum> Bounded<T>
where
    for<'a> &'a T: Sub<Output = T>,
{
    /// Get the width of the interval computed as *upper bound - lower bound*.
    pub fn width(&self) -> T {
        &self.max - &self.min
    }
}
impl<T: RealNum + One + Div<Output = T>> Bounded<T>
where
    for<'a> &'a T: Add<Output = T>,
{
    /// Get the center of the interval computed as *(upper bound + lower bound)/2*.
    pub fn mid(&self) -> T {
        if T::CHAR == 2 {
            // can not divide by 2, as 1+1=0
            self.min.clone() // as good as any really
        } else {
            (&self.max + &self.min) / (&T::one() + &T::one())
        }
    }
}
impl<T: RealNum + Zero + Sub<Output = T> + One + Div<Output = T>> Bounded<T> {
    /// Try to split the interval in the middle.
    /// Fails if one of the split intervals contains only one numeric value.
    /// If the interval is infinite, it works with the following rules:
    ///
    /// - (-∞, ∞) is split at `0`
    /// - (a, ∞) is split at `a + |a| + 1`
    /// - (-∞, a) is split at `a - |a| - 1`
    pub fn split_mid(self) -> Option<(Self, Self)> {
        if self.min == self.max {
            return None;
        }
        let split = |x: T, b: Self| Some((Self { min: b.min, max: x.clone() }, Self { min: x, max: b.max }));
        if T::CHAR == 2 {
            // can not divide by 2, as 1+1=0
            // split by using the two values directly
            // This is correct as the number type is a real number (i.e. one dimensional)
            // and there is only one such finite field of characteristic 2 (up to isomorphy)
            return split(self.min.clone(), self);
        }
        let mininf = !(self.min.clone() - self.min.clone()).is_zero();
        let maxinf = !(self.max.clone() - self.max.clone()).is_zero();
        if mininf && maxinf {
            // they are different, as that was already checked, so it must be (-∞, ∞) assuming all infinities are equal.
            return split(T::zero(), self);
        }
        if mininf {
            return split(
                if self.max >= T::zero() {
                    T::zero() - T::one()
                } else {
                    self.max.clone() + self.max.clone() - T::one()
                },
                self,
            );
        }
        if maxinf {
            return split(
                if self.min >= T::zero() {
                    self.min.clone() + self.min.clone() + T::one()
                } else {
                    T::one()
                },
                self,
            );
        }
        let mid = (self.min.clone() + self.max.clone()) / (T::one() + T::one());
        (mid != self.min && mid != self.max).then_some(()).and(split(mid, self))
    }
}

// Safety: `Bounded<T>` is `repr(C)` and contains only instances of `T`, so we
// can guarantee it contains no *added* padding. Thus, if `T: Zeroable`,
// `Bounded<T>` is also `Zeroable`
#[cfg(feature = "bytemuck")]
unsafe impl<T: RealNum + bytemuck::Zeroable> bytemuck::Zeroable for Bounded<T> {}
#[cfg(feature = "bytemuck")]
unsafe impl<T: RealNum + bytemuck::AnyBitPattern> bytemuck::AnyBitPattern for Bounded<T> {}
#[cfg(feature = "bytemuck")]
unsafe impl<T: RealNum + bytemuck::NoUninit> bytemuck::NoUninit for Bounded<T> {}

impl<T: RealNum + Neg<Output = T>> Neg for Bounded<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            min: -self.max,
            max: -self.min,
        }
    }
}

impl<T: RealNum + Add<Output = T>> Add for Bounded<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            min: self.min + rhs.min,
            max: self.max + rhs.max,
        }
    }
}

impl<T: RealNum + Sub<Output = T>> Sub for Bounded<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            min: self.min - rhs.max,
            max: self.max - rhs.min,
        }
    }
}

impl<T: RealNum + Mul<Output = T>> Mul for Bounded<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        // 4 results
        let mut corners = [
            self.min.clone() * rhs.min.clone(),
            self.max.clone() * rhs.min,
            self.min * rhs.max.clone(),
            self.max * rhs.max,
        ];
        // their order is unclear due to sign differences -> compute all and sort.
        // NaN values ain't allowed per construction, so the type is actually Ord.
        corners.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let [min, _, _, max] = corners;
        Self { min, max }
    }
}

impl<T: RealNum + Div<Output = T> + Zero + One + Neg<Output = T>> Div for Bounded<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        let mut rmin = rhs.min;
        let mut rmax = rhs.max;
        // if 0 is in [rmin, rmax], set both to zero with the correct zero sign (if such a thing exists)
        if rmin < T::zero() && rmax > T::zero() {
            rmin = -T::zero();
            rmax = T::zero();
        }
        if (rmin.is_zero() || rmax.is_zero()) && self.min <= T::zero() && self.max >= T::zero() {
            // 0 / 0 = NaN usually, but with bounded [-inf, inf] is a better solution
            return Self {
                min: -T::one() / T::zero(),
                max: T::one() / T::zero(),
            };
        }
        // 4 results
        let mut corners = [
            self.min.clone() / rmin.clone(),
            self.max.clone() / rmin,
            self.min / rmax.clone(),
            self.max / rmax,
        ];
        // their order is unclear due to sign differences -> compute all and sort.
        // NaN values ain't allowed per construction, so the type is actually Ord.
        corners.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let [min, _, _, max] = corners;
        Self { min, max }
    }
}

impl<T: RealNum + Add<Output = T>> Add<T> for Bounded<T> {
    type Output = Self;
    fn add(self, rhs: T) -> Self::Output {
        Self {
            min: self.min + rhs.clone(),
            max: self.max + rhs.clone(),
        }
    }
}

impl<T: RealNum + Sub<Output = T>> Sub<T> for Bounded<T> {
    type Output = Self;
    fn sub(self, rhs: T) -> Self::Output {
        Self {
            min: self.min - rhs.clone(),
            max: self.max - rhs.clone(),
        }
    }
}

impl<T: RealNum + Mul<Output = T>> Mul<T> for Bounded<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        let min = self.min * rhs.clone();
        let max = self.max * rhs.clone();
        if max >= min { Self { min, max } } else { Self { min: max, max: min } }
    }
}

impl<T: RealNum + Div<Output = T> + Zero> Div<T> for Bounded<T> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        if rhs >= T::zero() {
            Self {
                min: self.min / rhs.clone(),
                max: self.max / rhs.clone(),
            }
        } else {
            Self {
                min: self.max / rhs.clone(),
                max: self.min / rhs.clone(),
            }
        }
    }
}

macro_rules! forward_binary_impl {
    ($mul:ident, $Mul:ident, $mul_assign:ident, $MulAssign:ident, ($rhs:ty, $rhs2:ty), $($deps:tt)*) => {
        impl<'a, T: RealNum + $Mul<Output = T> $($deps)*> $Mul<$rhs> for &'a Bounded<T> {
            type Output = Bounded<T>;
            fn $mul(self, rhs: $rhs) -> Self::Output {
                self.clone().$mul(rhs.clone())
            }
        }
        impl<'a, T: RealNum + $Mul<Output = T> $($deps)*> $MulAssign<$rhs> for Bounded<T> {
            fn $mul_assign(&mut self, rhs: $rhs) {
                take_mut::take(self, |x| x.$mul(rhs.clone()));
            }
        }
        impl<'a, T: RealNum + $Mul<Output = T> $($deps)*> $MulAssign<$rhs2> for Bounded<T> {
            fn $mul_assign(&mut self, rhs: $rhs2) {
                take_mut::take(self, |x| x.$mul(rhs));
            }
        }
    };
}

forward_binary_impl!(add, Add, add_assign, AddAssign, (&'a Bounded<T>, Bounded<T>),);
forward_binary_impl!(sub, Sub, sub_assign, SubAssign, (&'a Bounded<T>, Bounded<T>),);
forward_binary_impl!(mul, Mul, mul_assign, MulAssign, (&'a Bounded<T>, Bounded<T>),);
forward_binary_impl!(div, Div, div_assign, DivAssign, (&'a Bounded<T>, Bounded<T>), + Zero + One + Neg<Output = T>);
forward_binary_impl!(add, Add, add_assign, AddAssign, (&'a T, T),);
forward_binary_impl!(sub, Sub, sub_assign, SubAssign, (&'a T, T),);
forward_binary_impl!(mul, Mul, mul_assign, MulAssign, (&'a T, T),);
forward_binary_impl!(div, Div, div_assign, DivAssign, (&'a T, T), + Zero + Neg<Output = T>);

impl<T: RealNum + Zero> Zero for Bounded<T> {
    fn zero() -> Self {
        Self {
            min: T::zero(),
            max: T::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        self.min.is_zero() && self.max.is_zero()
    }
}
impl<T: RealNum + One> One for Bounded<T> {
    fn one() -> Self {
        Self {
            min: T::one(),
            max: T::one(),
        }
    }
    fn is_one(&self) -> bool {
        self.min.is_one() && self.max.is_one()
    }
}

impl<T: RealNum> Conjugate for Bounded<T> {
    fn conj(&self) -> Self {
        self.clone()
    }
}

impl<T: RealNum + Zero + Mul<Output = T>> Num for Bounded<T> {
    const CHAR: u64 = T::CHAR;
    type Real = Bounded<T::Real>;
    fn abs_sqr(&self) -> Self::Real {
        // always positive and independent of sign (better bound than x*x)
        let mut min = self.min.abs_sqr();
        let mut max = self.max.abs_sqr();
        if min > max {
            (min, max) = (max, min);
        }
        if self.min.clone() * self.max.clone() < T::zero() {
            min.set_zero();
        }
        Self { min, max }
    }
    fn is_unit(&self) -> bool {
        self.min > T::zero() || self.max < T::zero()
    }
    fn re(&self) -> Self::Real {
        self.clone()
    }
}

macro_rules! forward_monotone {
    ($f:ident) => {
        fn $f(&self) -> Self {
            if self.min == self.max {
                Self::from(self.min.$f())
            } else {
                Self {
                    min: self.min.$f(),
                    max: self.max.$f(),
                }
            }
        }
    };
    ($f:ident, $min:expr) => {
        fn $f(&self) -> Self {
            Self {
                min: if self.min <= $min { ($min).$f() } else { self.min.$f() },
                max: self.max.$f(),
            }
        }
    };
    ($f:ident, $min:expr, $max:expr) => {
        fn $f(&self) -> Self {
            Self {
                min: if self.min <= $min { ($min).$f() } else { self.min.$f() },
                max: if self.max >= $max { ($max).$f() } else { self.max.$f() },
            }
        }
    };
}

impl<T: NumAlgebraic<Real = T> + Zero + PartialOrd + Neg<Output = T> + Mul<Output = T>> NumAlgebraic for Bounded<T> {
    fn abs(&self) -> Self::Real {
        // same as abs_sqr()
        let mut min = self.min.abs();
        let mut max = self.max.abs();
        if min > max {
            (min, max) = (max, min);
        }
        if self.min.clone() * self.max.clone() < T::zero() {
            min.set_zero();
        }
        Self { min, max }
    }
    fn copysign(&self, sign: &Self) -> Self {
        // only interesting if the sign is different.
        let smin = sign.min.sign();
        let smax = sign.max.sign();
        let amin = self.min.abs();
        let amax = self.max.abs();
        let r = if amin < amax { amax } else { amin };
        if smin != smax {
            Self { min: -r.clone(), max: r }
        } else if smin > T::zero() {
            Self { min: T::zero(), max: r }
        } else {
            Self { min: -r, max: T::zero() }
        }
    }
    forward_monotone!(sign);
    forward_monotone!(sqrt, T::zero());
}

impl<T: NumElementary<Real = T> + Zero + One + PartialOrd + Sub<Output = T> + Neg<Output = T> + Mul<Output = T> + Div<Output = T>> NumElementary
    for Bounded<T>
{
    forward_monotone!(cbrt);
    forward_monotone!(asin, -T::one(), T::one());
    forward_monotone!(asinh);
    forward_monotone!(acosh);
    forward_monotone!(tanh);
    forward_monotone!(atanh, -T::one(), T::one());
    forward_monotone!(atan);
    forward_monotone!(sinh);
    forward_monotone!(exp);
    forward_monotone!(exp_m1);
    forward_monotone!(ln, T::zero());
    forward_monotone!(ln_1p, -T::one());
    fn sin(&self) -> Self {
        let pi = (-T::one()).acos();
        let d = self.max.clone() - self.min.clone();
        if d > pi.clone() + pi.clone() {
            return Self {
                min: -T::one(),
                max: T::one(),
            };
        }
        let mut min = self.min.sin();
        if self.min == self.max {
            return Self::from(min);
        }
        let mut max = self.max.sin();
        let cmin = self.min.cos();
        let cmax = self.max.cos();
        if min > max {
            (min, max) = (max, min);
        }
        if d > pi && (cmin >= T::zero()) == (cmax >= T::zero()) {
            max.set_one();
            min = -T::one();
        }
        if cmin > T::zero() && cmax < T::zero() {
            max.set_one();
        }
        if cmin < T::zero() && cmax > T::zero() {
            min = -T::one();
        }
        Self { min, max }
    }
    fn cos(&self) -> Self {
        let pi = (-T::one()).acos();
        let d = self.max.clone() - self.min.clone();
        if d > pi.clone() + pi.clone() {
            return Self {
                min: -T::one(),
                max: T::one(),
            };
        }
        let mut min = self.min.cos();
        if self.min == self.max {
            return Self::from(min);
        }
        let mut max = self.max.cos();
        let cmin = self.min.sin();
        let cmax = self.max.sin();
        if min > max {
            (min, max) = (max, min);
        }
        if d > pi && (cmin >= T::zero()) == (cmax >= T::zero()) {
            max.set_one();
            min = -T::one();
        }
        if cmin > T::zero() && cmax < T::zero() {
            min = -T::one();
        }
        if cmin < T::zero() && cmax > T::zero() {
            max.set_one();
        }
        Self { min, max }
    }
    fn tan(&self) -> Self {
        if self.min == self.max {
            return Self::from(self.min.tan());
        }
        let pi = (-T::one()).acos();
        if self.max.clone() - self.min.clone() > pi {
            return Self {
                min: -T::one() / T::zero(),
                max: T::one() / T::zero(),
            };
        }
        let cmin = self.min.cos();
        let cmax = self.max.cos();
        if (cmin >= T::zero()) != (cmax >= T::zero()) {
            return Self {
                min: -T::one() / T::zero(),
                max: T::one() / T::zero(),
            };
        }
        Self {
            min: self.min.tan(),
            max: self.max.tan(),
        }
    }
    fn acos(&self) -> Self {
        Self {
            min: if self.max >= T::one() { T::zero() } else { self.max.acos() },
            max: if self.min <= -T::one() { (-T::one()).acos() } else { self.min.acos() },
        }
    }
    fn cosh(&self) -> Self {
        // same as abs_sqr()
        let mut min = self.min.cosh();
        if self.min == self.max {
            return Self::from(min);
        }
        let mut max = self.max.cosh();
        if min > max {
            (min, max) = (max, min);
        }
        if self.min.clone() * self.max.clone() < T::zero() {
            min.set_one();
        }
        Self { min, max }
    }
    fn atan2(&self, x: &Self) -> Self {
        // 4 results like mul
        let mut corners = [
            self.min.atan2(&x.min),
            self.max.atan2(&x.min),
            self.min.atan2(&x.max),
            self.max.atan2(&x.max),
        ];
        // their order is unclear due to sign differences -> compute all and sort.
        // NaN values ain't allowed per construction, so the type is actually Ord.
        corners.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let [min, _, _, max] = corners;
        Self { min, max }
    }
    fn pow(&self, exp: &Self) -> Self {
        (self.ln() * exp.clone()).exp()
    }
}

impl<T: RealNum + Neg<Output = T> + Zero + One + From<T::Discrete> + IntoDiscrete> IntoDiscrete for Bounded<T>
where
    T::Discrete: RealNum + Neg<Output = T::Discrete> + Zero + One + Div<Output = T::Discrete>,
{
    type Discrete = Bounded<<T as IntoDiscrete>::Discrete>;
    fn ceil(&self) -> Self::Discrete {
        Bounded {
            min: self.min.ceil(),
            max: self.max.ceil(),
        }
    }
    fn div_floor(&self, div: &Self) -> Self::Discrete {
        let mut rmin = div.min.clone();
        let mut rmax = div.max.clone();
        // if 0 is in [rmin, rmax], set both to zero with the correct zero sign (if such a thing exists)
        if rmin < T::zero() && rmax > T::zero() {
            rmin = -T::zero();
            rmax = T::zero();
        }
        if (rmin.is_zero() || rmax.is_zero()) && self.min <= T::zero() && self.max >= T::zero() {
            // 0 / 0 = NaN usually, but with bounded [-inf, inf] is a better solution
            return Bounded {
                min: -T::Discrete::one() / T::Discrete::zero(),
                max: T::Discrete::one() / T::Discrete::zero(),
            };
        }
        // 4 results
        let mut corners = [
            self.min.div_floor(&rmin),
            self.max.div_floor(&rmin),
            self.min.div_floor(&rmax),
            self.max.div_floor(&rmax),
        ];
        // their order is unclear due to sign differences -> compute all and sort.
        // NaN values ain't allowed per construction, so the type is actually Ord.
        corners.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let [min, _, _, max] = corners;
        Bounded { min, max }
    }
    fn floor(&self) -> Self::Discrete {
        Bounded {
            min: self.min.floor(),
            max: self.max.floor(),
        }
    }
    fn round(&self) -> Self::Discrete {
        Bounded {
            min: self.min.round(),
            max: self.max.round(),
        }
    }
}

impl<T: RealNum + Zero + One + Neg<Output = T> + Sub<Output = T> + Div<Output = T> + Euclid> Euclid for Bounded<T> {
    fn div_rem_euclid(&self, div: &Self) -> (Self, Self) {
        let d = self.max.clone() - self.min.clone();
        let mut rmin = div.min.clone();
        let mut rmax = div.max.clone();
        // if 0 is in [rmin, rmax], set both to zero with the correct zero sign (if such a thing exists)
        if rmin < T::zero() && rmax > T::zero() {
            rmin = -T::zero();
            rmax = T::zero();
        }
        let rem_max = if div.max > -div.min.clone() { div.max.clone() } else { -div.min.clone() };
        if rmin <= T::zero() && rmax >= T::zero() {
            // 0 / 0 = NaN usually, but with bounded [-inf, inf] is a better solution
            let min = T::one() / rmin.clone() / T::zero();
            let max = T::one() / rmax.clone() / T::zero();
            if min != max {
                return (
                    Self { min, max },
                    Self {
                        min: T::zero(),
                        max: rem_max,
                    },
                );
            }
        }
        // 4 results
        let mut corners = [
            self.min.clone().div_rem_euclid(&rmin).0,
            self.max.clone().div_rem_euclid(&rmin).0,
            self.min.clone().div_rem_euclid(&rmax).0,
            self.max.clone().div_rem_euclid(&rmax).0,
        ];
        // their order is unclear due to sign differences -> compute all and sort.
        corners.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let [min, _, _, max] = corners;
        let q = Self { min, max };
        let r;
        if d >= rem_max {
            // self always covers the entire region of div.
            r = Self {
                min: T::zero(),
                max: rem_max,
            };
        } else {
            r = (self.clone() - div * &q).max_const(T::zero()).min_const(rem_max);
        }
        (q, r)
    }
    fn is_valid_euclid(&self) -> bool {
        self.lower().is_valid_euclid()
    }
}

impl<F: FloatType, T: RealNum + ApproxFloat<F>> ApproxFloat<F> for Bounded<T> {
    fn from_approx(value: F, tol: F) -> Option<Self> {
        // create an interval with ~ tol as size around value
        let r = tol / F::from_u64(2);
        Some(Bounded {
            min: T::from_approx(value.clone() - r.clone(), r.clone())?,
            max: T::from_approx(value.clone() + r.clone(), r.clone())?,
        })
    }
    fn to_approx(&self) -> F {
        (self.min.to_approx() + self.max.to_approx()) / F::from_u64(2)
    }
}

/// Global minimization of a multivariate function which uses interval arithmetic.
/// Depending on the function, this procedure may need up to O(2^N) in time and memory (this is unavoidable).
/// For convex functions it usually requires only O(N) time and memory.
///
/// The returned value is either a point (`Ok`) or a bounded region in which multiple minima of the same quallity are found (`Err`).
///
/// # Performance Optimization
/// The call pattern for the function is `[a, b], [b, b], [b, c]`,
/// so for optimal performance one may implement it in a way, which reuses the
/// last upper bound as new lower bound. That is not possible with regular
/// interval arithmetic, but when using custom implementations for the bound
/// estimation, it can become relevant.
#[cfg(feature = "std")]
pub fn interval_minimize<T: Field + RealNum, const N: usize>(
    mut f: impl FnMut(&[Bounded<T>; N]) -> Bounded<T>,
    bounds: [Bounded<T>; N],
    tol: [T; N],
    ftol: T,
    max_iter: usize,
) -> Result<[T; N], [Bounded<T>; N]>
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
    assert!(ftol >= T::zero());
    assert!(tol.iter().all(|tol| tol >= &T::zero()));
    // check for NaNs in the bounds, as that invalidates the Ord implementation!
    // If there is any, return an error with the unchanged bounds immediately.
    // The invariants of the type don't allow NaNs, but with unsafe it can still
    // be created and result in panics in the heap below.
    if bounds.iter().any(|b| {
        b.lower().partial_cmp(b.lower()) != Some(core::cmp::Ordering::Equal) || b.upper().partial_cmp(b.upper()) != Some(core::cmp::Ordering::Equal)
    }) {
        return Err(bounds);
    }
    let initial_res = f(&bounds);
    #[derive(PartialEq, Hash)]
    struct Head<T: RealNum, const N: usize> {
        res: Bounded<T>,
        bounds: [Bounded<T>; N],
    }
    impl<T: RealNum, const N: usize> Eq for Head<T, N> {}
    impl<T: RealNum, const N: usize> Ord for Head<T, N> {
        fn cmp(&self, other: &Self) -> core::cmp::Ordering {
            // this should never fail for the bounded type, as it doesn't produce new NaNs.
            // If it fails, panic!
            other.res.lower().partial_cmp(self.res.lower()).unwrap()
        }
    }
    impl<T: RealNum, const N: usize> PartialOrd for Head<T, N> {
        fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    let mut heap = std::collections::BinaryHeap::<Head<T, N>>::new();
    let mut min_upper = initial_res.upper().clone();
    let mut min_upper_bounds = bounds.clone();
    heap.push(Head {
        res: initial_res,
        bounds: bounds,
    });
    let mut iterations = 0;
    while let Some(head) = heap.pop() {
        // check if the head can still improve the minimum by more than the tolerance
        if !(head.res.lower() + &ftol < min_upper) {
            continue; // it can't
        }
        if iterations >= max_iter {
            break;
        }
        // determine the largest direction and bisect along it.
        // this direction is very important, as it determines how well the bounds work.
        let mut i = head
            .bounds
            .iter()
            .zip(&tol)
            .enumerate()
            .max_by(|a, b| {
                let aw = a.1.0.width();
                let bw = b.1.0.width();
                if a.1.1.is_zero() && b.1.1.is_zero() || !(&aw - &aw).is_zero() || !(&bw - &bw).is_zero() {
                    // both have tolerance zero or one is infinite -> just compare unweighted
                    aw.partial_cmp(&bw).unwrap()
                } else {
                    (&aw * b.1.1).partial_cmp(&(&bw * a.1.1)).unwrap()
                }
            })
            .unwrap()
            .0;
        let mut a = head.bounds.clone();
        let mut b = head.bounds.clone();
        for _ in 0..N {
            // In here, there is no way to avoid O(N) work, as f(x) is at least O(N).
            // That means any attempt to reduce computation by using trees will not yield a significant improvment.
            if head.bounds[i].width() >= tol[i]
                && let Some((l, r)) = head.bounds[i].clone().split_mid()
            {
                (a[i], b[i]) = (l, r);
                // compute the function in the two parts and update min_upper.
                // also compute the midpoint to get a closer bound on min_upper!
                // ideally these computations would be combined into one as there is quite
                // some repetition, but that is not what interval arithmetic is made for...
                let a_res = f(&a);
                let mid = b.clone().map(|b| Bounded::from(b.min));
                let mid_res = f(&mid);
                let b_res = f(&b);
                // since mid is a subset of both a and b, mid_res will be a subset of both
                // a_res and b_res, so mid_res.max will always be smaller than the other two.
                if mid_res.upper() < &min_upper {
                    min_upper = mid_res.upper().clone();
                    min_upper_bounds = mid.clone();
                }
                // add the two parts back to the heap.
                heap.push(Head { res: a_res, bounds: a });
                heap.push(Head { res: b_res, bounds: b });
                iterations += 1;
                break;
            } else {
                // try another coordinate. Due to missing type information we can't know
                // if the initial choice of i was the "most splittable" choice, so we have
                // to try all choices to ensure we don't accidently drop a non-empty interval.
                i += 1;
                if i >= N {
                    i -= N;
                }
            }
        }
        // regularly filter the heap to remove garbage. Don't do this too often!
        // this is only meant to reduce the memory footprint of the algorithm.
        if iterations & 0x3FFF == 0 {
            heap.retain(|h| h.res.lower() + &ftol < min_upper);
        }
    }
    if heap.is_empty() {
        // finished successfully
        if min_upper_bounds.iter().zip(&tol).all(|(b, tol)| &b.clone().width() > tol) {
            Err(min_upper_bounds)
        } else {
            Ok(min_upper_bounds.map(|b| b.mid()))
        }
    } else {
        // if the heap has elements, the minimization was not successful.
        // compress the result by union of all bounded parts, which touch.
        let res = heap
            .into_iter()
            .filter(|h| h.res.lower() + &ftol < min_upper)
            .map(|h| h.bounds)
            .reduce(|a, b| core::array::from_fn(|i| a[i].clone().union(b[i].clone())))
            .unwrap(); // unwrap is ok, because it's non empty and at least the first element can't be filtered out.
        Err(res)
    }
}

/// Find all zeros/roots of a univariate function in a given interval `a <= b`.
/// The roots are given as bounds as well. For a function like `f(x)=0`,
/// this will result in one big interval. At most `2^max_iter_per_root`
/// can be found and they are ordered from smallest to largest.
///
/// Note, that if there is points where the function skips discontinuously over zero,
/// the jump is also considered a root. This is done this way, because most functions
/// in floating point arithmetic don't actually hit zero exactly on their roots.
///
/// # Panics
/// If the number type has characteristic 2 or
/// `a <= b` doesn't hold, e.g. because one of them is NaN.
#[cfg(feature = "std")]
pub fn interval_roots<T: RealNum + Zero + One + Sub<Output = T> + Div<Output = T>>(
    mut f: impl FnMut(&Bounded<T>) -> Bounded<T>,
    a: T,
    b: T,
    max_iter_per_root: usize,
) -> std::vec::Vec<Bounded<T>>
where
    for<'a> &'a T: Add<Output = T> + Sub<Output = T>,
{
    assert!(T::CHAR != 2);
    assert!(a <= b);

    let zero = &T::zero();
    let init = Bounded { min: a, max: b };
    if !f(&init).contains(zero) {
        // allocation free in this case
        return std::vec::Vec::with_capacity(0);
    }
    // the algorithm is trivial to implement using recursion, however I want to avoid stack overflows,
    // so I'm opting for an iterative "fractal" algorithm instead with at most O(n^2) steps.
    let mut roots = std::vec::Vec::with_capacity(max_iter_per_root + 1);
    let mut roots_swap = std::vec::Vec::with_capacity(max_iter_per_root + 1);
    roots.push((init, false));
    for _ in 0..=max_iter_per_root {
        for (bound, is_zero) in roots.drain(..) {
            if !is_zero && let Some((a, b)) = bound.clone().split_mid() {
                let fa = f(&a);
                let fb = f(&b);
                if fa.contains(zero) {
                    roots_swap.push((a, fa.is_zero()));
                }
                if fb.contains(zero) {
                    roots_swap.push((b, fb.is_zero()));
                }
            } else {
                roots_swap.push((bound, true));
            }
        }
        core::mem::swap(&mut roots, &mut roots_swap);
    }
    // roots can have bounds, which are back-to-back -> merge them!
    let mut res = roots.into_iter().map(|b| b.0).collect::<std::vec::Vec<_>>();
    let mut fixup = |b: &mut Bounded<T>| {
        if b.max != b.min {
            if b.clone().split_mid().is_none() {
                // there is no more value in this range, so one of the bounds is the best (or maybe both)
                let bmin = Bounded::from(b.min.clone());
                let bmax = Bounded::from(b.max.clone());
                let mut fmin = f(&bmin).mid();
                let mut fmax = f(&bmax).mid();
                if fmin < T::zero() {
                    fmin = T::zero() - fmin;
                }
                if fmax < T::zero() {
                    fmax = T::zero() - fmax;
                }
                if fmin < fmax {
                    *b = bmin;
                } else if fmin > fmax {
                    *b = bmax;
                }
            }
        }
    };
    for i in (1..res.len()).rev() {
        fixup(&mut res[i]);
        if res[i - 1].max == res[i].min {
            // This is a tradeoff. Either reallocate a new vector or do these removes.
            // It is expected that this doesn't happen very often for regular parameters.
            res[i - 1].max = res.remove(i).max;
        }
    }
    if !res.is_empty() {
        fixup(&mut res[0]);
    }
    res
}

/// Find the first zero/root of a univariate function in a given interval `a <= b`.
///
/// Note, that if there is points where the function skips discontinuously over zero,
/// the jump is also considered a root. This is done this way, because most functions
/// in floating point arithmetic don't actually hit zero exactly on their roots.
///
/// # Panics
/// If the number type has characteristic 2 or
/// `a <= b` doesn't hold, e.g. because one of them is NaN.
///
/// If `max_iter_per_root` is too large this function can cause stackoverflows.
/// The recursion depth is less or equal to `max_iter_per_root`.
pub fn interval_root<T: RealNum + Zero + One + Sub<Output = T> + Div<Output = T>>(
    mut f: impl FnMut(&Bounded<T>) -> Bounded<T>,
    a: T,
    b: T,
    max_iter_per_root: usize,
) -> Option<Bounded<T>>
where
    for<'a> &'a T: Add<Output = T> + Sub<Output = T>,
{
    assert!(T::CHAR != 2);
    assert!(a <= b);
    // the algorithm is trivial to implement using recursion and it is implemented using recursion,
    // because that makes it possible to do this in no_std mode and without const generics.
    // max_iter_per_root defines the depth of the stack directly.
    fn rec<T: RealNum + Zero + One + Sub<Output = T> + Div<Output = T>>(
        bound: Bounded<T>,
        i: usize,
        f: &mut impl FnMut(&Bounded<T>) -> Bounded<T>,
    ) -> Option<Bounded<T>>
    where
        for<'a> &'a T: Add<Output = T> + Sub<Output = T>,
    {
        if !f(&bound).contains(&T::zero()) {
            return None;
        }
        if i == 0 {
            Some(bound)
        } else if let Some((a, b)) = bound.clone().split_mid() {
            rec(a, i - 1, f).or_else(|| rec(b, i - 1, f))
        } else {
            Some(Bounded::from(bound.mid()))
        }
    }
    rec(Bounded { min: a, max: b }, max_iter_per_root, &mut f)
}

/// Compute an integral in a reliable way. This method can handle any function,
/// e.g. discontinuous functions. Due to the assumption, that the function
/// may be discontinuous, the bounds are limited to a precision of `O(1/max_iter)`.
///
/// Returns the result of the adaptive grid midpoint rule and the error bounds,
/// which may not be centered on the midpoint result.
#[cfg(feature = "std")]
pub fn interval_integral<T: Field + RealNum + core::fmt::LowerExp, const N: usize>(
    mut f: impl FnMut(&[Bounded<T>; N]) -> Bounded<T>,
    bounds: [Bounded<T>; N],
    tol: T,
    max_iter: usize,
) -> (T, Bounded<T>)
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
    assert!(tol >= T::zero());
    // use a binary heap to structure the integration order.
    #[derive(PartialEq, Hash)]
    struct Volume<T: RealNum, const N: usize> {
        res: Bounded<T>,
        bounds: [Bounded<T>; N],
        bvolume: T,
        volume: T,
    }
    impl<T: RealNum + Zero + One + Div<Output = T>, const N: usize> Volume<T, N>
    where
        for<'a> &'a T: AddMulSub<Output = T>,
    {
        pub fn new(res: Bounded<T>, bounds: [Bounded<T>; N]) -> Self {
            let mut bvolume = T::one();
            for b in &bounds {
                bvolume = bvolume * b.width();
            }
            let volume = &bvolume * &res.width();
            Self {
                res,
                bounds,
                bvolume,
                volume,
            }
        }
        pub fn mid(&self) -> [T; N] {
            core::array::from_fn(|i| self.bounds[i].mid())
        }
        pub fn integral(&self) -> Bounded<T> {
            let w = &self.bvolume;
            if (w - w).is_zero() {
                &self.res * w
            } else {
                Bounded::from(&T::zero() - w).extend(w.clone())
            }
        }
    }
    impl<T: RealNum, const N: usize> Eq for Volume<T, N> {}
    impl<T: RealNum + Zero + One + Sub<Output = T> + Div<Output = T>, const N: usize> Ord for Volume<T, N> {
        fn cmp(&self, other: &Self) -> core::cmp::Ordering {
            // this should never fail for the bounded type, as it doesn't produce new NaNs.
            // If it fails, panic!
            self.volume.partial_cmp(&other.volume).unwrap()
        }
    }
    impl<T: RealNum + Zero + One + Sub<Output = T> + Div<Output = T>, const N: usize> PartialOrd for Volume<T, N> {
        fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    let mut heap = std::collections::BinaryHeap::new();
    let mut vol_f = |b| Volume::new(f(&b), b);
    let initial = vol_f(bounds);
    let mut res = initial.integral();
    heap.push(initial);
    let mut iterations = 0;
    while iterations < max_iter
        && res.width() > tol
        && let Some(vol) = heap.pop()
    {
        // replace vol by the bisected volumes
        let mut i = vol
            .bounds
            .iter()
            .enumerate()
            .max_by(|a, b| {
                // both have tolerance zero or one is infinite -> just compare unweighted
                // can't produce new NaNs here, as the subtraction in width is ordered.
                a.1.width().partial_cmp(&b.1.width()).unwrap()
            })
            .unwrap()
            .0;
        let mut a = vol.bounds.clone();
        let mut b = vol.bounds.clone();
        for _ in 0..N {
            if let Some((l, r)) = vol.bounds[i].clone().split_mid() {
                (a[i], b[i]) = (l, r);
                let vol_a = vol_f(a);
                let vol_b = vol_f(b);
                let da = vol_a.integral();
                let db = vol_b.integral();
                // add the two parts back to the heap (filter zeros to make infinite integrals possible)
                let mut reinit = false;
                if !vol_a.res.is_zero() {
                    heap.push(vol_a);
                } else if !(&vol_a.bvolume - &vol_a.bvolume).is_zero() {
                    reinit = true;
                }
                if !vol_b.res.is_zero() {
                    heap.push(vol_b);
                } else if !(&vol_b.bvolume - &vol_b.bvolume).is_zero() {
                    reinit = true;
                }
                if reinit {
                    res = Bounded::zero();
                    for v in &heap {
                        res = res + v.integral();
                    }
                } else {
                    // correct the integration result (avoid infinities!)
                    let w = res.width();
                    if (&w - &w).is_zero() {
                        let vol_int = vol.integral();
                        res.max = res.max - vol_int.max;
                        res.min = res.min - vol_int.min;
                        res = res + da;
                        res = res + db;
                    }
                }
                break;
            } else {
                // try another coordinate. Due to missing type information we can't know
                // if the initial choice of i was the "most splittable" choice, so we have
                // to try all choices to ensure we don't accidently drop a non-empty interval.
                i += 1;
                if i >= N {
                    i -= N;
                }
            }
        }
        // Non-splittable intervals are considered zero size (even though they may contain 1ulp)
        // This is done deliberately to filter out singularities.
        // if res is non finite, recalculate it now.
        let w = res.width();
        if !(&w - &w).is_zero() {
            res = Bounded::zero();
            for v in &heap {
                res = res + v.integral();
            }
        }
        // still count up iterations though, as every interval might be non splittable and
        // I want to avoid infinite loops at all cost.
        iterations += 1;
    }
    let mut res_mid = T::zero();
    for vol in heap {
        res_mid = res_mid + f(&vol.mid().map(Bounded::from)).mid() * vol.bvolume;
    }
    (res_mid, res)
}
