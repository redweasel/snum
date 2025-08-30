//! implements a `Complex<T>` type, that in large parts matches (is a copy of) the complex type of `num_complex`.
//! However a lot more functionallity is available with weak trait bounds.

use crate::*;
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::*;
use take_mut::take;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

impl<T> Complex<T> {
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

impl<T: Zero> Complex<T> {
    pub fn real(re: T) -> Self {
        Self {
            re,
            im: Zero::zero(),
        }
    }
    pub fn imag(im: T) -> Self {
        Self {
            re: Zero::zero(),
            im,
        }
    }
}

impl<T: Zero> Zero for Complex<T>
where
    for<'a> &'a T: Add<&'a T, Output = T>,
{
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }
    fn zero() -> Self {
        Self {
            re: Zero::zero(),
            im: Zero::zero(),
        }
    }
}

impl<T: Zero + One> One for Complex<T>
where
    for<'a> &'a T: AddMulSub<Output = T>,
{
    fn is_one(&self) -> bool {
        self.re.is_one() && self.im.is_zero()
    }
    fn one() -> Self {
        Self {
            re: One::one(),
            im: Zero::zero(),
        }
    }
}

impl<T: Zero + One> Complex<T> {
    pub fn i() -> Self {
        Self {
            re: Zero::zero(),
            im: One::one(),
        }
    }
}

impl<T: Neg<Output = T>> Complex<T> {
    /// Multiply with i with correct handling of infinities.
    pub fn mul_i(self) -> Self {
        Self {
            re: -self.im,
            im: self.re,
        }
    }
}

// Safety: `Complex<T>` is `repr(C)` and contains only instances of `T`, so we
// can guarantee it contains no *added* padding. Thus, if `T: Zeroable`,
// `Complex<T>` is also `Zeroable`
#[cfg(feature = "bytemuck")]
unsafe impl<T: bytemuck::Zeroable> bytemuck::Zeroable for Complex<T> {}

// Safety: `Complex<T>` is `repr(C)` and contains only instances of `T`, so we
// can guarantee it contains no *added* padding. Thus, if `T: Pod`,
// `Complex<T>` is also `Pod`
#[cfg(feature = "bytemuck")]
unsafe impl<T: bytemuck::Pod> bytemuck::Pod for Complex<T> {}

impl<T: Zero> From<T> for Complex<T> {
    fn from(value: T) -> Self {
        Self {
            re: value,
            im: Zero::zero(),
        }
    }
}

impl<T: FromU64 + Zero> FromU64 for Complex<T> {
    #[inline(always)]
    fn from_u64(value: u64) -> Self {
        Complex::from(T::from_u64(value))
    }
}

impl<T: Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

macro_rules! impl_add {
    ($Add:ident, $add:ident) => {
        impl<T> $Add<Complex<T>> for Complex<T>
        where
            for<'a> &'a T: $Add<&'a T, Output = T>,
        {
            type Output = Complex<T>;
            fn $add(self, rhs: Complex<T>) -> Self::Output {
                Self {
                    re: self.re.$add(&rhs.re),
                    im: self.im.$add(&rhs.im),
                }
            }
        }
        impl<'a, T: Clone + $Add<T, Output = T>> $Add<&'a Complex<T>> for &'a Complex<T> {
            type Output = Complex<T>;
            fn $add(self, rhs: &'a Complex<T>) -> Self::Output {
                Complex {
                    re: self.re.clone().$add(rhs.re.clone()),
                    im: self.im.clone().$add(rhs.im.clone()),
                }
            }
        }
    };
}
impl_add!(Add, add);
impl_add!(Sub, sub);

impl<T> Mul<Complex<T>> for Complex<T>
where
    for<'a> &'a T: AddMulSub<Output = T>,
{
    type Output = Complex<T>;
    fn mul(self, rhs: Complex<T>) -> Self::Output {
        Self {
            re: &(&self.re * &rhs.re) - &(&self.im * &rhs.im),
            im: &(&self.im * &rhs.re) + &(&self.re * &rhs.im),
        }
    }
}
impl<'a, T: Clone + Add<T, Output = T> + Mul<T, Output = T> + Sub<T, Output = T>>
    Mul<&'a Complex<T>> for &'a Complex<T>
{
    type Output = Complex<T>;
    fn mul(self, rhs: &'a Complex<T>) -> Self::Output {
        Complex {
            re: self.re.clone() * rhs.re.clone() - self.im.clone() * rhs.im.clone(),
            im: self.im.clone() * rhs.re.clone() + self.re.clone() * rhs.im.clone(),
        }
    }
}

macro_rules! impl_add_real {
    ($Add: ident, $add: ident) => {
        impl<T> $Add<T> for Complex<T>
        where
            for<'a> &'a T: $Add<&'a T, Output = T>,
        {
            type Output = Complex<T>;
            fn $add(self, rhs: T) -> Self::Output {
                Self {
                    re: self.re.$add(&rhs),
                    im: self.im,
                }
            }
        }
        impl<'a, T: Clone + $Add<T, Output = T>> $Add<&'a T> for &'a Complex<T> {
            type Output = Complex<T>;
            fn $add(self, rhs: &'a T) -> Self::Output {
                Complex {
                    re: self.re.clone().$add(rhs.clone()),
                    im: self.im.clone(),
                }
            }
        }
        // can't implement the reverse, because Rust doesn't allow it.
    };
}
impl_add_real!(Add, add);
impl_add_real!(Sub, sub);

macro_rules! impl_mul_real {
    ($Mul: ident, $mul: ident) => {
        impl<T> $Mul<T> for Complex<T>
        where
            for<'a> &'a T: $Mul<&'a T, Output = T>,
        {
            type Output = Complex<T>;
            fn $mul(self, rhs: T) -> Self::Output {
                Self {
                    re: self.re.$mul(&rhs),
                    im: self.im.$mul(&rhs),
                }
            }
        }
        impl<'a, T: Clone + $Mul<T, Output = T>> $Mul<&'a T> for &'a Complex<T> {
            type Output = Complex<T>;
            fn $mul(self, rhs: &'a T) -> Self::Output {
                Complex {
                    re: self.re.clone().$mul(rhs.clone()),
                    im: self.im.clone().$mul(rhs.clone()),
                }
            }
        }
    };
}
impl_mul_real!(Mul, mul);
impl_mul_real!(Div, div);
impl_mul_real!(Rem, rem);

impl<T> Div for Complex<T>
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
    type Output = Complex<T>;
    fn div(self, rhs: Complex<T>) -> Self::Output {
        let abs_sqr = &(&rhs.re * &rhs.re) + &(&rhs.im * &rhs.im);
        Self {
            re: &(&(&self.re * &rhs.re) + &(&self.im * &rhs.im)) / &abs_sqr,
            im: &(&(&self.im * &rhs.re) - &(&self.re * &rhs.im)) / &abs_sqr,
        }
    }
}
impl<'a, T: Clone + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Div<Output = T>> Div
    for &'a Complex<T>
{
    type Output = Complex<T>;
    fn div(self, rhs: &'a Complex<T>) -> Self::Output {
        let abs_sqr = rhs.re.clone() * rhs.re.clone() + rhs.im.clone() * rhs.im.clone();
        Complex {
            re: (self.re.clone() * rhs.re.clone() + self.im.clone() * rhs.im.clone())
                / abs_sqr.clone(),
            im: (self.im.clone() * rhs.re.clone() - self.re.clone() * rhs.im.clone()) / abs_sqr,
        }
    }
}

impl<T: Clone + Neg<Output = T> + Add<Output = T> + Mul<Output = T> + Div<Output = T>> Complex<T> {
    pub fn recip(self) -> Self {
        let abs_sqr = self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone();
        Complex {
            re: self.re / abs_sqr.clone(),
            im: -self.im / abs_sqr,
        }
    }
}

impl<T: Clone + One> Rem for Complex<T>
where
    for<'a> &'a T: AddMulSubDiv<Output = T> + Rem<&'a T, Output = T>,
{
    type Output = Complex<T>;
    fn rem(self, rhs: Self) -> Self::Output {
        let Complex { re, im } = self.clone() / rhs.clone();
        let gaussian = Complex::new(&re - &(&re % &T::one()), &im - &(&im % &T::one()));
        self - rhs * gaussian
    }
}

impl<
    'a,
    T: Clone
        + One
        + Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Rem<Output = T>,
> Rem for &'a Complex<T>
{
    type Output = Complex<T>;
    fn rem(self, rhs: Self) -> Self::Output {
        let Complex { re, im } = self / rhs;
        let gaussian = Complex::new(re.clone() - (re % T::one()), im.clone() - (im % T::one()));
        self - &(rhs * &gaussian)
    }
}

impl<T: Num + Euclid + Zero + One + Neg<Output = T> + Sub<T, Output = T> + Div<T, Output = T>>
    Euclid for Complex<T>
{
    /// Euclidean division of complex numbers, such that `|r|^2 <= |b|^2/2`
    /// is satisfied for the remainder `r` with non negative real part.
    fn div_rem_euclid(&self, b: &Self) -> (Self, Self) {
        if b.re.is_zero() && b.im.is_zero() {
            // invalid division -> invalid result, which still holds up the equation self = q * div + r = r;
            return (T::zero().into(), self.clone());
        }
        let a = self;
        // NOTE: this is very performance critical code, yet it needs precise function
        let b_sqr = b.re.clone() * b.re.clone() + b.im.clone() * b.im.clone(); // avoid some trait bounds
        if b_sqr.is_unit() {
            return (&(a * &b.conj()) / &b_sqr, T::zero().into());
        }
        let two = T::one() + T::one();
        // https://stackoverflow.com/a/18067292
        let rounded_div = |a: T, b: T| {
            let b2 = b.clone() / two.clone();
            (if a.is_valid_euclid() == b.is_valid_euclid() {
                a + b2
            } else {
                a - b2
            }) / b
        };
        let ab = a * &b.conj();
        let m = rounded_div(ab.re, b_sqr.clone()); // = (a/b).re().round() (round(1/2) == 1)
        let n = rounded_div(ab.im, b_sqr); // = (a/b).im().round()
        let q = Complex { re: m, im: n };
        let r = a - &(b * &q);
        // unique euclidean division ensures the following property:
        //debug_assert!(r.re.clone()*r.re.clone()+r.im.clone()*r.im.clone() <= (b.re.clone()*b.re.clone()+b.im.clone()*b.im.clone()) / two);
        (q, r)
    }
    fn is_valid_euclid(&self) -> bool {
        // this has nothing to do anymore with negative/positive,
        // as the remainder algorithm also returns values with negative real part.
        true
    }
}

macro_rules! forward_assign_impl {
    ($($AddAssign:ident, ($($Add:ident),*), $(($One:ident),)? $add_assign:ident, $add:ident),+) => {
        $(impl<T: Clone $(+ $One)?> $AddAssign for Complex<T>
            where for<'a> &'a T: Add<Output = T> $(+ $Add<Output = T>)+ {
            fn $add_assign(&mut self, rhs: Complex<T>) {
                take(self, |x| x.$add(rhs));
            }
        }
        impl<T: Clone $(+ $One)?> $AddAssign<T> for Complex<T>
            where for<'a> &'a T: Add<Output = T> $(+ $Add<Output = T>)+ {
            fn $add_assign(&mut self, rhs: T) {
                take(self, |x| x.$add(rhs));
            }
        }
        impl<'a, T: Clone $(+ $One)? $(+ $Add<Output = T>)+> $AddAssign<&'a Complex<T>> for Complex<T> {
            fn $add_assign(&mut self, rhs: &'a Complex<T>) {
                take(self, |x| (&x).$add(rhs));
            }
        }
        impl<'a, T: Clone $(+ $One)? $(+ $Add<Output = T>)+> $AddAssign<&'a T> for Complex<T> {
            fn $add_assign(&mut self, rhs: &'a T) {
                take(self, |x| (&x).$add(rhs));
            }
        })+
    };
}
forward_assign_impl!(
    AddAssign,
    (Add),
    add_assign,
    add,
    SubAssign,
    (Sub),
    sub_assign,
    sub,
    MulAssign,
    (Add, Mul, Sub),
    mul_assign,
    mul,
    DivAssign,
    (Add, Mul, Sub, Div),
    div_assign,
    div,
    RemAssign,
    (Add, Mul, Sub, Div, Rem),
    (One),
    rem_assign,
    rem
);

impl<T: Zero + Add<Output = T>> Sum for Complex<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::new(T::zero(), T::zero()), |acc, c| Self {
            re: acc.re.add(c.re),
            im: acc.im.add(c.im),
        })
    }
}
impl<'a, T: Zero> Sum<&'a Complex<T>> for Complex<T>
where
    for<'b> &'b T: Add<Output = T>,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Complex<T>>,
    {
        iter.fold(Self::new(T::zero(), T::zero()), |acc, c| Self {
            re: (&acc.re).add(&c.re),
            im: (&acc.im).add(&c.im),
        })
    }
}

impl<T: Zero + One> Product for Complex<T>
where
    for<'a> &'a T: AddMulSub<Output = T>,
{
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::new(T::one(), T::zero()), |acc, c| acc * c)
    }
}
impl<'a, T: Clone + Zero + One> Product<&'a Complex<T>> for Complex<T>
where
    for<'b> &'b T: AddMulSub<Output = T>,
{
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Complex<T>>,
    {
        iter.fold(Self::new(T::one(), T::zero()), |acc, c| acc * c.clone())
    }
}

impl<T: NumElementary + Zero + Neg<Output = T> + Mul<T, Output = T>> Complex<T> {
    /// Calculate the principal Arg of self.
    #[inline(always)]
    pub fn arg(&self) -> T {
        self.im.atan2(&self.re)
    }
    /// Convert to polar form (r, theta), such that `self = r * exp(i * theta)`.
    ///
    /// Returns (`r`, `theta`)
    pub fn to_polar(&self) -> (T, T) {
        // implement abs_sqr without the reference based addition and multiplication to keep trait bound simpler.
        let abs_sqr = self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone();
        (abs_sqr.sqrt(), self.arg())
    }
    /// Convert a polar representation  `r * exp(i * theta)` into a complex number.
    pub fn from_polar(r: T, theta: T) -> Self {
        Self::new(r.clone() * theta.cos(), r * theta.sin())
    }
    /// Compute `cis(phase) := exp(i * phase)`, which is the more efficient version of `from_polar(1, phase)`.
    /// "cis" is an acronym for "*cos i sin".
    pub fn cis(phase: T) -> Self {
        Self::new(phase.cos(), phase.sin())
    }
}

impl<T: Clone + Neg<Output = T>> Conjugate for Complex<T> {
    #[inline(always)]
    fn conj(&self) -> Self {
        // treat T as real valued and don't cascade complex conjugation
        Complex {
            re: self.re.clone(),
            im: -self.im.clone(),
        }
    }
}
impl<T: Num + Zero + Neg<Output = T>> Num for Complex<T>
where
    for<'a> &'a T: AddMul<Output = T>,
{
    type Real = T;
    const CHAR: u64 = T::CHAR;
    #[inline(always)]
    fn abs_sqr(&self) -> Self::Real {
        &(&self.re * &self.re) + &(&self.im * &self.im)
    }
    #[inline(always)]
    fn re(&self) -> Self::Real {
        self.re.clone()
    }
    #[inline(always)]
    fn is_unit(&self) -> bool {
        // Note, this can easily overflow.
        self.abs_sqr().is_unit()
    }
}

impl<T: AlgebraicField<Real = T> + NumElementary + PartialOrd> NumAlgebraic for Complex<T>
where
    Complex<T>: Num<Real = T>,
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
    fn sqrt(&self) -> Self {
        if self.is_zero() {
            return Self {
                re: T::zero(),
                im: self.im.clone(),
            }; // copy imaginary zero sign
        }
        if !self.re.is_zero() && &(&(&self.im * &self.im) / &self.re) + &self.re == self.re {
            let sqrt = self.re.abs().sqrt();
            let im = &self.im / &(&(T::one() + T::one()) * &sqrt);
            if self.re >= T::zero() {
                return Complex { im, re: sqrt };
            } else {
                // this makes it different for 0.0 and -0.0 !!!
                return Complex::new(im.abs(), sqrt.copysign(&self.im));
            }
        }
        // Use angle bisection
        // Note, this approach is limited by the types epsilon at 1.0 for re << 0
        // e.g. in theory (-1.0 + 1e-20 i).sqrt() = (0.5e-20 + i), but rounding will occur here.
        // However (-1.0f64 + 2e-8 i).sqrt() = (1e-8 + i) still works.
        let len = self.abs(); // sqrt eval 1
        let mut half: Complex<T> = self.clone() / len.clone();
        half.re = half.re + T::one();
        let fac = (len / half.abs_sqr()).sqrt(); // sqrt eval 2
        half * fac
    }
    /// Computes the principal value of the cube root of `self`.
    ///
    /// This function has one branch cut:
    ///
    /// * `(-∞, 0)`, continuous from above.
    ///
    /// The branch satisfies `-π/3 ≤ arg(cbrt(z)) ≤ π/3`.
    ///
    /// Note that this does not match the usual result for the cube root of
    /// negative real numbers. For example, the real cube root of `-8` is `-2`,
    /// but the principal complex cube root of `-8` is `1 + i√3`.
    fn cbrt(&self) -> Self {
        if self.im.is_zero() {
            if self.re >= T::zero() {
                // simple positive real ∛r, and copy `im` for its sign
                Self::new(self.re.cbrt(), self.im.clone())
            } else {
                // ∛(r e^(iπ)) = ∛r e^(iπ/3) = ∛r/2 + i∛r√3/2
                // ∛(r e^(-iπ)) = ∛r e^(-iπ/3) = ∛r/2 - i∛r√3/2
                let one = T::one();
                let two = &one + &one;
                let three = &two + &one;
                let re = (-self.re.clone()).cbrt() / two;
                let im = &three.sqrt() * &re;
                Self::new(re.clone(), im.copysign(&self.im))
            }
        } else if self.re.is_zero() {
            // ∛(r e^(iπ/2)) = ∛r e^(iπ/6) = ∛r√3/2 + i∛r/2
            // ∛(r e^(-iπ/2)) = ∛r e^(-iπ/6) = ∛r√3/2 - i∛r/2
            let one = T::one();
            let two = &one + &one;
            let three = &two + &one;
            let im = self.im.abs().cbrt() / two;
            let re = &three.sqrt() * &im;
            Self::new(re, im.copysign(&self.im))
        } else {
            // cbrt can NOT be done faster than with polar decomposition.
            // see: impossibility of the trisection of an angle.
            // exact formula: cbrt(r e^(it)) = cbrt(r) e^(it/3)
            // alternatively, there would be an extremely fast converging algorithm using sqrt and cbrt.
            let one = T::one();
            let three = &one + &one + one;
            let (r, theta) = self.to_polar(); // used NumElementary here!
            Self::from_polar(r.cbrt(), theta / three)
        }
    }
    #[inline(always)]
    fn abs(&self) -> Self::Real {
        // `ComplexFloat` in `num_complex` uses hypot.
        // Due to the trait bounds, I have to do it by sqrt(), which has slightly worse behavior.
        // This is 2x as fast as hypot. In most cases abs_sqr() should be sufficient though.
        self.abs_sqr().sqrt()
    }
    #[inline(always)]
    fn sign(&self) -> Self {
        let a = self.abs_sqr();
        if a.is_zero() {
            Complex::one()
        } else {
            self.clone() / a.sqrt()
        }
    }
    #[inline(always)]
    fn copysign(&self, sign: &Self) -> Self {
        let (a, b) = (self.abs_sqr(), sign.abs_sqr());
        if b.is_zero() {
            a.sqrt().into()
        } else {
            sign.clone() * (a / b).sqrt()
        }
    }
}

// NumElementary implementation with very few conditionals (only based on zero check)
impl<T: NumElementary + AlgebraicField<Real = T> + PartialOrd> NumElementary for Complex<T>
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
    fn sin(&self) -> Self {
        Self {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }

    fn cos(&self) -> Self {
        Self {
            re: self.re.cos() * self.im.cosh(),
            im: -self.re.sin() * self.im.sinh(),
        }
    }

    fn tan(&self) -> Self {
        let x = self + self;
        let d = x.re.cos() + x.im.cosh();
        Self {
            re: &x.re.sin() / &d,
            im: &x.im.sinh() / &d,
        }
    }

    /// Computes the principal value of the inverse sine of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞, -1)`, continuous from above.
    /// * `(1, ∞)`, continuous from below.
    ///
    /// The branch satisfies `-π/2 ≤ Re(asin(z)) ≤ π/2`.
    fn asin(&self) -> Self {
        // formula: arcsin(z) = -i ln(sqrt(1-z^2) + iz)
        -(self.clone().mul_i() + (-(self * self - T::one())).sqrt())
            .ln()
            .mul_i()
    }

    /// Computes the principal value of the inverse cosine of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞, -1)`, continuous from above.
    /// * `(1, ∞)`, continuous from below.
    ///
    /// The branch satisfies `0 ≤ Re(acos(z)) ≤ π`.
    fn acos(&self) -> Self {
        // formula: arccos(z) = -i ln(i sqrt(1-z^2) + z)
        -(self + &(-(self * self - T::one())).sqrt().mul_i())
            .ln()
            .mul_i()
    }

    /// Computes the principal value of the inverse tangent of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞i, -i]`, continuous from the left.
    /// * `[i, ∞i)`, continuous from the right.
    ///
    /// The branch satisfies `-π/2 ≤ Re(atan(z)) ≤ π/2`.
    fn atan(&self) -> Self {
        // formula: arctan(z) = (ln(1+iz) - ln(1-iz))/(2i)
        // This can not simply be implemented as ln((1+iz)/(1-iz))/(2i), as that would break at the branch cuts.
        // see atanh for the explanation of the fix.
        if self.im.is_zero() {
            return self.re.atan().into();
        }
        let one = &T::one();
        let two = (one + one).copysign(&self.im);
        let s = (self.clone().mul_i() - self.im.sign()).abs_sqr();
        (Complex::new(one - &self.abs_sqr(), &self.re * &two) / s).ln().mul_i() / -two
    }

    /// Compute the principal value of the extended inverse tangent.
    /// For real x > 0, this is equivalent to arctan(y/x), where `y = self`.
    ///
    /// This function has branchcuts defined by the implicit equations:
    /// `y * x* - y* * sqrt(x^2 + y^2)` purely imaginary and `|y| >= |x + sqrt(x^2 + y^2)|`
    fn atan2(&self, x: &Self) -> Self {
        if self.is_zero() && x.is_zero() {
            return Complex::zero(); // avoid NaN
        }
        // implement it as one function
        // formula: 2*arctan(y / (x + (x*x + y*y).sqrt()))
        let div = x + &(x * x + self * self).sqrt();
        let res = if div.is_zero() {
            // div = x * (1 - (1 + (self/x)^2).sqrt()) with real x < 0
            // use 1. order Taylor series for sqrt to get div ~ (self*self).sqrt()/2
            let div = (self*self).sqrt();
            // directly at the branchcut, decide for one side.
            T::zero().acos().copysign(&(self * &div.conj()).re()).into()
        } else {
            (self / &div).atan()
        };
        &res + &res
    }

    fn exp(&self) -> Self {
        let r = self.re.exp();
        if r.is_zero() {
            return Self::zero();
        }
        if self.im.is_zero() {
            return Self::real(r);
        }
        Self {
            re: &r * &self.im.cos(),
            im: &r * &self.im.sin(),
        }
    }

    fn exp_m1(&self) -> Self {
        let r = self.re.exp_m1();
        let two = T::one() + T::one();
        let s = (&self.im / &two).sin();
        let s = &s * &s * two;
        // (r + 1) * cos - 1 = r * cos + (cos - 1) // lossy at im~0
        // cos - 1 = -2sin(im/2)^2 // precise at im~0
        Self {
            re: &r * &(&T::one() - &s) - s,
            im: (r + T::one()) * self.im.sin(),
        }
    }

    /// Computes the principal value of natural logarithm of `self`.
    ///
    /// This function has one branch cut:
    ///
    /// * `(-∞, 0]`, continuous from above.
    ///
    /// The branch satisfies `-π ≤ arg(ln(z)) ≤ π`.
    #[inline(always)]
    fn ln(&self) -> Self {
        let (r, phi) = self.to_polar();
        Self {
            re: r.ln(),
            im: phi,
        }
    }

    /// Computes the principal value of natural logarithm of `self + 1`.
    ///
    /// This function has one branch cut:
    ///
    /// * `(-∞, -1]`, continuous from above.
    ///
    /// The branch satisfies `-π ≤ arg(ln(z+1)) ≤ π`.
    #[inline(always)]
    fn ln_1p(&self) -> Self {
        // no good way to write this without conditionals...
        (self + &T::one()).ln()
    }

    fn sinh(&self) -> Self {
        Self {
            re: self.re.sinh() * self.im.cos(),
            im: self.re.cosh() * self.im.sin(),
        }
    }

    fn cosh(&self) -> Self {
        Self {
            re: self.re.cosh() * self.im.cos(),
            im: self.re.sinh() * self.im.sin(),
        }
    }

    fn tanh(&self) -> Self {
        // formula: tanh(a + bi) = (sinh(2a) + i*sin(2b))/(cosh(2a) + cos(2b))
        let x = self + self;
        let d = x.re.cosh() + x.im.cos();
        Self {
            re: &x.re.sinh() / &d,
            im: &x.im.sin() / &d,
        }
    }

    /// Computes the principal value of inverse hyperbolic sine of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞i, -i)`, continuous from the left.
    /// * `(i, ∞i)`, continuous from the right.
    ///
    /// The branch satisfies `-π/2 ≤ Im(asinh(z)) ≤ π/2`.
    fn asinh(&self) -> Self {
        // formula: arsinh(z) = ln(z + sqrt(1+z^2))
        (self + &(self * self + T::one()).sqrt()).ln()
    }

    /// Computes the principal value of inverse hyperbolic cosine of `self`.
    ///
    /// This function has one branch cut:
    ///
    /// * `(-∞, 1)`, continuous from above.
    ///
    /// The branch satisfies `-π ≤ Im(acosh(z)) ≤ π` and `0 ≤ Re(acosh(z)) < ∞`.
    fn acosh(&self) -> Self {
        // formula: arcosh(z) = 2 ln(sqrt((z+1)/2) + sqrt((z-1)/2))
        let one = &T::one();
        let two = &(one + one);
        let res = &((&(self + one) / two).sqrt() + (&(self - one) / two).sqrt()).ln();
        res + res
    }

    /// Computes the principal value of inverse hyperbolic tangent of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞, -1]`, continuous from above.
    /// * `[1, ∞)`, continuous from below.
    ///
    /// The branch satisfies `-π/2 ≤ Im(atanh(z)) ≤ π/2`.
    fn atanh(&self) -> Self {
        // formula: artanh(z) = (ln(1+z) - ln(1-z))/2
        // This can not simply be implemented as ln((1+z)/(1-z))/2, as that would break at the branch cuts.
        // To fix that, use ln((1+z)/(1-z))/2 on z.re < 0 and otherwise -ln((1-z)/(1+z))/2.
        // The division will erase zero signs, so instead implement the division more carefully
        // using (1+z)(1-z*) = 1+2i*Im(z)-|z|^2 and (1-z)(1+z*) = 1-2i*Im(z)-|z|^2.
        // Then combine the two branches using copysign and sign functions.
        if self.re.is_zero() {
            return Complex::imag(self.im.atan());
        }
        let one = &T::one();
        let two = -(one + one).copysign(&self.re);
        let s = (self + &self.re.sign()).abs_sqr();
        (Complex::new(one - &self.abs_sqr(), &self.im * &two) / s).ln() / two
    }

    /// Raises `self` to a complex power.
    fn pow(&self, exp: &Self) -> Self {
        // a slight branch here to handle the zero cases
        if exp.is_zero() {
            return Self::one();
        }
        if self.is_zero() {
            return Self::zero();
        }
        (&self.ln() * exp).exp()
    }
}

#[macro_export]
macro_rules! complex {
    ($x:literal + $y:literal i) => {
        $crate::Complex::new($x, $y)
    };
    ($x:literal - $y:literal i) => {
        $crate::Complex::new($x, -$y)
    };
    ($x:literal + $y:literal j) => {
        $crate::Complex::new($x, $y)
    };
    ($x:literal - $y:literal j) => {
        $crate::Complex::new($x, -$y)
    };
    (($x:expr) + ($y:expr) i) => {
        $crate::Complex::new($x, $y)
    };
    (($x:expr) - ($y:expr) i) => {
        $crate::Complex::new($x, -$y)
    };
    (($x:expr) + ($y:expr) j) => {
        $crate::Complex::new($x, $y)
    };
    (($x:expr) - ($y:expr) j) => {
        $crate::Complex::new($x, -$y)
    };
    ($x:literal + ($y:expr) i) => {
        $crate::Complex::new($x, $y)
    };
    ($x:literal - ($y:expr) i) => {
        $crate::Complex::new($x, -$y)
    };
    ($x:literal + ($y:expr) j) => {
        $crate::Complex::new($x, $y)
    };
    ($x:literal - ($y:expr) j) => {
        $crate::Complex::new($x, -$y)
    };
    (($x:expr) + $y:literal i) => {
        $crate::Complex::new($x, $y)
    };
    (($x:expr) - $y:literal i) => {
        $crate::Complex::new($x, -$y)
    };
    (($x:expr) + $y:literal j) => {
        $crate::Complex::new($x, $y)
    };
    (($x:expr) - $y:literal j) => {
        $crate::Complex::new($x, -$y)
    };
    ($x:literal i) => {
        $crate::Complex::new($crate::Zero::zero(), $x)
    };
    ($x:literal j) => {
        $crate::Complex::new($crate::Zero::zero(), $x)
    };
    (($x:expr) i) => {
        $crate::Complex::new($crate::Zero::zero(), $x)
    };
    (($x:expr) j) => {
        $crate::Complex::new($crate::Zero::zero(), $x)
    };
    ($x:expr) => {
        $x.into()
    };
}

#[inline(never)]
fn fmt_re_im(
    f: &mut fmt::Formatter<'_>,
    real: fmt::Arguments<'_>,
    imag: fmt::Arguments<'_>,
) -> fmt::Result {
    fmt_complex(f, format_args!("{re}{im}i", re = real, im = imag))
}

#[cfg(feature = "std")]
#[inline(always)]
// Currently, we can only apply width using an intermediate `String` (and thus `std`)
pub(crate) fn fmt_complex(f: &mut fmt::Formatter<'_>, complex: fmt::Arguments<'_>) -> fmt::Result {
    use std::string::ToString;
    if let Some(width) = f.width() {
        let s = complex.to_string();
        match f.align() {
            None | Some(fmt::Alignment::Right) => write!(f, "{s:>0$}", width),
            Some(fmt::Alignment::Center) => write!(f, "{s:^0$}", width),
            Some(fmt::Alignment::Left) => write!(f, "{s:<0$}", width),
        }
    } else {
        write!(f, "{}", complex)
    }
}

#[cfg(not(feature = "std"))]
#[inline(always)]
pub(crate) fn fmt_complex(f: &mut fmt::Formatter<'_>, complex: fmt::Arguments<'_>) -> fmt::Result {
    write!(f, "{}", complex)
}

// string conversions
// PartialOrd is required for the pretty printing in std mode.
macro_rules! impl_display {
    ($Display: ident, $s: literal, $pre: literal) => {
        impl<T> fmt::$Display for Complex<T>
        where
            T: fmt::$Display + Clone + Zero + Sub<T, Output = T>,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                return if f.alternate() {
                    if f.sign_plus() {
                        if let Some(prec) = f.precision() {
                            fmt_re_im(
                                f,
                                format_args!(concat!("{:+#.1$", $s, "}"), self.re, prec),
                                format_args!(concat!("{:+#.1$", $s, "}"), self.im, prec),
                            )
                        } else {
                            fmt_re_im(
                                f,
                                format_args!(concat!("{:+#", $s, "}"), self.re),
                                format_args!(concat!("{:+#", $s, "}"), self.im),
                            )
                        }
                    } else {
                        if let Some(prec) = f.precision() {
                            fmt_re_im(
                                f,
                                format_args!(concat!("{:#.1$", $s, "}"), self.re, prec),
                                format_args!(concat!("{:+#.1$", $s, "}"), self.im, prec),
                            )
                        } else {
                            fmt_re_im(
                                f,
                                format_args!(concat!("{:#", $s, "}"), self.re),
                                format_args!(concat!("{:+#", $s, "}"), self.im),
                            )
                        }
                    }
                } else {
                    if f.sign_plus() {
                        if let Some(prec) = f.precision() {
                            fmt_re_im(
                                f,
                                format_args!(concat!("{:+.1$", $s, "}"), self.re, prec),
                                format_args!(concat!("{:+.1$", $s, "}"), self.im, prec),
                            )
                        } else {
                            fmt_re_im(
                                f,
                                format_args!(concat!("{:+", $s, "}"), self.re),
                                format_args!(concat!("{:+", $s, "}"), self.im),
                            )
                        }
                    } else {
                        if let Some(prec) = f.precision() {
                            fmt_re_im(
                                f,
                                format_args!(concat!("{:.1$", $s, "}"), self.re, prec),
                                format_args!(concat!("{:+.1$", $s, "}"), self.im, prec),
                            )
                        } else {
                            fmt_re_im(
                                f,
                                format_args!(concat!("{:", $s, "}"), self.re),
                                format_args!(concat!("{:+", $s, "}"), self.im),
                            )
                        }
                    }
                };
            }
        }
    };
}
impl_display!(Display, "", "");
impl_display!(LowerExp, "e", "");
impl_display!(UpperExp, "E", "");
impl_display!(LowerHex, "x", "0x");
impl_display!(UpperHex, "X", "0x");
impl_display!(Octal, "o", "0o");
impl_display!(Binary, "b", "0b");

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for Complex<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (&self.re, &self.im).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for Complex<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (re, im) = serde::Deserialize::deserialize(deserializer)?;
        Ok(Self::new(re, im))
    }
}
