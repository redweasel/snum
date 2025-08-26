//! implements a Quaternion<T> type

// Note that this is not explicitly written for gamedev as many libraries are. It can still be used for it though.

use crate::complex::*;
use crate::num::*;
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::*;
use take_mut::take;

/// A quaternion defined as `q = re + im_i * i + im_j * j + im_k * k`.
/// The quaternion group is defined by the rules `i^2=j^2=k^2=-1, ij=k`.
/// All three variables i,j,k are considered the complex part, which change
/// sign under conjugation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Quaternion<T> {
    pub im_i: T,
    pub im_j: T,
    pub im_k: T,
    pub re: T,
}

impl<T> Quaternion<T> {
    pub const fn new(re: T, im_i: T, im_j: T, im_k: T) -> Self {
        Self {
            re,
            im_i,
            im_j,
            im_k,
        }
    }
}

impl<T: Zero> Zero for Quaternion<T>
where
    for<'a> &'a T: Add<Output = T>,
{
    fn zero() -> Self {
        Self {
            im_i: T::zero(),
            im_j: T::zero(),
            im_k: T::zero(),
            re: T::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im_i.is_zero() && self.im_j.is_zero() && self.im_k.is_zero()
    }
}

impl<T: Zero + One + Sub<Output = T>> One for Quaternion<T>
where
    for<'a> &'a T: Mul<Output = T>,
{
    fn one() -> Self {
        Self {
            im_i: T::zero(),
            im_j: T::zero(),
            im_k: T::zero(),
            re: T::one(),
        }
    }
    fn is_one(&self) -> bool {
        self.re.is_one() && self.im_i.is_zero() && self.im_j.is_zero() && self.im_k.is_zero()
    }
}

impl<T: Conjugate + Neg<Output = T>> Conjugate for Quaternion<T> {
    #[inline(always)]
    fn conj(&self) -> Self {
        Self {
            im_i: -self.im_i.conj(),
            im_j: -self.im_j.conj(),
            im_k: -self.im_k.conj(),
            re: self.re.conj(),
        }
    }
}

impl<T: Neg<Output = T>> Neg for Quaternion<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            im_i: -self.im_i,
            im_j: -self.im_j,
            im_k: -self.im_k,
            re: -self.re,
        }
    }
}

impl<T: Zero> From<T> for Quaternion<T> {
    fn from(value: T) -> Self {
        Self {
            re: value,
            im_i: T::zero(),
            im_j: T::zero(),
            im_k: T::zero(),
        }
    }
}

macro_rules! impl_add {
    ($Add:ident, $add:ident) => {
        impl<T> $Add for Quaternion<T>
        where
            for<'a> &'a T: $Add<Output = T>,
        {
            type Output = Quaternion<T>;
            fn $add(self, rhs: Self) -> Self::Output {
                Self {
                    im_i: self.im_i.$add(&rhs.im_i),
                    im_j: self.im_j.$add(&rhs.im_j),
                    im_k: self.im_k.$add(&rhs.im_k),
                    re: self.re.$add(&rhs.re),
                }
            }
        }
        impl<'a, T: Clone + $Add<Output = T>> $Add for &'a Quaternion<T> {
            type Output = Quaternion<T>;
            fn $add(self, rhs: Self) -> Self::Output {
                Quaternion {
                    im_i: self.im_i.clone().$add(rhs.im_i.clone()),
                    im_j: self.im_j.clone().$add(rhs.im_j.clone()),
                    im_k: self.im_k.clone().$add(rhs.im_k.clone()),
                    re: self.re.clone().$add(rhs.re.clone()),
                }
            }
        }
        impl<T: Clone + $Add<Output = T>> $Add<T> for Quaternion<T> {
            type Output = Quaternion<T>;
            fn $add(self, rhs: T) -> Self::Output {
                Self {
                    re: self.re.$add(rhs),
                    im_i: self.im_i,
                    im_j: self.im_j,
                    im_k: self.im_k,
                }
            }
        }
        impl<'a, 'b, T: Clone> $Add<&'b T> for &'a Quaternion<T>
        where
            &'a T: $Add<&'b T, Output = T>,
        {
            type Output = Quaternion<T>;
            fn $add(self, rhs: &'b T) -> Self::Output {
                Quaternion {
                    re: self.re.$add(rhs),
                    im_i: self.im_i.clone(),
                    im_j: self.im_j.clone(),
                    im_k: self.im_k.clone(),
                }
            }
        }
    };
}
impl_add!(Add, add);
impl_add!(Sub, sub);

macro_rules! impl_mul_real {
    ($Mul: ident, $mul: ident) => {
        impl<T> $Mul<T> for Quaternion<T>
        where
            for<'a> &'a T: $Mul<Output = T>,
        {
            type Output = Quaternion<T>;
            fn $mul(self, rhs: T) -> Self::Output {
                Self {
                    im_i: self.im_i.$mul(&rhs),
                    im_j: self.im_j.$mul(&rhs),
                    im_k: self.im_k.$mul(&rhs),
                    re: self.re.$mul(&rhs),
                }
            }
        }

        impl<'a, T: Clone + $Mul<Output = T>> $Mul<&'a T> for &'a Quaternion<T> {
            type Output = Quaternion<T>;
            fn $mul(self, rhs: &'a T) -> Self::Output {
                Quaternion {
                    im_i: self.im_i.clone().$mul(rhs.clone()),
                    im_j: self.im_j.clone().$mul(rhs.clone()),
                    im_k: self.im_k.clone().$mul(rhs.clone()),
                    re: self.re.clone().$mul(rhs.clone()),
                }
            }
        }
    };
}
impl_mul_real!(Mul, mul);
impl_mul_real!(Div, div);
impl_mul_real!(Rem, rem);

impl<T: Add<Output = T> + Sub<Output = T>> Mul for Quaternion<T>
where
    for<'a> &'a T: Mul<Output = T>,
{
    type Output = Quaternion<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        Quaternion {
            im_i: &self.re * &rhs.im_i + &self.im_i * &rhs.re + &self.im_j * &rhs.im_k
                - &self.im_k * &rhs.im_j,
            im_j: &self.re * &rhs.im_j + &self.im_j * &rhs.re + &self.im_k * &rhs.im_i
                - &self.im_i * &rhs.im_k,
            im_k: &self.re * &rhs.im_k + &self.im_k * &rhs.re + &self.im_i * &rhs.im_j
                - &self.im_j * &rhs.im_i,
            re: &self.re * &rhs.re
                - &self.im_i * &rhs.im_i
                - &self.im_j * &rhs.im_j
                - &self.im_k * &rhs.im_k,
        }
    }
}
impl<'a, T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>> Mul for &'a Quaternion<T> {
    type Output = Quaternion<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        Quaternion {
            im_i: self.re.clone() * rhs.im_i.clone()
                + self.im_i.clone() * rhs.re.clone()
                + self.im_j.clone() * rhs.im_k.clone()
                - self.im_k.clone() * rhs.im_j.clone(),
            im_j: self.re.clone() * rhs.im_j.clone()
                + self.im_j.clone() * rhs.re.clone()
                + self.im_k.clone() * rhs.im_i.clone()
                - self.im_i.clone() * rhs.im_k.clone(),
            im_k: self.re.clone() * rhs.im_k.clone()
                + self.im_k.clone() * rhs.re.clone()
                + self.im_i.clone() * rhs.im_j.clone()
                - self.im_j.clone() * rhs.im_i.clone(),
            re: self.re.clone() * rhs.re.clone()
                - self.im_i.clone() * rhs.im_i.clone()
                - self.im_j.clone() * rhs.im_j.clone()
                - self.im_k.clone() * rhs.im_k.clone(),
        }
    }
}

impl<T: Num<Real = T> + Neg<Output = T> + Zero> Num for Quaternion<T>
where
    for<'a> &'a T: Mul<Output = T>,
{
    const CHAR: u64 = T::CHAR;
    type Real = T::Real;
    fn abs_sqr(&self) -> Self::Real {
        self.im_i.abs_sqr() + self.im_j.abs_sqr() + self.im_k.abs_sqr() + self.re.abs_sqr()
    }
    #[inline(always)]
    fn re(&self) -> Self::Real {
        self.re.re()
    }
    #[inline(always)]
    fn is_unit(&self) -> bool {
        self.abs_sqr().is_unit()
    }
}

impl<T: Conjugate + Neg<Output = T> + Add<Output = T> + Sub<Output = T>> Div for Quaternion<T>
where
    for<'a> &'a T: Mul<Output = T> + Div<Output = T>,
{
    type Output = Quaternion<T>;
    fn div(self, rhs: Self) -> Self::Output {
        let abs_sqr = &rhs.im_i * &rhs.im_i.conj()
            + &rhs.im_j * &rhs.im_j.conj()
            + &rhs.im_k * &rhs.im_k.conj()
            + &rhs.re * &rhs.re.conj();
        self * rhs.conj() / abs_sqr
    }
}
impl<
    'a,
    T: Clone + Conjugate + Neg<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T>,
> Div for &'a Quaternion<T>
where
    for<'b> &'b T: Mul<Output = T>,
{
    type Output = Quaternion<T>;
    fn div(self, rhs: Self) -> Self::Output {
        let abs_sqr = &rhs.im_i * &rhs.im_i.conj()
            + &rhs.im_j * &rhs.im_j.conj()
            + &rhs.im_k * &rhs.im_k.conj()
            + &rhs.re * &rhs.re.conj();
        &(self.clone() * rhs.conj()) / &abs_sqr
    }
}
impl<T: Conjugate + Neg<Output = T> + Add<Output = T> + Sub<Output = T>> Quaternion<T>
where
    for<'a> &'a T: Mul<Output = T> + Div<Output = T>,
{
    pub fn recip(self) -> Self {
        let abs_sqr = &self.im_i * &self.im_i.conj()
            + &self.im_j * &self.im_j.conj()
            + &self.im_k * &self.im_k.conj()
            + &self.re * &self.re.conj();
        self.conj() / abs_sqr
    }
}

// TODO reduce trait bounds!
macro_rules! forward_assign_impl {
    ($($AddAssign:ident, ($($Add:ident),*), $(($One:ident),)? $add_assign:ident, $add:ident),+) => {
        $(impl<T: Clone $(+$One)? $(+ $Add<Output = T>)+> $AddAssign for Quaternion<T>
        where for<'a> &'a T: Add<Output = T> $(+ $Add<Output = T>)+ {
            fn $add_assign(&mut self, rhs: Quaternion<T>) {
                take(self, |x| x.$add(rhs));
            }
        }
        impl<T: Clone $(+$One)? $(+ $Add<Output = T>)+> $AddAssign<T> for Quaternion<T>
        where for<'b> &'b T: Add<Output = T> $(+ $Add<Output = T>)+ {
            fn $add_assign(&mut self, rhs: T) {
                take(self, |x| x.$add(rhs));
            }
        }
        impl<'a, T: Clone $(+$One)? $(+ $Add<Output = T>)+> $AddAssign<&'a Quaternion<T>> for Quaternion<T>
        where for<'b> &'b T: Add<Output = T> $(+ $Add<Output = T>)+ {
            fn $add_assign(&mut self, rhs: &'a Quaternion<T>) {
                take(self, |x| (&x).$add(rhs));
            }
        }
        impl<'a, T: Clone $(+$One)? $(+ $Add<Output = T>)+> $AddAssign<&'a T> for Quaternion<T>
        where for<'b> &'b T: Add<Output = T> $(+ $Add<Output = T>)+ {
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
    (Neg, Add, Mul, Sub, Div),
    (Conjugate),
    div_assign,
    div
);

impl<T: Zero> Sum for Quaternion<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(
            Self::new(T::zero(), T::zero(), T::zero(), T::zero()),
            |acc, c| Self {
                re: acc.re.add(c.re),
                im_i: acc.im_i.add(c.im_i),
                im_j: acc.im_j.add(c.im_j),
                im_k: acc.im_k.add(c.im_k),
            },
        )
    }
}
impl<'a, T: Zero> Sum<&'a Quaternion<T>> for Quaternion<T>
where
    for<'b> &'b T: Add<Output = T>,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Quaternion<T>>,
    {
        iter.fold(
            Self::new(T::zero(), T::zero(), T::zero(), T::zero()),
            |acc, c| Self {
                re: (&acc.re).add(&c.re),
                im_i: (&acc.im_i).add(&c.im_i),
                im_j: (&acc.im_j).add(&c.im_j),
                im_k: (&acc.im_k).add(&c.im_k),
            },
        )
    }
}

impl<T: Zero + One + Sub<Output = T>> Product for Quaternion<T>
where
    for<'a> &'a T: Mul<Output = T>,
{
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(
            Self::new(T::one(), T::zero(), T::zero(), T::zero()),
            |acc, c| acc * c,
        )
    }
}
impl<'a, T: Clone + Zero + One + Sub<Output = T>> Product<&'a Quaternion<T>> for Quaternion<T>
where
    for<'b> &'b T: Mul<Output = T>,
{
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Quaternion<T>>,
    {
        iter.fold(
            Self::new(T::one(), T::zero(), T::zero(), T::zero()),
            |acc, c| acc * c.clone(),
        )
    }
}

impl<T: NumAlgebraic<Real = T> + Zero + One> Quaternion<T>
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
    pub fn im_abs(&self) -> T {
        (self.im_i.abs_sqr() + self.im_j.abs_sqr() + self.im_k.abs_sqr()).sqrt()
    }
    pub fn sign_im(&self) -> Self {
        let len = self.im_abs();
        if len.is_zero() {
            return Self {
                re: T::zero(),
                im_i: T::one().copysign(&self.im_i), // decide on one of ijk for a valid solution
                im_j: self.im_j.clone(),
                im_k: self.im_k.clone(),
            }; // keeping zero signs
        }
        Self {
            re: T::zero(),
            im_i: &self.im_i / &len,
            im_j: &self.im_j / &len,
            im_k: &self.im_k / &len,
        }
    }

    fn from_complex(&self, im_abs: T, c: Complex<T>) -> Self {
        debug_assert!(!im_abs.is_zero());
        let i = &c.im / &im_abs;
        // handle infinity correctly using branches
        Self {
            im_i: if self.im_i.is_zero() {
                T::zero()
            } else {
                &i * &self.im_i
            },
            im_j: if self.im_j.is_zero() {
                T::zero()
            } else {
                &i * &self.im_j
            },
            im_k: if self.im_k.is_zero() {
                T::zero()
            } else {
                &i * &self.im_k
            },
            re: c.re,
        }
    }
}

impl<T: NumElementary<Real = T> + Zero + One> Quaternion<T>
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
    /// Convert axis-angle representation of a rotation, where the length of the 3d vector defines the angle, into a unit quaternion.
    pub fn from_axis_angle(v: &[T; 3]) -> Self {
        let len = (v[0].abs_sqr() + v[1].abs_sqr() + v[2].abs_sqr()).sqrt();
        let one = T::one();
        if len.is_zero() {
            return Quaternion::new(one, T::zero(), T::zero(), T::zero());
        }
        let half = &len / &(&one + &one);
        let (c, s) = (half.cos(), half.sin());
        let s = &s / &len;
        Self {
            re: c,
            im_i: &s * &v[0],
            im_j: &s * &v[1],
            im_k: &s * &v[2],
        }
    }
    /// Convert a unit quaternion into an axis-angle representation of a rotation, where the length of the 3d vector defines the angle.
    pub fn to_axis_angle(&self) -> [T; 3] {
        let one = T::one();
        let len = self.im_abs();
        if len.is_zero() {
            return [T::zero(), T::zero(), T::zero()];
        }
        let a = &(&one + &one) * &self.re.acos(); // not always the optimal formula for precision, but otherwise PI is required.
        let s = &a / &len; // more stable than a / sqrt(1 - self.re.abs_sqr())
        [&self.im_i * &s, &self.im_j * &s, &self.im_k * &s]
    }
}

impl<T: NumElementary<Real = T> + AlgebraicField + PartialOrd> NumAlgebraic for Quaternion<T>
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
    #[inline(always)]
    fn abs(&self) -> Self::Real {
        self.abs_sqr().sqrt()
    }
    #[inline(always)]
    fn sign(&self) -> Self {
        self / &self.abs()
    }
    #[inline(always)]
    fn copysign(&self, sign: &Self) -> Self {
        sign.sign() * self.abs()
    }
    fn sqrt(&self) -> Self {
        if self.is_zero() {
            return Self {
                im_i: self.im_i.clone(),
                im_j: self.im_j.clone(),
                im_k: self.im_k.clone(),
                re: T::zero(),
            }; // copy imaginary zero signs
        }
        let im_sqr = self.im_i.abs_sqr() + self.im_j.abs_sqr() + self.im_k.abs_sqr();
        if im_sqr.is_zero() {
            // TODO handle not just zero
            let sqrt = self.re.abs().sqrt();
            //let im = &self.im / &(&(T::one() + T::one()) * &sqrt);
            if self.re >= T::zero() {
                return Self {
                    re: sqrt,
                    ..self.clone()
                };
            } else {
                // this makes it different for 0.0 and -0.0 !!!
                return Self::new(
                    im_sqr.sqrt(),
                    sqrt.copysign(&self.im_i),
                    T::zero(),
                    T::zero(),
                );
            }
        }
        // Use angle bisection
        // Note currently it's limited by the types epsilon at 1.0 for re << 0
        // e.g. in theory (-1.0 + 1e-20 i).sqrt() = (0.5e-20 + i), but rounding will occur here.
        // However (-1.0f64 + 2e-8 i).sqrt() = (1e-8 + i) still works.
        let len = self.abs(); // sqrt eval 1
        let mut half: Quaternion<T> = self.clone() / len.clone();
        half.re = half.re + T::one();
        let fac = (&len / &half.abs_sqr()).sqrt(); // sqrt eval 2
        half * fac
    }
    fn cbrt(&self) -> Self {
        // use polar form
        let im_len = self.im_abs();
        if im_len.is_zero() {
            return Self {
                re: self.re.cbrt(),
                ..self.clone()
            }; // clone the zero signs
        }
        let Complex { re, im } = Complex::new(self.re.clone(), im_len.clone()).cbrt();
        let f = &im / &im_len;
        Self {
            re,
            im_i: &f * &self.im_i,
            im_j: &f * &self.im_j,
            im_k: &f * &self.im_k,
        }
    }
}

macro_rules! forward_function {
    // for functions, which are real on the entire real input range
    ($foo:ident) => {
        fn $foo(&self) -> Self {
            let len = self.im_abs();
            if len.is_zero() {
                return Self {
                    re: self.re.$foo(),
                    im_i: self.im_i.clone(),
                    im_j: self.im_j.clone(),
                    im_k: self.im_k.clone(),
                }; // clone zero signs
            }
            let c = Complex::new(self.re.clone(), len.clone()).$foo();
            self.from_complex(len, c)
        }
    };
    // for functions, which are complex on part of the real input range
    ($foo:ident+) => {
        fn $foo(&self) -> Self {
            let c = Complex::new(self.re.clone(), self.im_abs()).$foo();
            let si = self.sign_im();
            Self {
                im_i: &c.im * &si.im_i,
                im_j: &c.im * &si.im_j,
                im_k: &c.im * &si.im_k,
                re: c.re,
            }
        }
    }
}

// The following elementary functions can also be derived from SU(2) matrices.
// They are the same as for complex numbers, because they always act on a complex orbit.
impl<T: NumElementary + AlgebraicField<Real = T> + PartialOrd> NumElementary for Quaternion<T>
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
    forward_function!(sin);
    forward_function!(cos);
    forward_function!(tan);
    forward_function!(asin+);
    forward_function!(acos+);
    forward_function!(atan);
    forward_function!(sinh);
    forward_function!(cosh);
    forward_function!(tanh);
    forward_function!(asinh);
    forward_function!(acosh+);
    forward_function!(atanh+);

    fn atan2(&self, x: &Self) -> Self {
        let len = self.im_abs();
        let lenx = x.im_abs();
        if lenx.is_zero() {
            if len.is_zero() {
                return Self {
                    re: self.re.atan2(&x.re),
                    im_i: T::zero(),
                    im_j: T::zero(),
                    im_k: T::zero(),
                };
            }
            let c = Complex::new(self.re.clone(), len.clone())
                .atan2(&Complex::new(x.re.clone(), T::zero()));
            return self.from_complex(len, c);
        }
        if len.is_zero() {
            let c = Complex::new(self.re.clone(), T::zero())
                .atan2(&Complex::new(x.re.clone(), lenx.clone()));
            return self.from_complex(lenx, c);
        }
        // to get this without conditionals, implement it as
        // formula: 2*arctan(y / (x + (x*x + y*y).sqrt()))
        let div = x + &(x * x + self * self).sqrt();
        let res = if div.is_zero() {
            // directly at the branchcut, decide for one side.
            // TODO what happens here for complex values???
            T::zero().acos().copysign(&(self * &div.conj()).re()).into()
        } else {
            (self / &div).atan()
        };
        &res + &res
    }

    fn exp(&self) -> Self {
        // core rotation formula, most easily derived using Pauli matrices (or ijk) in
        // the power series of the exponential function.
        let r = self.re.exp();
        if r.is_zero() {
            return Self::zero();
        }
        let len = self.im_abs();
        if len.is_zero() {
            return Self {
                re: r,
                im_i: T::zero(),
                im_j: T::zero(),
                im_k: T::zero(),
            };
        }
        let (c, s) = (len.cos(), len.sin());
        let rs = &(&r * &s) / &len;
        Self {
            re: &r * &c,
            im_i: &rs * &self.im_i,
            im_j: &rs * &self.im_j,
            im_k: &rs * &self.im_k,
        }
    }

    fn exp_m1(&self) -> Self {
        let r = self.re.exp_m1();
        let two = T::one() + T::one();
        let len = self.im_abs();
        // (r + 1) * cos - 1 = r * cos + (cos - 1) // lossy at im~0
        // cos - 1 = -2sin(im/2)^2 // precise at im~0
        let (s2, s) = ((&len / &two).sin(), len.sin());
        let s2 = &s2 * &s2 * two;
        let rs = &(&r * &s) / &len;
        Self {
            re: &r * &(&T::one() - &s2) - s2,
            im_i: &rs * &self.im_i,
            im_j: &rs * &self.im_j,
            im_k: &rs * &self.im_k,
        }
    }

    fn ln(&self) -> Self {
        let re = self.abs();
        let q = self / &re;
        let len = q.im_abs();
        if len.is_zero() {
            let im_i = if self.re >= T::zero() {
                T::zero()
            } else {
                (T::zero().acos() * (T::one() + T::one())).copysign(&self.im_i)
            };
            return Self {
                re: re.ln(),
                im_i,
                im_j: T::zero(),
                im_k: T::zero(),
            };
        }
        let s = &q.re.acos() / &len; // more stable than a / sqrt(1 - self.re.abs_sqr())
        Self {
            re: re.ln(),
            im_i: &q.im_i * &s,
            im_j: &q.im_j * &s,
            im_k: &q.im_k * &s,
        }
    }

    #[inline(always)]
    fn ln_1p(&self) -> Self {
        // no good way to write this without conditionals...
        (self + &T::one()).ln()
    }

    /// Raises `self` to a quaternion power.
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

/// Write a quaternion in the notation `real + i x + j y + k z`. Any term can be left out, however the order needs to be kept.
#[macro_export]
macro_rules! quaternion {
    ($re:literal $(+ i $i:literal)? $(+ i ($ie:expr))? $(+ j $j:literal)? $(+ j ($je:expr))? $(+ k $k:literal)? $(+ k ($ke:expr))?) => {
        {
            #[allow(unused_mut)]
            let mut q = $crate::quaternion::Quaternion::from($re);
            $(q.im_i = q.im_i + $i;)?
            $(q.im_i = q.im_i + ($ie);)?
            $(q.im_j = q.im_j + $j;)?
            $(q.im_j = q.im_j + ($je);)?
            $(q.im_k = q.im_k + $k;)?
            $(q.im_k = q.im_k + ($ke);)?
            q
        }
    };
    ($x:expr) => {
        $x.into()
    };
}

#[inline(never)]
fn fmt_quat(
    f: &mut fmt::Formatter<'_>,
    re_neg: bool,
    real: fmt::Arguments<'_>,
    imag_i: fmt::Arguments<'_>,
    imag_j: fmt::Arguments<'_>,
    imag_k: fmt::Arguments<'_>,
    _prefix: &str,
) -> fmt::Result {
    let sign = if re_neg {
        ""
    } else if f.sign_plus() {
        "+"
    } else {
        ""
    };

    fmt_complex(
        f,
        format_args!(
            "{}{re}{i}i{j}j{k}k",
            sign,
            re = real,
            i = imag_i,
            j = imag_j,
            k = imag_k
        ),
    )
}

// string conversions
macro_rules! impl_display {
    ($Display: ident, $s: literal, $pre: literal) => {
        impl<T> fmt::$Display for Quaternion<T>
        where
            T: fmt::$Display + Clone + Zero + PartialOrd + Sub<T, Output = T>,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let prefix = $pre;
                return if let Some(prec) = f.precision() {
                    if f.alternate() {
                        fmt_quat(
                            f,
                            self.re < T::zero(),
                            format_args!(concat!("{:#.1$", $s, "}"), self.re, prec),
                            format_args!(concat!("{:+#.1$", $s, "}"), self.im_i, prec),
                            format_args!(concat!("{:+#.1$", $s, "}"), self.im_j, prec),
                            format_args!(concat!("{:+#.1$", $s, "}"), self.im_k, prec),
                            prefix,
                        )
                    } else {
                        fmt_quat(
                            f,
                            self.re < T::zero(),
                            format_args!(concat!("{:.1$", $s, "}"), self.re, prec),
                            format_args!(concat!("{:+.1$", $s, "}"), self.im_i, prec),
                            format_args!(concat!("{:+.1$", $s, "}"), self.im_j, prec),
                            format_args!(concat!("{:+.1$", $s, "}"), self.im_k, prec),
                            prefix,
                        )
                    }
                } else {
                    if f.alternate() {
                        fmt_quat(
                            f,
                            self.re < T::zero(),
                            format_args!(concat!("{:#", $s, "}"), self.re),
                            format_args!(concat!("{:+#", $s, "}"), self.im_i),
                            format_args!(concat!("{:+#", $s, "}"), self.im_j),
                            format_args!(concat!("{:+#", $s, "}"), self.im_k),
                            prefix,
                        )
                    } else {
                        fmt_quat(
                            f,
                            self.re < T::zero(),
                            format_args!(concat!("{:", $s, "}"), self.re),
                            format_args!(concat!("{:+", $s, "}"), self.im_i),
                            format_args!(concat!("{:+", $s, "}"), self.im_j),
                            format_args!(concat!("{:+", $s, "}"), self.im_k),
                            prefix,
                        )
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
impl<T: serde::Serialize> serde::Serialize for Quaternion<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (&self.im_i, &self.im_j, &self.im_k, &self.re).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for Quaternion<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (im_i, im_j, im_k, re) = serde::Deserialize::deserialize(deserializer)?;
        Ok(Self::new(re, im_i, im_j, im_k))
    }
}
