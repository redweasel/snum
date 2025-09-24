//! This crate is a replacement for the popular `num` ecosystem.
//! It allows much more (almost maximal) flexibility with the defined algebraic structures.
//! Due to it's more general type handling, this crate doesn't add much weight, even though it
//! contains both [mod@complex] numbers and [rational] numbers. In particular there is only a few places,
//! where things had to be defined for all available integer types and the implementations are trivially small.
//! There is no dependence on [Copy] anywhere and nothing is limited to buildin types.
//!
//! A central objective of this crate is to only have traits which are essential, or very useful.
//! - Essential traits are e.g. [Zero], [One], [Num], [NumAlgebraic], [NumElementary] and [Euclid]. (these need to be implemented)
//! - Grouping/alias traits are [Ring], [Field], [AlgebraicField] and [AddMulSubDiv] (+ lesser variants).
//! - Derived traits are [Cancel], [SafeDiv], [PowerU], [PowerI], [IntMul].
//!
//! After implementing ring and field traits, one might ask for a commutative marker trait for multiplication and addition,
//! however that is explicitly not implemented, as it's not essential to a functioning type system in Rust. Algorithms
//! should tell in their description if they work for commutative types only, if not obvious.
//!
//! All operator implementations, which mix references and owned structs are considered bloat, as the
//! real world performance benefit hasn't been demonstrated. Note that any type with expensive clone
//! could just internally use `Arc` or `Cow` to make it cheap again. Usually one can already write equations
//! optimal with non-mixed operations. Moreover, no crates (should) depend on having the mixed operators.
//! Similarly the assign operators like `AssignAdd` are implemented based on the reference `Add` operation.
//! This is done without cloning thanks to [take_mut].
//!
//! Whenever deciding between precision and performance, the question of *what is required more frequently*
//! and *what can be implemented outside of this library* is asked. E.g. the [Ratio] type does expensive
//! canceling to get the most range out of its integers, whereas [Complex] doesn't do that, as it's usually
//! used with floats.
//!
//! The resulting [Complex] and [Ratio] types are slightly different in some places, but largely compatible with `num`.
//! E.g. The multiplicative inverse on these commutative fields is called `recip` instead of `inv`, as some type might be an invertible
//! function (e.g. Moebius transform), which needs `inv` for inversion wrt composition, but can still have `recip`.
//!
//! Float approximations of numbers are implemented using the trait [ApproxFloat]. They can be chained through types.
//! E.g. a [Ratio] with [SqrtExt] as numerator and denominator can directly be converted to a float.
//!
//! # Features
//!
//! - `std` (default, however not required)
//! - `libm` as a replacement for `std` when using floats.
//! - `quaternion` for the [Quaternion] type.
//! - `rational` for the [Ratio] and [SqrtExt] types.
//! - `rand` for uniform, normal and unitary random distributions for complex types.
//! - `bytemuck`
//! - `ibig` to include trait implementations for `ibig`
//! - `serde`
//!
//! # Testing Status
//! The tests from `num_complex` and `num_rational` are copied where applicable.
//! In this process, bugs in their testing code have been found. The improved testing code
//! is no longer fully succeeding for `num_complex` and `num_rational`.
//!
//! Note, that it is impossible to test all combinations, which are allowed in this crate.
//! There is many cases in the `rational` part, where the gcd doesn't converge (infinite loop),
//! because e.g. it is evaluated on a domain which is not Euclidean. Some functions have relaxed
//! trait bounds to the point, where they can be called on types that don't work.
//! E.g. [SqrtExt] in large part only works for types with commutative multiplication.
//! Be careful and respect math.
//!
//! # Limitations
//! To make other crates, which are not considered in this one, work, one needs to implement `Num` for them.
//! This can only be done in said crate, or by creating a wrapper and forwarding all functionallity.
//!
//! For complex numbers, binary operations like `T + Complex<T>` are not implemented,
//! as that would require an implementation for every specific type (bloat for nothing).
//!
//! Overflows are well avoided in [rational], but no checked functions are implemented.
//! Other places, like [mod@complex] and [extension] are prone to integer overflow.
//! Use wrappers on the int types, e.g. `Wrapping<T>` or `Saturating<T>` or `enum Checked<T> { Value(T), Overflow }` to manage the overflows.
//!
//! The by-reference implementations of math operations need to use the non reference implementations
//! of the inner type to avoid recursive trait evaluations by SIMD types. This introduces a significant
//! amount of clones. Clones are kept minimal in most cases. If clones slow down the calculation, consider
//! wrapping your types in `Rc`.
//! 
//! ### TODOs
//! - `Zero`, `Conjugate` and `Euclid` should have derive macros just like `Clone`, currently there is [impl_zero_default!], [impl_conjugate_real!], [impl_euclid_field!] and [impl_num_wrapper!].
//! - Add string parsing for complex and rational types (and hide it behind a feature flag to avoid bloat)
//! - As an improvement, implement a `Gaussian` type for integral complex numbers, which uses canceling to avoid overflows.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

mod complex;
mod float;
mod fmt;
mod from;
mod num;
mod power;
#[cfg(feature = "quaternion")]
pub mod quaternion;
#[cfg(feature = "rand")]
pub mod rand;

pub use complex::*;
pub use float::*;
pub use from::*;
pub use num::*;
pub use power::*;

#[cfg(feature = "rational")]
mod continued_fractions;
#[cfg(feature = "rational")]
pub mod extension;
#[cfg(feature = "rational")]
pub mod rational;

#[cfg(feature = "rational")]
pub use continued_fractions::*;

// global imports for docs and tests
#[cfg(feature = "quaternion")]
#[allow(unused_imports)] // they are for the docs
use self::quaternion::*;
#[cfg(feature = "rational")]
#[allow(unused_imports)] // they are for the docs
use self::{extension::*, rational::*};

#[cfg(test)]
mod tests;

macro_rules! forward_assign_impl {
    ($type:ident;$($AddAssign:ident, ($Add:ident$(,$Add3:ident)*), ($($Add2:ident),*), $(($One:ident),)? $({$Cancel:ident},)? $([$Mul:ident],)? $add_assign:ident, $add:ident;)+) => {
        $(impl<T: Clone $(+ $Cancel)? $(+ $One)? $(+ $Add2<Output = T>)*> $AddAssign for $type<T>
            where for<'a> &'a T: $Add<Output = T> $(+ $Add3<Output = T>)* $(+ $Mul<Output = T>)? {
            fn $add_assign(&mut self, rhs: $type<T>) {
                take(self, |x| x.$add(rhs));
            }
        }
        impl<T: Clone $(+ $Cancel)? + $Add<Output = T> $(+ $Add3<Output = T>)* $(+ $Add2<Output = T>)*> $AddAssign<T> for $type<T> {
            fn $add_assign(&mut self, rhs: T) {
                take(self, |x| x.$add(rhs));
            }
        }
        impl<'a, T: Clone $(+ $Cancel)? $(+ $One)? + $Add<Output = T> $(+ $Add3<Output = T>)* $(+ $Add2<Output = T>)*> $AddAssign<&'a $type<T>> for $type<T>
        $(where for<'b> &'b T: $Mul<Output = T>)? {
            fn $add_assign(&mut self, rhs: &'a $type<T>) {
                take(self, |x| (&x).$add(rhs));
            }
        }
        impl<'a, T: Clone $(+ $Cancel)? $(+ $Add2<Output = T>)*> $AddAssign<&'a T> for $type<T>
        where for<'b> &'b T: $Add<Output = T> $(+ $Add3<Output = T>)* {
            fn $add_assign(&mut self, rhs: &'a T) {
                take(self, |x| (&x).$add(rhs));
            }
        })+
    };
}
pub(crate) use forward_assign_impl;
