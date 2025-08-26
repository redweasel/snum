//! This crate is a replacement for the popular `num` ecosystem.
//! It allows much more (almost maximal) flexibility with the defined algebraic structures.
//! Due to it's more general type handling, this crate doesn't add much weight, even though it
//! contains both complex numbers and rational numbers. In particular there is only a few places,
//! where things had to be defined for all available integer types and the implementations are trivially small.
//! There is no dependence on [Copy] anywhere and nothing is limited to buildin types.
//! 
//! A central objective of this crate is to only have traits which are essential, or very useful.
//! - Essential traits are e.g. [Zero], [One], [Num], [NumAlgebraic], [NumAnalytic] and [Euclid]. (these need to be implemented)
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
//! - `rational` for the [Ratio] and [SqrtExt] types.
//! - `bytemuck`
//! - `num-bigint` to include trait implementations for it
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
//! Be careful.
//! 
//! # Limitations
//! To make other crates, which are not considered in this one, work, one needs to implement `Num` for them.
//! This can only be done in said crate, or by creating a wrapper and forwarding all functionallity.
//! 
//! For complex numbers, binary operations lke `T + Complex<T>` are not implemented,
//! as that would require an implementation for every specific type (bloat for nothing).
//! 
//! Overflows are well avoided in [rational], but no checked functions are implemented.
//! Other places, like [mod@complex] and [extension] are prone to integer overflow. So get the checked or wrapping variants,
//! use custom wrappers on the int types. E.g. `enum Checked<T> { Value(T), Overflow }`.
//! 
//! ### TODOs
//! - As an improvement, implement a `Gaussian` type for integral complex numbers, which uses canceling to avoid overflows.
//! - `Zero`, `Conjugate` and `Euclid` should have derive macros just like `Clone`, currently there is [impl_zero_default!], [impl_conjugate_real!] and [impl_euclid_field!].
//! - Add a macro, which, based on Deref, forwards all arithmetic operations of a wrapper type automatically.
//! - Hide approximation from floats for rational and sqrt types behind a feature flag.
//! - Add string parsing for complex and rational types (and hide it behind a feature flag to avoid bloat)
//! - It appears, that the reference implementation need to use the non reference implementations to avoid recursive trait evaluations. Check this again! Otherwise switch to using the optimal operations.
//! - Rename NumAnalytic to NumElementary, as that is more fitting

#![no_std]

#[cfg(feature = "std")]
extern crate std;

mod num;
mod from;
mod power;
pub mod complex;
mod float;
#[cfg(feature = "quaternion")]
pub mod quaternion;

pub use num::*;
pub use from::*;
pub use power::*;
pub use float::*;

#[cfg(feature = "rational")]
pub mod rational;
#[cfg(feature = "rational")]
pub mod extension;
#[cfg(feature = "rational")]
mod continued_fractions;

#[cfg(feature = "rational")]
pub use continued_fractions::*;

// global imports for docs and tests
#[cfg(feature = "rational")]
#[allow(unused_imports)] // they are for the docs
use self::{rational::*, extension::*};
#[allow(unused_imports)] // they are for the docs
use self::{complex::*};

#[cfg(test)]
mod tests;

// More ideas for useful things:
/*
/// trait for numbers, where the N-th power can be represented using terms of order N-1 and less.
/// E.g. `i^2 = -1` or `sqrt(2)^2 = 2`. More complicated examples are the factor-ring `R[x]/(x^3+1-x)`,
/// where `x^3=x-1`. To use this, create a custom type and implement this trait on it.
pub trait Factor<T, const N: usize> {
    fn factor(x: &T) -> [T; N];
}

/// trait to compute the squared part of a number.
pub trait SquareFree: Sized {
    /// decomposes `self` into a square free part `x` and the rest, such that `self=x*y^2`.
    /// Returns (x, y)
    fn square_free_factorisation(&self) -> (Self, Self);
}

// mostly useful for number theory.
pub trait PartialSqrt: Sized {
    /// Computes the square root if possible.
    fn partial_sqrt(&self) -> Option<Self>;
}

// type for a*b^N with fixed N? similar to fractions!
// type for mobius transforms (mostly number theory and complex analysis -> into the polynomial package?)
// type for polynomial roots: `AlgebraicNum` (addition, multiplication and division are again roots of polynomials -> polynomial package)
*/
