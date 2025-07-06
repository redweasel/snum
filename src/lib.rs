//! This crate is a replacement for the popular `num` ecosystem.
//! It allows much more (almost maximal) flexibility with the defined algebraic structures.
//! Due to it's more general type handling, this crate doesn't add much weight, even though it
//! contains both complex numbers and rational numbers. In particular there is only a few places,
//! where things had to be defined for all available integer types and the implementations are trivially small.
//! There is no dependence on [Copy] anywhere and nothing is limited to buildin types.
//! 
//! A central objective of this crate is to only have traits which are essential, or very useful.
//! - Essential traits are e.g. [Zero], [One], [Num], [NumAlgebraic], [NumAnalytic] and [RemEuclid]. (these need to be implemented)
//! - Grouping/alias traits are [Ring], [Field], [AlgebraicField] and [AddMulSubDiv] (+ lesser variants).
//! - Derived traits are [Cancel], [Power], [IntMul].
//! 
//! All operator implementations, which mix references and owned structs are considered bloat, as the
//! real world performance benefit hasn't been demonstrated. Note that any type with expensive clone
//! could just internally use `Arc` to make it cheap again. Usually one can already write equations
//! optimal with non-mixed operations. Moreover, no crates (should) depend on having the mixed operators.
//! Similarly the assign operators like `AssignAdd` are implemented based on the reference `Add` operation.
//! This is done without cloning thanks to [take_mut].
//! 
//! The resulting [complex::Complex] and [rational::Ratio] types are slightly different in some places, but largely compatible with `num`.
//! E.g. The multiplicative inverse on these commutative fields is called `recip` instead of `inv`, as some type might be an invertible
//! (e.g. Moebius transform) function, which needs `inv` for inversion wrt composition, but can still have `recip`.
//! 
//! # Testing Status
//! Some code has been copied from `num_complex` and `num_rational`. In particular the test code has been copied where applicable.
//! In this process, bugs in the testing code have been found.
//! 
//! # Limitations
//! To make other crates, which are not considered in this one, work, one needs to implement `Num` for them.
//! This can only be done in said crate, or by creating a wrapper and forwarding all functionallity.
//! 
//! For complex numbers, binary operations lke `T + Complex<T>` are not implemented,
//! as that would require an implementation for every specific type (bloat for nothing).
//! 
//! Overflows are well avoided in [rational], but no checked functions are implemented.
//! Other places, like [complex] are prone to integer overflow. So get the checked or wrapping variants,
//! use custom wrappers on the int types. E.g. `enum Checked<T> { Value(T), Overflow }`.
//! 
//! ### TODOs
//! - As an improvement, implement a `Gaussian` type for integral complex numbers, which uses canceling to avoid overflows.
//! - Add a macro, which, based on Deref, forwards all arithmetic operations of a wrapper type automatically.
//! - Hide approximation from floats for rational and sqrt types behind a feature flag.
//! - Add string parsing for complex and rational types (and hide it behind a feature flag to avoid bloat)

#![no_std]

#[cfg(any(test, feature = "std"))]
extern crate std;

mod num;
pub mod complex;
pub mod rational;
pub mod extension;
mod from;
mod float;
mod power;
mod continued_fractions;

pub use num::*;
pub use from::*;
pub use float::*;
pub use power::*;
pub use continued_fractions::*;

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
