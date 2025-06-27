//! This crate is a replacement for the popular `num_traits` ecosystem.
//! It allows much more (almost maximal) flexibility with the defined algebraic structures.
//! Due to it's more general type handling, this crate doesn't add much weight, even though it
//! contains both complex numbers and rational numbers.
//! 
//! Overflows are only handled well in [rational]. Other places, like [complex] are prone to integer overflow.
//! TODO As an improvement, implement a `Gaussian` type for integral complex numbers, which handles overflows well.

#![no_std]

#[cfg(any(test, feature = "std"))]
extern crate std;

mod num;
pub mod complex;
pub mod rational;
pub mod float;
mod from;
mod power;

pub use num::*;
pub use from::*;
pub use power::*;

// TODO look at nalgebra's ClosedAdd, ClosedMul,... and implement that here as well, as it will allow += to be used sometimes (performance?)
// TODO implement "mint" as an optional dependency
// it's also much less confusing. Rem could then do the elementwise multiplication, but better to just put that into a function.

#[cfg(test)]
mod tests;
