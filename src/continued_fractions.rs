//! Traits for evaluating the continued fraction with iterators

use core::borrow::Borrow;

use crate::{Cancel, IntoDiscrete, rational::Ratio};

// forward evaluation of continued fractions is a bit harder than backwards evaluation, as one has to use Möbius transformations.
fn start<T: Cancel>() -> (T, T, T, T) {
    (T::one(), T::zero(), T::zero(), T::one())
}
fn next<T: Cancel>(accum: (T, T, T, T), x: &T) -> (T, T, T, T) {
    let (a, b, c, d) = accum;
    (x.clone() * a.clone() + b, a, x.clone() * c.clone() + d, c)
}
fn end<T: Cancel>(accum: (T, T, T, T), end: T) -> Ratio<T> {
    let (a, b, c, d) = accum;
    // uncancelled results. Usually they are already cancelled by construction
    Ratio::new_raw(a + b * end.clone(), c + d * end)
}

/// Trait extension to iterators to allow evaluating them as continued fractions.
/// This is done forward, so `[1, 2, 2]` turns into `[1+end, 1+1/(2+end), 1+1/(2+1/(2+end))]`
/// ```rust
/// use snum::*;
/// use snum::rational::*;
/// 
/// let end = 2;
/// let mut iter = [1, 2, 2].iter().continued_fraction(end);
/// let _1 = Ratio::new(1, 1);
/// assert_eq!(Some(_1 + end), iter.next());
/// assert_eq!(Some(_1 + _1/(_1*2 + end)), iter.next());
/// assert_eq!(Some(_1 + _1/(_1*2 + _1/(_1*2 + end))), iter.next());
/// assert_eq!(None, iter.next());
/// ```
pub struct ContinuedFractionIter<T: Cancel, I: Iterator> {
    iter: I,
    accum: (T, T, T, T),
    end: T,
}
impl<T: Cancel, J: Borrow<T>, I: Iterator<Item = J>> Iterator for ContinuedFractionIter<T, I> {
    type Item = Ratio<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|x| {
            take_mut::take(&mut self.accum, |accum| next(accum, &x.borrow()));
            end(self.accum.clone(), self.end.clone())
        })
    }
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if let Some(max_len) = self.iter.size_hint().1 {
            if n > max_len {
                return None;
            }
        }
        // doing multiple steps at once, makes the computation of the outputs in between unecessary.
        for _ in 0..=n {
            if let Some(x) = self.iter.next() {
                take_mut::take(&mut self.accum, |accum| next(accum, &x.borrow()));
            } else {
                return None;
            }
        }
        Some(end(self.accum.clone(), self.end.clone()))
    }
    fn last(mut self) -> Option<Self::Item> {
        // in this case one can also backwards evaluated if the iterator is a DoubleEndedIterator, however that required specialisaton.
        // -> compute forwards until the end, don't compute intermediate outputs
        while let Some(x) = self.iter.next() {
            take_mut::take(&mut self.accum, |accum| next(accum, &x.borrow()));
        }
        Some(end(self.accum, self.end))
    }
}

/// Trait extension to iterators to allow evaluating them as continued fractions.
pub trait IntoContinuedFraction<T: Sized> {
    type IntoIter: Iterator;
    /// Takes an iterator and generates `Self` from the elements by evaluating the continued fraction.
    /// This is done forward, so `[1, 2, 2]` turns into `[1+1/1, 1+1/(2+1/1), 1+1/(2+1/(2+1/1))]`
    fn continued_fraction(self, end: T) -> Self::IntoIter;
}
impl<'a, T: 'a + Cancel, J: Borrow<T>, I: Iterator<Item = J>> IntoContinuedFraction<T> for I {
    type IntoIter = ContinuedFractionIter<T, Self>;
    fn continued_fraction(self, end: T) -> ContinuedFractionIter<T, Self> {
        ContinuedFractionIter {
            iter: self,
            accum: start(),
            end,
        }
    }
}

/// Iterator for a simple/regular continued fraction.
pub struct DevelopContinuedFraction<T> {
    // remaining value
    numer: T,
    denom: T,
}

// same trait bounds as for iterator to simplify error messages
impl<T: Cancel + IntoDiscrete> DevelopContinuedFraction<T> {
    pub fn new(value: T) -> Self {
        Self {
            numer: value,
            denom: T::one(),
        }
    }
}

impl<T: Cancel + IntoDiscrete> Iterator for DevelopContinuedFraction<T> {
    type Item = <T as IntoDiscrete>::Output;
    fn next(&mut self) -> Option<Self::Item> {
        if self.numer.is_zero() {
            // finished
            return None;
        }
        let i = self.numer.div_rem_euclid(&self.denom).0.floor();
        (self.numer, self.denom) = {
            let r = self.numer.clone() - T::from(i.clone()) * self.denom.clone();
            if r.is_zero() {
                (T::zero(), T::zero()) // end
            } else {
                // cancel is needed here!
                self.denom.clone().cancel(r)
            }
        };
        Some(i)
    }
}
