//! Traits for evaluating the continued fraction with iterators

use core::borrow::Borrow;

use crate::{rational::Ratio, Cancel, IntoDiscrete};

trait ContinuedFraction: Sized {
    type Output: Sized;
    fn start() -> (Self, Self, Self, Self);
    fn next(accum: (Self, Self, Self, Self), value: &Self) -> (Self, Self, Self, Self);
    fn end(accum: (Self, Self, Self, Self), end: Self) -> Self::Output;
}

// forward evaluation of continued fractions is a bit harder than backwards evaluation, as one has to use Möbius transformations.
impl<T: Cancel> ContinuedFraction for T {
    type Output = Ratio<Self>;
    fn start() -> (Self, Self, Self, Self) {
        (Self::one(), Self::zero(), Self::zero(), Self::one())
    }
    fn next(accum: (Self, Self, Self, Self), x: &Self) -> (Self, Self, Self, Self) {
        let (a, b, c, d) = accum;
        (x.clone() * a.clone() + b, a, x.clone() * c.clone() + d, c)
    }
    fn end(accum: (Self, Self, Self, Self), end: T) -> Self::Output {
        let (a, b, c, d) = accum;
        // uncancelled results. Usually they are already cancelled by construction
        Ratio::new_raw(a * end.clone() + b, c * end + d)
    }
}

/// Trait extension to iterators to allow evaluating them as continued fractions.
/// This is done forward, so `[1, 2, 2]` turns into `[1+1/1, 1+1/(2+1/1), 1+1/(2+1/(2+1/1))]`
pub struct ContinuedFractionIter<T: Cancel, I: Iterator> {
    iter: I,
    accum: (T, T, T, T),
    end: T,
}
impl<T: Cancel, J: Borrow<T>, I: Iterator<Item = J>> Iterator
    for ContinuedFractionIter<T, I>
{
    type Item = Ratio<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|x| {
            take_mut::take(&mut self.accum, |accum| T::next(accum, &x.borrow()));
            T::end(self.accum.clone(), self.end.clone())
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
                take_mut::take(&mut self.accum, |accum| T::next(accum, &x.borrow()));
            } else {
                return None;
            }
        }
        Some(T::end(self.accum.clone(), self.end.clone()))
    }
    fn last(mut self) -> Option<Self::Item> {
        // in this case one can also backwards evaluated if the iterator is a DoubleEndedIterator, however that required specialisaton.
        // -> compute forwards until the end, don't compute intermediate outputs
        while let Some(x) = self.iter.next() {
            take_mut::take(&mut self.accum, |accum| T::next(accum, &x.borrow()));
        }
        Some(T::end(self.accum, self.end))
    }
}

/// Trait extension to iterators to allow evaluating them as continued fractions.
pub trait IntoContinuedFraction<T: Sized> {
    type IntoIter: Iterator;
    /// Takes an iterator and generates `Self` from the elements by evaluating the continued fraction.
    /// This is done forward, so `[1, 2, 2]` turns into `[1+1/1, 1+1/(2+1/1), 1+1/(2+1/(2+1/1))]`
    fn continued_fraction(self, end: T) -> Self::IntoIter;
}
impl<'a, T: 'a + Cancel, J: Borrow<T>, I: Iterator<Item = J>> IntoContinuedFraction<T>
    for I
{
    type IntoIter = ContinuedFractionIter<T, Self>;
    fn continued_fraction(self, end: T) -> ContinuedFractionIter<T, Self> {
        ContinuedFractionIter {
            iter: self,
            accum: T::start(),
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
