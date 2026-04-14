//! Very basic root-finding algorithms

use crate::*;

/// Bisection is a root finding algorithm for monotone functions,
/// which finds a root in a fixed interval [a, b]. (`a` and `b` can be swapped)
/// 
/// If the function values f(a) and f(b) have differing signs (or are zero),
/// this procedure always succeeds, otherwise it returns the closest to zero
/// boundary as `Err(x)`.
/// 
/// The number of iterations can be limited. To use no limit
/// (limit at machine precision), specify 0. This may lead to
/// infinite loops if e.g. one of the bounds is non finite.
/// This method also works for integers, even though they are
/// not a true [Field].
/// 
/// For bisecting a sorted slice, see `binary_search`, which is implemented for slices.
pub fn bisect<T: Field + RealNum>(f: impl Fn(T) -> T, mut a: T, mut b: T, max_iter: usize) -> Result<T, T>
where for<'a> &'a T: AddMulSubDiv<Output = T> {
    if a != a || b != b {
        return Err(a + b); // NaN values detected
    }
    let mut fa = f(a.clone());
    if a == b {
        return if fa.is_zero() { Ok(a) } else { Err(a) };
    }
    let mut fb = f(b.clone());
    let sign = fa.clone();
    if &sign * &fb > T::zero() || sign != sign {
        return Err(if (fa > fb) == (fa > T::zero()) { b } else { a });
    }
    if fa.is_zero() {
        return Ok(a);
    }
    if fb.is_zero() {
        return Ok(b);
    }
    let two = T::one() + T::one();
    let mut i = 0;
    while max_iter == 0 || i < max_iter {
        let x = &(&a + &b) / &two; // also works for integers!
        if x == a || x == b {
            return Ok(x);
        }
        let f_val = f(x.clone());
        let val = &f_val * &sign;
        if val > T::zero() {
            fa = f_val;
            a = x;
        }
        else if val < T::zero() {
            fb = f_val;
            b = x;
        }
        else {
            return Ok(x);
        }
        i += 1;
    }
    Ok(&(&(&fa * &(&b - &a)) / &(&fa - &fb)) + &a)
}

/// "Trisection" is a root finding algorithm for analytic monotone or strictly monotone functions,
/// which finds a root in a fixed interval [a, b]. (analytic on the open interval)
/// It doesn't actually trisect intervals, it's rather a bisection with where the previous
/// two intervals are kept and then the correct one is bisected (-> 3 intervals -> "trisection").
/// This method is significantly faster than bisection, but similarly stable.
/// 
/// If the function values f(a) and f(b) have differing signs (or are zero),
/// this procedure always succeeds, otherwise it returns the closest to zero
/// boundary as `Err(x)`.
/// 
/// The number of iterations can be limited. To use no limit
/// (limit at machine precision), specify 0. This may lead to
/// infinite loops if e.g. one of the bounds is non finite.
pub fn trisect<T: Field + RealNum>(f: impl Fn(T) -> T, mut a: T, mut b: T, max_iter: usize) -> Result<T, T>
where for<'a> &'a T: AddMulSubDiv<Output = T> {
    if a != a || b != b {
        return Err(a + b); // NaN values detected
    }
    let mut fa = f(a.clone());
    if a == b {
        return if fa.is_zero() { Ok(a) } else { Err(a) };
    }
    assert!(a < b);
    let mut fb = f(b.clone());
    let sign = fa.clone();
    if &sign * &fb > T::zero() || sign != sign {
        return Err(if (fa > fb) == (fa > T::zero()) { b } else { a });
    }
    if fa.is_zero() {
        return Ok(a);
    }
    if fb.is_zero() {
        return Ok(b);
    }
    let two = T::one() + T::one();
    let mut p = &(&a + &b) / &two;
    let mut fp = f(p.clone());
    let mut i = 0;
    loop {
        // compute x from Moebius transform of 3 points (similar to Halley's method)
        // Interpolation formula from https://math.stackexchange.com/a/3807668
        let z0 = (&a - &p) * (&fp * &(&fb - &fa));
        let z1 = (&a - &b) * (&(&fp - &fa) * &fb);
        let mut x = (&z0 * &b - &z1 * &p) / (z0 - z1);
        if x < a {
            x = a.clone();
        }
        else if x > b {
            x = b.clone();
        }
        if max_iter != 0 && i >= max_iter {
            return Ok(x);
        }
        if x == a || x == b || x == p {
            return Ok(x);
        }
        let f_val = f(x.clone());
        let val = &fp * &sign;
        if val > T::zero() {
            if x <= p {
                return Ok(p);
            }
            fa = fp;
            a = p;
            fp = f_val;
            p = x;
        }
        else if val < T::zero() {
            if x >= p {
                return Ok(p);
            }
            fb = fp;
            b = p;
            fp = f_val;
            p = x;
        }
        else {
            return Ok(x);
        }
        i += 1;
    }
}