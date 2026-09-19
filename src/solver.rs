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
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
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
        } else if val < T::zero() {
            fb = f_val;
            b = x;
        } else {
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
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
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
        } else if x > b {
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
        } else if val < T::zero() {
            if x >= p {
                return Ok(p);
            }
            fb = fp;
            b = p;
            fp = f_val;
            p = x;
        } else {
            return Ok(x);
        }
        i += 1;
    }
}

/// Use Golden-section search to find a minimum.
/// It will always converge to a minimum, even if there is multiple or the function is not continuous.
/// There is no guarantee that it finds the global minimum.
///
/// If the type `T` is complex, this method will do a line search from `a` to `b`.
///
/// See: https://en.wikipedia.org/wiki/Golden-section_search
pub fn minimize_golden<T: Num + Zero + One, Q: PartialOrd>(f: impl Fn(&T) -> Q, mut a: T, mut b: T, max_iter: usize) -> Result<T, T>
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
    // The field must support x * 5 / 8 for at least one element.
    assert!(
        T::CHAR == 0 || T::CHAR > 8,
        "The finite field with characteristic {} is too small",
        T::CHAR
    );
    if a != a || b != b {
        return Err(a + b); // NaN values detected
    }
    let mut fa = f(&a);
    if a == b {
        return Ok(a);
    }
    let mut fb = f(&b);
    let two = T::one() + T::one();
    let eight = &(&two + &two) * &two;
    let three = &two + &T::one();
    let five = &eight - &three;
    // To allow more general number types, the golden ratio is approximated. This should not affect the convergence too much.
    // The choosen approximation is 5/3 = 1.666, as using 8=2³ as divisor seems like a good idea for speed and it's close enough.
    let mut p = &(&five * &a + &three * &b) / &eight;
    let mut fp = f(&p);
    for _ in 0..max_iter {
        // conditions for the start until a \/ shape is found
        if fa <= fp && fa < fb {
            // unbalanced / or /\, do a biased step.
            fb = fp;
            (p, b) = (&(&three * &p + &five * &a) / &eight, p);
            fp = f(&p);
        } else if fb <= fp {
            // unbalanced \ or /\, do a biased step.
            fa = fp;
            (p, a) = (&(&three * &p + &five * &b) / &eight, p);
            fp = f(&p);
            (a, b) = (b, a);
            (fa, fb) = (fb, fa);
        } else {
            let x = &(&five * &p + &three * &b) / &eight;
            let fx = f(&x);
            if fx > fp {
                b = a;
                fb = fa;
                a = x;
                fa = fx;
            } else {
                a = p;
                fa = fp;
                p = x;
                fp = fx;
            }
        }
        if a == p || p == b || fp == fa && fp == fb {
            return Ok(p);
        }
    }
    Err(&(&a + &b) / &two)
}

/// Use Brent's method to find a minimum of a continuous function.
/// Not to be confused with Brent's method for root-finding.
/// It will always converge to a minimum, even if there is multiple.
/// There is no guarantee that it finds the global minimum.
///
/// See: Numerical Recipes 10.3 Parabolic Interpolation and Brent's Method in One Dimension  
/// This is however not a faithful implementation, so it has slightly changed (mostly similar) performance characteristics.
pub fn minimize_brent<T: Field + RealNum>(f: impl Fn(&T) -> T, mut a: T, mut b: T, max_iter: usize, checked: bool) -> Result<T, T>
where
    for<'a> &'a T: AddMulSubDiv<Output = T>,
{
    // The field must support x * 2^6
    assert!(
        T::CHAR == 0 || T::CHAR > 1 << 6,
        "The finite field with characteristic {} is too small",
        T::CHAR
    );
    if a != a || b != b {
        return Err(a + b); // NaN values detected
    }
    let mut fa = f(&a);
    if a == b {
        return Ok(a);
    }
    assert!(a < b);
    let mut fb = f(&b);
    // start with bisected interval for maximal early information, use the golden ratio from here onwards
    let two = T::one() + T::one();
    let n = two.powu(6);
    let n2 = two.powu(5);
    let mut p = &(&a + &b) / &two; // bisection is optimal for high quality quadratic interpolation
    let mut fp = f(&p);
    for _ in 0..=max_iter {
        if fa == fp && fp == fb && p != a && p != b || a == b {
            // proven exact local minimum
            return Ok(p);
        }
        // conditions for the start until a \/ shape is found
        if fa < fp && fa <= fb {
            // unbalanced / or /\, do a fast-step to balance it
            b = p.clone();
            fb = fp;
            p = &a + &(&(&p - &a) / &n);
            fp = f(&p);
        } else if fb < fp {
            // unbalanced \ or /\, do a fast-step to balance it
            a = p.clone();
            fa = fp;
            p = &b + &(&(&p - &b) / &n);
            fp = f(&p);
        } else if a != p && fa == fp {
            b = p.clone();
            fb = fp;
            p = &(&a + &p) / &two;
            fp = f(&p);
        } else if b != p && fb == fp {
            a = p.clone();
            fa = fp;
            p = &(&b + &p) / &two;
            fp = f(&p);
        } else {
            // compute the expected minimum using quadratic interpolation
            // it's always in [a, b] since fp is smaller than fa and fb
            let ba = &b - &a;
            if &a + &(&ba / &two) == a {
                // the interval has no more values -> minimum found
                return Ok(if fa < fb { a } else { b });
            }
            let bp = &b - &p;
            let bcfba = &bp * &(&fb - &fa);
            let bafbc = &ba * &(&fb - &fp);
            let q = &(&bafbc - &bcfba) * &two;
            let mut dx = T::zero();
            if q.is_zero() {
                // in this case, the 3 points have been collinear.
            } else {
                // Note, this would benefit greatly from mul_add (as in the precise complex multiplication)
                dx = (&ba * &bafbc - &bp * &bcfba) / q;
            }
            let mut x = &b - &dx;
            // if the same minimum is predicted twice, check it faster (without this the braketing would have linear convergence)
            if x == p {
                // check the previous and next ulp. if one of them fails, set the x accordingly.
                if checked {
                    let mut ulp = x.clone();
                    for _ in 0..1024 {
                        let next = &ulp / &two;
                        if &x + &next == x {
                            break;
                        }
                        ulp = next;
                    }
                    let a1 = &x - &ulp;
                    let b1 = &x + &ulp;
                    let fa1 = f(&a1);
                    let fb1 = f(&b1);
                    if fp <= fa1 && fp <= fb1 {
                        return Ok(p);
                    } else if fp > fa1 {
                        x = a1;
                    } else {
                        x = b1;
                    }
                } else {
                    return Ok(x);
                }
            }
            // limit x so it has some distance from the borders
            let xmin = &a + &(&ba / &n2);
            if x < xmin {
                x = xmin;
            } else {
                let xmax = &b - &(&ba / &n2);
                if x > xmax {
                    x = xmax;
                }
            }
            let fx = f(&x);
            if fx < fp {
                // x will become p
                if p < x {
                    a = p;
                    fa = fp;
                } else {
                    b = p;
                    fb = fp;
                }
                p = x;
                fp = fx;
            } else {
                // x will become a boundary
                if p < x {
                    b = x;
                    fb = fx;
                } else {
                    a = x;
                    fa = fx;
                }
            }
        }
        //std::println!("{a:?}, {p:?}, {b:?} ({fa:?}, {fp:?}, {fb:?})");
    }
    Err(p)
}
