use crate::*;

macro_rules! assign_op {
    ($x:ident + $y:expr) => {
        $x += $y
    };
    ($x:ident - $y:expr) => {
        $x -= $y
    };
    ($x:ident * $y:expr) => {
        $x *= $y
    };
    ($x:ident / $y:expr) => {
        $x /= $y
    };
    ($x:ident % $y:expr) => {
        $x %= $y
    };
}

// Test both a + b and a += b and with references
macro_rules! test_op {
    ($a:ident $op:tt $b:expr, $answer:expr) => {
        #[allow(unused_parens)]
        {
            assert_eq!($a $op $b, $answer);
            assert_eq!(&$a $op &$b, $answer);
            let mut x = $a;
            assign_op!(x $op $b);
            assert_eq!(x, $answer);
            let mut x = $a;
            assign_op!(x $op &$b);
            assert_eq!(x, $answer);
        }
    };
}

macro_rules! test_div_rem {
    ($a:ident, $b:expr) => {
        let (q, r) = ($a / $b, $a % $b);
        test_op!($a / $b, q);
        test_op!($a % $b, r);
        let b = $b;
        test_op!(b * q, $a - r);
        test_op!($a - r, $b * q);
    };
}

#[rustfmt::skip]
#[allow(unused_macros)]
macro_rules! for_integers {
    ($code:block) => {
        { type T = u8; $code }{ type T = u16; $code }{ type T = u32; $code }{ type T = u64; $code }{ type T = u128; $code }{ type T = i8; $code }{ type T = i16; $code }{ type T = i32; $code }{ type T = i64; $code }{ type T = i128; $code }
    }; // isize and usize are one of those and don't need to be tested separately.};
}

#[cfg(not(feature = "std"))]
use core::fmt::{self, Write};
#[cfg(not(feature = "std"))]
struct NoStdTester {
    cursor: usize,
    buf: [u8; Self::BUF_SIZE],
}

#[cfg(not(feature = "std"))]
impl NoStdTester {
    const WRITE_ERR: &'static str = "Formatted output too long";
    const BUF_SIZE: usize = 64;

    fn new() -> Self {
        Self {
            cursor: 0,
            buf: [0; Self::BUF_SIZE],
        }
    }

    fn to_str(&self) -> Result<&str, core::str::Utf8Error> {
        core::str::from_utf8(&self.buf[..self.cursor])
    }
}

#[cfg(not(feature = "std"))]
impl Write for NoStdTester {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            *self.buf.get_mut(self.cursor).ok_or(fmt::Error {})? = byte;
            self.cursor += 1;
        }
        Ok(())
    }
}

macro_rules! assert_fmt_eq {
    ($fmt_args:expr, $string:expr) => {
        #[cfg(not(feature = "std"))]
        {
            let mut tester = NoStdTester::new();
            write!(tester, "{}", $fmt_args).expect(NoStdTester::WRITE_ERR);
            assert_eq!(tester.to_str(), Ok($string));
        }
        #[cfg(feature = "std")]
        assert_eq!(std::fmt::format($fmt_args), $string);
    };
}

#[test]
fn test_gcd() {
    assert!(!f32::NAN.is_valid_euclid());
    assert!(!f64::NAN.is_valid_euclid());
    assert!(!f32::NEG_INFINITY.is_valid_euclid());
    assert!(!f64::NEG_INFINITY.is_valid_euclid());
    assert!(!f32::INFINITY.is_valid_euclid());
    assert!(!f64::INFINITY.is_valid_euclid());

    fn test<T: Copy + Num + core::fmt::Display + Cancel + core::ops::Div<Output = T>>(
        a: T,
        b: T,
        _gcd: T,
        _lcm: T,
    ) {
        assert_eq!(lcm(a, b), _lcm);
        assert_eq!(gcd(a, b), _gcd);
        assert_eq!(lcm(b, a), _lcm);
        assert_eq!(gcd(b, a), _gcd);
        let ((x, y), g2) = bezout(a, b);
        assert_eq!(a * x + b * y, g2);
        if a != Zero::zero() || b != Zero::zero() {
            assert_eq!(_gcd, g2, "{a} {b}");
        }
        let ((x, y), g2) = bezout(b, a);
        assert_eq!(b * x + a * y, g2);
        if a != Zero::zero() || b != Zero::zero() {
            assert_eq!(_gcd, g2, "{a} {b}");
        }
    }
    let a = 19 * 17 * 5 * 3;
    let b = 19 * 5 * 3;
    test(a, b, b, a);
    test(-a, b, b, a);
    test(-a, -b, b, a);
    test(a, -b, b, a);
    let c = 19 * 5 * 3 * 7;
    let l = 19 * 5 * 3 * 7 * 17;
    test(a, c, b, l);
    test(-a, c, b, l);
    test(-a, -c, b, l);
    test(a, -c, b, l);
    // zero tests
    test(0, b, b, 0);
    test(0, 0, 1, 0);
    // float tests
    let a = a as f32;
    let b = b as f32;
    test(a, b, b, a);
    test(-a, b, b, a);
    test(-a, -b, b, a);
    test(a, -b, b, a);

    // gcd always converges, but here the result is tiny:
    assert!(gcd(core::f64::consts::PI, core::f64::consts::E).abs() <= f64::EPSILON * 2.);
    assert!(bezout(core::f64::consts::PI, core::f64::consts::E).1.abs() <= f64::EPSILON * 2.);
}

#[cfg(all(feature = "rand", feature = "quaternion"))]
#[test]
fn test_rand() {
    use crate::rand::*;
    use ::rand::*;
    use ::rand::distr::*;
    let mut rng = ::rand::rngs::SmallRng::seed_from_u64(0xF8A39B09);

    // do a statistical tests with low sample number.
    let mut count = 0;
    let mut sum = 0.0f32;
    for _ in 0..100 {
        let x: f32 = rng.sample(StandardUniform);
        let c: Complex<f32> = rng.sample(StandardUniform);
        let q: Quaternion<f32> = rng.sample(StandardUniform);
        assert!(0. <= x && x < 1.);
        assert!(0. <= c.re && c.re < 1.);
        assert!(0. <= c.im && c.im < 1.);
        assert!(0. <= q.re && q.re < 1.);
        assert!(0. <= q.im_i && q.im_i < 1.);
        assert!(0. <= q.im_j && q.im_j < 1.);
        assert!(0. <= q.im_k && q.im_k < 1.);
        count += (x <= 0.5) as i32;
        count += (c.re <= 0.5) as i32;
        count += (c.im <= 0.5) as i32;
        count += (q.re <= 0.5) as i32;
        count += (q.im_i <= 0.5) as i32;
        count += (q.im_j <= 0.5) as i32;
        count += (q.im_k <= 0.5) as i32;
        sum += x;
        sum += c.re;
        sum += c.im;
        sum += q.re;
        sum += q.im_i;
        sum += q.im_j;
        sum += q.im_k;
    }
    // count should be ~100/2*7=350
    // at the time of testing this with the fixed seed, this has been 357
    assert!(count.abs_diff(350) < 15, "got {count}"); // allow a small error
    assert!((sum - 350.).abs() < 15., "got {sum}");

    // do the same test for normally distributed numbers
    let mut count = 0;
    let mut max = 0.0f32;
    let mut sum = 0.0f32;
    for _ in 0..100 {
        let x: f32 = rng.sample(StandardNormal);
        let c: Complex<f32> = rng.sample(StandardNormal);
        let q: Quaternion<f32> = rng.sample(StandardNormal);
        count += (x <= 0.0) as i32;
        count += (c.re <= 0.0) as i32;
        count += (c.im <= 0.0) as i32;
        count += (c.im * c.re <= 0.0) as i32; // this is also chance 1/2 and good to test!
        count += (q.re <= 0.0) as i32;
        count += (q.im_i <= 0.0) as i32;
        count += (q.im_j <= 0.0) as i32;
        count += (q.im_k <= 0.0) as i32;
        max = max.max(x);
        max = max.max(c.re);
        max = max.max(c.im);
        max = max.max(q.re);
        max = max.max(q.im_i);
        max = max.max(q.im_j);
        max = max.max(q.im_k);
        sum += x;
        sum += c.re;
        sum += c.im;
        sum += q.re;
        sum += q.im_i;
        sum += q.im_j;
        sum += q.im_k;
    }
    // count should be ~100/2*8=400
    assert!(count.abs_diff(400) < 15, "got {count}"); // allow a small error
    assert!(max > 2.5);
    assert!(max < 10.);
    assert!(sum.abs() < 30., "got {sum}");

    // now check if the unitary numbers are properly unitary
    let mut count = 0;
    let mut sum_c = Complex::<f32>::zero();
    let mut sum_q = Quaternion::<f32>::zero();
    for _ in 0..100 {
        let x: f32 = rng.sample(StandardUnitary);
        let c: Complex<f32> = rng.sample(StandardUnitary);
        let q: Quaternion<f32> = rng.sample(StandardUnitary);
        assert_eq!(x.abs() - 1.0, 0.0);
        assert!((c.abs() - 1.0).abs() < 2e-7);
        assert!((q.abs() - 1.0).abs() < 2e-7);
        count += (x <= 0.0) as i32;
        count += (c.re <= 0.0) as i32;
        count += (c.im <= 0.0) as i32;
        count += (c.im * c.re <= 0.0) as i32; // this is also chance 1/2 and good to test!
        count += (q.re <= 0.0) as i32;
        count += (q.im_i <= 0.0) as i32;
        count += (q.im_j <= 0.0) as i32;
        count += (q.im_k <= 0.0) as i32;
        sum_c += c;
        sum_q += q;
    }
    // count should be ~100/2*8=400
    assert!(count.abs_diff(400) < 15, "got {count}"); // allow a small error
    assert!(sum_c.abs() < 15., "got {sum_c}");
    assert!(sum_q.abs() < 15., "got {sum_q}");
}

#[cfg(any(feature = "std", feature = "libm"))]
#[test]
fn test_sign() {
    for x in [-2.0f64, -1.0, -1e-100, -0.0, 0.0, 1e-100, 1.0, 2.0] {
        assert_eq!(x.sign(), if x.is_sign_positive() { 1.0 } else { -1.0 });
        assert_eq!(1.0f64.copysign(x), if x.is_sign_positive() { 1.0 } else { -1.0 });
    }
    for x in [-2.0f32, -1.0, -1e-100, -0.0, 0.0, 1e-100, 1.0, 2.0] {
        assert_eq!(x.sign(), if x.is_sign_positive() { 1.0 } else { -1.0 });
        assert_eq!(1.0f32.copysign(x), if x.is_sign_positive() { 1.0 } else { -1.0 });
    }
}

#[allow(non_upper_case_globals)]
mod complex {
    use super::*;
    use core::*;

    type Complex64 = Complex<f64>;

    pub const _0_0i: Complex64 = Complex::new(0.0, 0.0);
    pub const _1_0i: Complex64 = Complex::new(1.0, 0.0);
    pub const _2_0i: Complex64 = Complex::new(2.0, 0.0);
    pub const neg_2_0i: Complex64 = Complex::new(-2.0, -0.0);
    pub const _1_1i: Complex64 = Complex::new(1.0, 1.0);
    pub const _0_1i: Complex64 = Complex::new(0.0, 1.0);
    pub const _neg1_1i: Complex64 = Complex::new(-1.0, 1.0);
    pub const _05_05i: Complex64 = Complex::new(0.5, 0.5);
    pub const _4_2i: Complex64 = Complex::new(4.0, 2.0);
    pub const _small1: Complex64 = Complex::new(1e-8, 0.0);
    pub const _small2: Complex64 = Complex::new(-1e-8, 0.0);
    pub const _small3: Complex64 = Complex::new(0.0, 1e-8);
    pub const _small4: Complex64 = Complex::new(0.0, -1e-8);
    pub const all_consts: [Complex64; 12] = [_0_0i, _1_0i, _1_1i, _neg1_1i, _05_05i, _2_0i, neg_2_0i, _4_2i, _small1, _small2, _small3, _small4];
    pub const _1_infi: Complex64 = Complex::new(1.0, f64::INFINITY);
    pub const _neg1_infi: Complex64 = Complex::new(-1.0, f64::INFINITY);
    pub const _1_nani: Complex64 = Complex::new(1.0, f64::NAN);
    pub const _neg1_nani: Complex64 = Complex::new(-1.0, f64::NAN);
    pub const _inf_0i: Complex64 = Complex::new(f64::INFINITY, 0.0);
    pub const _neginf_1i: Complex64 = Complex::new(f64::NEG_INFINITY, 1.0);
    pub const _neginf_neg1i: Complex64 = Complex::new(f64::NEG_INFINITY, -1.0);
    pub const _inf_1i: Complex64 = Complex::new(f64::INFINITY, 1.0);
    pub const _inf_neg1i: Complex64 = Complex::new(f64::INFINITY, -1.0);
    pub const _neginf_infi: Complex64 = Complex::new(f64::NEG_INFINITY, f64::INFINITY);
    pub const _inf_infi: Complex64 = Complex::new(f64::INFINITY, f64::INFINITY);
    pub const _neginf_nani: Complex64 = Complex::new(f64::NEG_INFINITY, f64::NAN);
    pub const _inf_nani: Complex64 = Complex::new(f64::INFINITY, f64::NAN);
    pub const _nan_0i: Complex64 = Complex::new(f64::NAN, 0.0);
    pub const _nan_1i: Complex64 = Complex::new(f64::NAN, 1.0);
    pub const _nan_neg1i: Complex64 = Complex::new(f64::NAN, -1.0);
    pub const _nan_nani: Complex64 = Complex::new(f64::NAN, f64::NAN);

    #[allow(dead_code)]
    fn close(a: Complex64, b: Complex64) -> bool {
        close_to_tol(a, b, 4e-15, b.abs_sqr())
    }
    #[allow(dead_code)]
    fn close_abs(a: Complex64, b: Complex64) -> bool {
        close_to_tol(a, b, 4e-15, 1.0 + a.abs_sqr())
    }
    #[allow(dead_code)]
    pub fn close_to_tol(a: Complex64, b: Complex64, tol: f64, sqr: f64) -> bool {
        // returns true if a and b are reasonably close
        let close = (a == b) || (a - b).abs_sqr() <= tol * tol * sqr;
        #[cfg(feature = "std")]
        if !close {
            std::println!("{:?} != {:?}", a, b);
        }
        close
    }
    // Version that also works if re or im are +inf, -inf, or nan
    #[allow(dead_code)]
    pub fn close_naninf(a: Complex64, b: Complex64) -> bool {
        close_naninf_to_tol(a, b, 1.0e-15)
    }
    pub fn close_naninf_to_tol(a: Complex64, b: Complex64, tol: f64) -> bool {
        let mut close = true;

        // Compare the real parts
        if a.re.is_finite() {
            if b.re.is_finite() {
                close = (a.re == b.re) || (a.re - b.re).abs() < tol * b.re.abs();
            } else {
                close = false;
            }
        } else if (a.re.is_nan() && !b.re.is_nan())
            || (a.re.is_infinite()
                && a.re.is_sign_positive()
                && !(b.re.is_infinite() && b.re.is_sign_positive()))
            || (a.re.is_infinite()
                && a.re.is_sign_negative()
                && !(b.re.is_infinite() && b.re.is_sign_negative()))
        {
            close = false;
        }

        // Compare the imaginary parts
        if a.im.is_finite() {
            if b.im.is_finite() {
                close &= (a.im == b.im) || (a.im - b.im).abs() < tol * b.im.abs();
            } else {
                close = false;
            }
        } else if (a.im.is_nan() && !b.im.is_nan())
            || (a.im.is_infinite()
                && a.im.is_sign_positive()
                && !(b.im.is_infinite() && b.im.is_sign_positive()))
            || (a.im.is_infinite()
                && a.im.is_sign_negative()
                && !(b.im.is_infinite() && b.im.is_sign_negative()))
        {
            close = false;
        }

        #[cfg(feature = "std")]
        if close == false {
            std::println!("{:?} != {:?}", a, b);
        }
        close
    }

    #[test]
    fn test_consts() {
        // check our constants are what Complex::new creates
        fn test(c: Complex64, r: f64, i: f64) {
            assert_eq!(c, Complex::new(r, i));
        }
        test(_0_0i, 0.0, 0.0);
        test(_1_0i, 1.0, 0.0);
        test(_1_1i, 1.0, 1.0);
        test(_neg1_1i, -1.0, 1.0);
        test(_05_05i, 0.5, 0.5);

        assert_eq!(_0_0i, Zero::zero());
        assert_eq!(_1_0i, One::one());
    }

    #[test]
    fn test_euclid() {
        for i in -5i64..=5 {
            for j in -5..=5 {
                for a in -5..=5 {
                    for b in -5..=5 {
                        //println!("{i} {j} {a} {b}");
                        let x = Complex::new(a, b);
                        let d = Complex::new(i, j);
                        let (q, r) = x.div_rem_euclid(&d);
                        assert!(r.is_valid_euclid());
                        assert_eq!(q * d + r, x);
                        if i != 0 || j != 0 {
                            assert!(r.abs_sqr() <= d.abs_sqr() / 2);
                        }
                        else {
                            assert!(q.is_zero());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_macros() {
        // this is only a syntax test
        let _ = complex![1 + 2 i];
        let _ = complex![(1+2) + 2 i];
        let _ = complex![1 + (2-1) i];
        let _ = complex![(1+2) + (2+3) i];
        let _ = complex![1 + 2 j];
        let _ = complex![(1+2) + 2 j];
        let _ = complex![1 + (2-1) j];
        let _ = complex![(1+2) + (2+3) j];
        let _ = complex![1 - 2 i];
        let _ = complex![(1+2) - 2 i];
        let _ = complex![1 - (2-1) i];
        let _ = complex![(1+2) - (2+3) i];
        let _ = complex![1 - 2 j];
        let _ = complex![(1+2) - 2 j];
        let _ = complex![1 - (2-1) j];
        let _ = complex![(1+2) - (2+3) j];
        let _ = complex![(2+3) i];
        let _ = complex![(2+3) j];
        let _: i32 = complex![2 + 3]; // uses .into() so a type is needed
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    mod analytic {
        use super::*;
        #[test]
        fn test_sqrt1() {
            // test my custom implementations in Num
            // first of all, here is the stable standard implementation from num_complex:
            fn sqrt(y: Complex<f64>) -> Complex<f64> {
                if y.im.is_zero() {
                    if y.re.is_sign_positive() {
                        // simple positive real √r, and copy `im` for its sign
                        Complex::new(y.re.sqrt(), y.im)
                    } else {
                        // √(r e^(iπ)) = √r e^(iπ/2) = i√r
                        // √(r e^(-iπ)) = √r e^(-iπ/2) = -i√r
                        let im = (-y.re).sqrt();
                        if y.im.is_sign_positive() {
                            Complex::new(0.0, im)
                        } else {
                            Complex::new(0.0, -im)
                        }
                    }
                } else if y.re.is_zero() {
                    // √(r e^(iπ/2)) = √r e^(iπ/4) = √(r/2) + i√(r/2)
                    // √(r e^(-iπ/2)) = √r e^(-iπ/4) = √(r/2) - i√(r/2)
                    let x = (y.im.abs() / 2.0).sqrt();
                    if y.im.is_sign_positive() {
                        Complex::new(x, x)
                    } else {
                        Complex::new(x, -x)
                    }
                } else {
                    // formula: sqrt(r e^(it)) = sqrt(r) e^(it/2)
                    let (r, theta) = y.to_polar();
                    Complex::from_polar(r.sqrt(), theta / 2.0)
                }
            }
            // test Complex::sqrt
            let a = Complex::new(0.0, 0.0);
            assert_eq!(a, sqrt(a));
            let a = Complex::new(1.0, 0.0);
            assert_eq!(a, sqrt(a));
            let a = Complex::new(-1.0, 0.0);
            assert_eq!(Complex::new(0.0, 1.0), sqrt(a));
            let a = Complex::new(5.0, 2.0);
            assert!((a.sqrt() - sqrt(a)).abs() < 1e-15);
            let a = Complex::new(-5.0, 2.0);
            assert!((a.sqrt() - sqrt(a)).abs() < 1e-15);
            let a = Complex::new(-5.0, -2.0);
            assert!((a.sqrt() - sqrt(a)).abs() < 1e-15);
            let a = Complex::new(5.0, -2.0);
            assert!((a.sqrt() - sqrt(a)).abs() < 1e-15);
            // if the values come really close to the negative real axis, the error rises
            // here the errorbound is already 1e-13, but it easily gets much worse.
            let a = Complex::new(-5.0, 0.01);
            assert!(
                (a.sqrt() - sqrt(a)).abs() < 1e-13,
                "error real {} vs my {}",
                a.sqrt(),
                sqrt(a)
            );
            // if the imaginary value is too small compares with the real value, it will snap to the axis
            let a = Complex::new(5.0, 2.0e-16);
            assert!((a.sqrt() - sqrt(a)).abs() < 1e-15);
            let a = Complex::new(5.0, -2.0e-16);
            assert!((a.sqrt() - sqrt(a)).abs() < 1e-15);
            let a = Complex::new(5.0, -2.0e-13);
            assert!((a.sqrt() - sqrt(a)).abs() < 1e-15);
            let a = Complex::new(5.0, -2.0e-10);
            assert!((a.sqrt() - sqrt(a)).abs() < 1e-15);
            let a = Complex::new(5.0, -2.0e-6);
            assert!((a.sqrt() - sqrt(a)).abs() < 1e-15);
            let a = Complex::new(5.0, -2.0e-3);
            assert!((a.sqrt() - sqrt(a)).abs() < 1e-15);
            let a = Complex::new(-1.0, 2.0e-20);
            assert!((a.sqrt() - Complex::new(1.0e-20, 1.0)).abs() < 0.2e-20);
            let a = Complex::new(-4.0, 8.0e-20);
            assert!((a.sqrt() - Complex::new(2.0e-20, 2.0)).abs() < 0.2e-20);
            let a = Complex::new(1.0, 2.0e-20);
            assert!((a.sqrt() - Complex::new(1.0, 1.0e-20)).abs() < 0.2e-20);
            let a = Complex::new(4.0, 8.0e-20);
            assert!((a.sqrt() - Complex::new(2.0, 2.0e-20)).abs() < 0.2e-20);
            let a = Complex::new(-1.0, 2.0e-15);
            assert!(
                (a.sqrt() - Complex::new(1.0e-15, 1.0)).abs() < 0.5e-15,
                "had {}",
                a.sqrt()
            );
            let a = Complex::new(-1.0, 2.0e-16);
            assert!((a.sqrt() - Complex::new(1.0e-16, 1.0)).abs() < 0.5e-16);
            // im: 0.0
            let a = Complex::new(-4.0, 0.0f64);
            assert!(
                (Complex::new(0.0, 2.0) - a.sqrt()).abs() < 1e-15,
                "error, this is wrong: {}",
                a.sqrt()
            );
            // im: -0.0
            let a = Complex::new(-4.0, -0.0f64);
            assert!(
                (Complex::new(0.0, -2.0) - a.sqrt()).abs() < 1e-15,
                "error, this is wrong: {}",
                a.sqrt()
            );
            // small negative real values
            let a = Complex::real(-1e-18);
            assert!((a.sqrt() - Complex::new(0.0, 1e-9)).abs() < 1e-16, "{a}");
            let a = Complex::new(-1e-18, -0.0);
            assert!((a.sqrt() - Complex::new(0.0, -1e-9)).abs() < 1e-16, "{a}");
        }

        #[test]
        fn test_arg() {
            fn test(c: Complex64, arg: f64) {
                assert!((c.arg() - arg).abs() < 1.0e-6)
            }
            test(_1_0i, 0.0);
            test(_1_1i, 0.25 * f64::consts::PI);
            test(_neg1_1i, 0.75 * f64::consts::PI);
            test(_05_05i, 0.25 * f64::consts::PI);
        }

        #[test]
        fn test_polar_conv() {
            fn test(c: Complex64) {
                let (r, theta) = c.to_polar();
                assert!((c - Complex::from_polar(r, theta)).abs() < 1e-6);
            }
            for &c in all_consts.iter() {
                test(c);
            }
        }

        #[test]
        fn test_exp() {
            assert!(close(_1_0i.exp(), _1_0i * f64::consts::E));
            assert_eq!(_0_0i.exp(), _1_0i);
            assert!(close(_0_1i.exp(), Complex::new(1.0.cos(), 1.0.sin())));
            assert!(close(_05_05i.exp() * _05_05i.exp(), _1_1i.exp()));
            assert!(close((_0_1i * (-f64::consts::PI)).exp(), _1_0i * (-1.0)));
            for &c in all_consts.iter() {
                // e^conj(z) = conj(e^z)
                assert_eq!(c.conj().exp(), c.exp().conj());
                // e^(z + 2 pi i) = e^z
                assert!(close(c.exp(), (c + _0_1i * (f64::consts::PI * 2.0)).exp()));
            }

            // The test values below were taken from https://en.cppreference.com/w/cpp/numeric/complex/exp
            assert!(close_naninf(_1_infi.exp(), _nan_nani));
            assert!(close_naninf(_neg1_infi.exp(), _nan_nani));
            assert!(close_naninf(_1_nani.exp(), _nan_nani));
            assert!(close_naninf(_neg1_nani.exp(), _nan_nani));
            assert!(close_naninf(_inf_0i.exp(), _inf_0i));
            //assert!(close_naninf(_neginf_1i.exp(), Complex::cis(1.0) * 0.0));
            //assert!(close_naninf(_neginf_neg1i.exp(), Complex::cis(-1.0) * 0.0));
            //assert!(close_naninf(_inf_1i.exp(), Complex::cis(1.0) * f64::INFINITY));
            //assert!(close_naninf(_inf_neg1i.exp(),Complex::cis(-1.0) * f64::INFINITY));
            assert!(close_naninf(_neginf_infi.exp(), _0_0i)); // Note: ±0±0i: signs of zeros are unspecified
            //assert!(close_naninf(_inf_infi.exp(), _inf_nani)); // Note: ±∞+NaN*i: sign of the real part is unspecified
            assert!(close_naninf(_neginf_nani.exp(), _0_0i)); // Note: ±0±0i: signs of zeros are unspecified
            //assert!(close_naninf(_inf_nani.exp(), _inf_nani)); // Note: ±∞+NaN*i: sign of the real part is unspecified
            assert!(close_naninf(_nan_0i.exp(), _nan_0i));
            assert!(close_naninf(_nan_1i.exp(), _nan_nani));
            assert!(close_naninf(_nan_neg1i.exp(), _nan_nani));
            assert!(close_naninf(_nan_nani.exp(), _nan_nani));
        }

        #[test]
        fn test_exp_m1() {
            assert!(close(_1_0i.exp_m1(), _1_0i * (f64::consts::E - 1.)));
            assert_eq!((-_0_0i).exp_m1(), _0_0i);
            assert!((-_0_0i).exp_m1().re.is_sign_negative());
            assert_eq!(_0_0i.exp_m1(), _0_0i);
            assert!(close(
                _0_1i.exp_m1() + 1.0,
                Complex::new(1.0.cos(), 1.0.sin())
            ));
            assert!(close(
                (_05_05i.exp_m1() + 1.) * (_05_05i.exp_m1() + 1.),
                _1_1i.exp_m1() + 1.
            ));
            assert!(close_abs(
                (_0_1i * (-f64::consts::PI)).exp_m1(),
                _1_0i * (-2.0)
            ));
            for &c in all_consts.iter() {
                // e^conj(z) = conj(e^z)
                assert_eq!(c.conj().exp_m1(), c.exp_m1().conj());
                // e^(z + 2 pi i) = e^z
                assert!(close_abs(
                    c.exp_m1(),
                    (c + _0_1i * (f64::consts::TAU)).exp_m1()
                ));
                // e^(z + pi i)-1 = -(e^z-1) - 2
                assert!(close_abs(
                    c.exp_m1(),
                    -(c + _0_1i * f64::consts::PI).exp_m1() - 2.
                ));
            }
            assert!(close(
                (_1_0i * 1e-12).exp_m1(),
                _1_0i * (NumElementary::exp_m1(&1e-12))
            ));
            assert!(close(
                (_1_0i * 1e-16).exp_m1(),
                _1_0i * (NumElementary::exp_m1(&1e-16))
            ));
            assert!(close((_0_1i * 1e-16).exp_m1(), _0_1i * 1e-16));
        }

        #[test]
        fn test_ln() {
            assert!(close(_1_0i.ln(), _0_0i));
            assert!(close(_0_1i.ln(), _0_1i * (f64::consts::PI / 2.0)));
            assert!(close(_0_0i.ln(), Complex::new(f64::NEG_INFINITY, 0.0)));
            assert!(close(Complex::new(f64::INFINITY, 0.0).ln(), Complex::new(f64::INFINITY, 0.0)));
            assert!(close(Complex::new(-f64::INFINITY, 0.0).ln(), Complex::new(f64::INFINITY, f64::consts::PI)));
            assert!(close(Complex::new(-f64::INFINITY, -0.0).ln(), Complex::new(f64::INFINITY, -f64::consts::PI)));
            assert!(close(Complex::new(0.0, f64::INFINITY).ln(), Complex::new(f64::INFINITY, f64::consts::FRAC_PI_2)));
            assert!(close(Complex::new(0.0, -f64::INFINITY).ln(), Complex::new(f64::INFINITY, -f64::consts::FRAC_PI_2)));
            assert!(close(
                (_neg1_1i * _05_05i).ln(),
                _neg1_1i.ln() + _05_05i.ln()
            ));
            for &c in all_consts.iter() {
                // ln(conj(z() = conj(ln(z))
                assert!(close(c.conj().ln(), c.ln().conj()));
                // for this branch, -pi <= arg(ln(z)) <= pi
                assert!(-f64::consts::PI <= c.ln().arg() && c.ln().arg() <= f64::consts::PI);
                // test ln_1p
                assert!(close((c + 1.0).ln(), c.ln_1p()));
            }
        }

        #[test]
        fn test_powc() {
            let a = Complex::new(2.0, -3.0);
            let b = Complex::new(3.0, 0.0);
            //assert!(close(a.pow(b), a.powf(b.re)));
            //assert!(close(b.pow(a), a.expf(b.re)));
            let c = Complex::new(1.0 / 3.0, 0.1);
            assert!(close_to_tol(
                a.pow(&c),
                Complex::new(1.65826, -0.33502),
                1e-5,
                1.0
            ));
            let z = Complex::new(0.0, 0.0);
            assert!(close(z.pow(&b), z));
            //assert!(z.pow(&Complex64::new(0., f64::INFINITY)).re.is_nan());
            //assert!(z.pow(&Complex64::new(10., f64::INFINITY)).re.is_nan());
            //assert!(z.pow(&_inf_infi).re.is_nan());
            assert!(close(z.pow(&Complex64::new(f64::INFINITY, 0.)), z));
            //assert!(z.pow(&Complex64::new(-1., 0.)).re.is_infinite());
            //assert!(z.pow(&Complex64::new(-1., 0.)).im.is_nan());

            for c in all_consts.iter() {
                assert_eq!(c.pow(&_0_0i), _1_0i);
            }
            assert_eq!(_nan_nani.pow(&_0_0i), _1_0i);
        }

        #[test]
        fn test_sqrt() {
            assert!(close(_0_0i.sqrt(), _0_0i));
            assert!(close(_1_0i.sqrt(), _1_0i));
            assert!(close(Complex::new(-1.0, 0.0).sqrt(), _0_1i));
            assert!(close(Complex::new(-1.0, -0.0).sqrt(), _0_1i * (-1.0)));
            assert!(close(_0_1i.sqrt(), _05_05i * (2.0.sqrt())));
            for &c in all_consts.iter() {
                // sqrt(conj(z() = conj(sqrt(z))
                assert!(close(c.conj().sqrt(), c.sqrt().conj()));
                // for this branch, -pi/2 <= arg(sqrt(z)) <= pi/2
                assert!(
                    -f64::consts::FRAC_PI_2 <= c.sqrt().arg()
                        && c.sqrt().arg() <= f64::consts::FRAC_PI_2
                );
                // sqrt(z) * sqrt(z) = z
                assert!(close(c.sqrt() * c.sqrt(), c));
            }
        }

        #[test]
        fn test_sqrt_real() {
            for n in (0..100).map(f64::from) {
                // √(n² + 0i) = n + 0i
                let n2 = n * n;
                assert_eq!(Complex64::new(n2, 0.0).sqrt(), Complex64::new(n, 0.0));
                // √(-n² + 0i) = 0 + ni
                assert_eq!(Complex64::new(-n2, 0.0).sqrt(), Complex64::new(0.0, n));
                // √(-n² - 0i) = 0 - ni
                assert_eq!(Complex64::new(-n2, -0.0).sqrt(), Complex64::new(0.0, -n));
            }
        }

        #[test]
        fn test_sqrt_imag() {
            for n in (0..100).map(f64::from) {
                // √(0 + n²i) = n e^(iπ/4)
                let n2 = n * n;
                assert!(close(
                    Complex64::new(0.0, n2).sqrt(),
                    Complex64::from_polar(n, f64::consts::FRAC_PI_4)
                ));
                // √(0 - n²i) = n e^(-iπ/4)
                assert!(close(
                    Complex64::new(0.0, -n2).sqrt(),
                    Complex64::from_polar(n, -f64::consts::FRAC_PI_4)
                ));
            }
        }

        #[test]
        fn test_cbrt() {
            assert!(close(_0_0i.cbrt(), _0_0i));
            assert!(close(_1_0i.cbrt(), _1_0i));
            assert!(close(
                Complex::new(-1.0, 0.0).cbrt(),
                Complex::new(0.5, 0.75.sqrt())
            ));
            assert!(close(
                Complex::new(-1.0, -0.0).cbrt(),
                Complex::new(0.5, -(0.75.sqrt()))
            ));
            assert!(close(_0_1i.cbrt(), Complex::new(0.75.sqrt(), 0.5)));
            assert!(close(_0_1i.conj().cbrt(), Complex::new(0.75.sqrt(), -0.5)));
            for &c in all_consts.iter() {
                // cbrt(conj(z() = conj(cbrt(z))
                assert_eq!(c.conj().cbrt(), c.cbrt().conj());
                // for this branch, -pi/3 <= arg(cbrt(z)) <= pi/3
                assert!(
                    -f64::consts::FRAC_PI_3 <= c.cbrt().arg()
                        && c.cbrt().arg() <= f64::consts::FRAC_PI_3
                );
                // cbrt(z) * cbrt(z) cbrt(z) = z
                assert!(close(c.cbrt() * c.cbrt() * c.cbrt(), c));
            }
        }

        #[test]
        fn test_cbrt_real() {
            for n in (0..100).map(f64::from) {
                // ∛(n³ + 0i) = n + 0i
                let n3 = n * n * n;
                assert!(close(
                    Complex64::new(n3, 0.0).cbrt(),
                    Complex64::new(n, 0.0)
                ));
                // ∛(-n³ + 0i) = n e^(iπ/3)
                assert!(close(
                    Complex64::new(-n3, 0.0).cbrt(),
                    Complex64::from_polar(n, f64::consts::FRAC_PI_3)
                ));
                // ∛(-n³ - 0i) = n e^(-iπ/3)
                assert!(close(
                    Complex64::new(-n3, -0.0).cbrt(),
                    Complex64::from_polar(n, -f64::consts::FRAC_PI_3)
                ));
            }
        }

        #[test]
        fn test_cbrt_imag() {
            for n in (0..100).map(f64::from) {
                // ∛(0 + n³i) = n e^(iπ/6)
                let n3 = n * n * n;
                assert!(close(
                    Complex64::new(0.0, n3).cbrt(),
                    Complex64::from_polar(n, f64::consts::FRAC_PI_6)
                ));
                // ∛(0 - n³i) = n e^(-iπ/6)
                assert!(close(
                    Complex64::new(0.0, -n3).cbrt(),
                    Complex64::from_polar(n, -f64::consts::FRAC_PI_6)
                ));
            }
        }

        #[test]
        fn test_sign() {
            for &c in all_consts.iter() {
                // sign(conj(z)) = conj(sign(z))
                assert_eq!(c.conj().sign(), c.sign().conj());
                // sign(-z) = -sign(z)
                assert_eq!((c * -1.0).sign(), c.sign() * -1.0);
                if !c.abs_sqr().is_zero() {
                    // sign(z/|z|) = sign(z)
                    assert!(close((c / c.abs()).sign(), c.sign()));
                }
                // |sign(z)| = 1
                assert!((c.sign().abs() - 1.0).abs() < 1e-14);
                // copysign(1, z) = sign(z)
                assert!(close(_1_0i.copysign(&c), c.sign()));
                // copysign(i, z) = sign(z)
                assert!(close(_0_1i.copysign(&c), c.sign()));
            }
        }

        #[test]
        fn test_sin() {
            assert!(close(_0_0i.sin(), _0_0i));
            assert_eq!((-_0_0i).sin(), _0_0i);
            assert!((-_0_0i).sin().re.is_sign_negative());
            assert!(close_abs((_1_0i * f64::consts::PI * 2.0).sin(), _0_0i));
            assert!(close(_0_1i.sin(), _0_1i * 1.0.sinh()));
            for &c in all_consts.iter() {
                // sin(conj(z)) = conj(sin(z))
                assert_eq!(c.conj().sin(), c.sin().conj());
                // sin(-z) = -sin(z)
                assert!(close((c * -1.0).sin(), c.sin() * -1.0));
            }
        }

        #[test]
        fn test_cos() {
            assert!(close(_0_0i.cos(), _1_0i));
            assert!(close((_1_0i * f64::consts::PI * 2.0).cos(), _1_0i));
            assert!(close(_0_1i.cos(), _1_0i * (1.0.cosh())));
            for &c in all_consts.iter() {
                // cos(conj(z)) = conj(cos(z))
                assert_eq!(c.conj().cos(), c.cos().conj());
                // cos(-z) = cos(z)
                assert!(close((c * -1.0).cos(), c.cos()));
            }
        }

        #[test]
        fn test_cis() {
            assert!(close(Complex::cis(&0.0), _1_0i));
            assert!(close(Complex::cis(&core::f64::consts::FRAC_PI_2), _0_1i));
            assert!(close(Complex::cis(&-core::f64::consts::FRAC_PI_2), -_0_1i));
            assert!(close(Complex::cis(&core::f64::consts::PI), -_1_0i));
        }

        #[test]
        fn test_tan() {
            assert!(close(_0_0i.tan(), _0_0i));
            assert_eq!((-_0_0i).tan(), _0_0i);
            assert!((-_0_0i).tan().re.is_sign_negative());
            assert!(close((_1_0i * f64::consts::PI / 4.0).tan(), _1_0i));
            assert!(close_abs((_1_0i * f64::consts::PI).tan(), _0_0i));
            for &c in all_consts.iter() {
                // tan(conj(z)) = conj(tan(z))
                assert_eq!(c.conj().tan(), c.tan().conj());
                // tan(-z) = -tan(z)
                assert!(close((c * -1.0).tan(), c.tan() * (-1.0)));
            }
        }

        #[test]
        fn test_asin() {
            assert_eq!(_0_0i.asin(), _0_0i);
            assert_eq!((-_0_0i).asin(), _0_0i);
            assert!((-_0_0i).asin().re.is_sign_negative());
            assert!(close(_1_0i.asin(), _1_0i * (f64::consts::PI / 2.0)));
            assert!(close(
                (_1_0i * -1.0).asin(),
                _1_0i * (-f64::consts::PI / 2.0)
            ));
            assert!(close(_0_1i.asin(), _0_1i * ((1.0 + 2.0.sqrt()).ln())));
            for &c in all_consts.iter() {
                // asin(conj(z)) = conj(asin(z))
                assert!(close_abs(c.conj().asin(), c.asin().conj()), "{c}");
                // asin(-z) = -asin(z)
                assert!(close_abs((c * -1.0).asin(), c.asin() * -1.0));
                // for this branch, -pi/2 <= asin(z).re <= pi/2
                assert!(
                    -f64::consts::PI / 2.0 <= c.asin().re && c.asin().re <= f64::consts::PI / 2.0
                );
            }
        }

        #[test]
        fn test_acos() {
            assert!(close(_0_0i.acos(), _1_0i * (f64::consts::PI / 2.0)));
            assert!(close_abs(_1_0i.acos(), _0_0i));
            assert!(close((_1_0i * -1.0).acos(), _1_0i * f64::consts::PI));
            assert!(close(
                _0_1i.acos(),
                Complex::new(f64::consts::PI / 2.0, (2.0.sqrt() - 1.0).ln())
            ));
            for &c in all_consts.iter() {
                // acos(conj(z)) = conj(acos(z))
                assert!(close(c.conj().acos(), c.acos().conj()));
                // for this branch, 0 <= acos(z).re <= pi
                assert!(0.0 <= c.acos().re && c.acos().re <= f64::consts::PI);
            }
        }

        #[test]
        fn test_atan() {
            assert_eq!(_0_0i.atan(), _0_0i);
            assert_eq!((-_0_0i).atan(), _0_0i);
            assert!((-_0_0i).atan().re.is_sign_negative());
            assert!(close(_1_0i.atan(), _1_0i * (f64::consts::PI / 4.0)));
            assert!(close(Complex::real(f64::INFINITY).atan(), _1_0i * (f64::consts::PI / 2.0)));
            assert!(close(Complex::real(-f64::INFINITY).atan(), _1_0i * (-f64::consts::PI / 2.0)));
            assert!(close(
                (_1_0i * -1.0).atan(),
                _1_0i * (-f64::consts::PI / 4.0)
            ));
            assert!(close(_0_1i.atan(), Complex::new(0.0, f64::INFINITY)));
            for &c in all_consts.iter() {
                // atan(conj(z)) = conj(atan(z))
                assert!(close(c.conj().atan(), c.atan().conj()));
                // atan(-z) = -atan(z)
                assert!(close((c * -1.0).atan(), c.atan() * -1.0));
                // for this branch, -pi/2 <= atan(z).re <= pi/2
                assert!(
                    -f64::consts::PI / 2.0 <= c.atan().re && c.atan().re <= f64::consts::PI / 2.0
                );
            }
        }

        #[test]
        fn test_atan2() {
            assert!(close(_0_0i.atan2(&_1_0i), _0_0i));
            assert!(close(_1_0i.atan2(&_1_0i), _1_0i * (f64::consts::PI / 4.0)));
            assert!(close(_1_1i.atan2(&_1_1i), _1_0i * (f64::consts::PI / 4.0)));
            assert!(close(_1_0i.atan2(&_0_0i), _1_0i * (f64::consts::PI / 2.0)));
            assert!(close((-_1_0i).atan2(&_0_0i), _1_0i * (-f64::consts::PI / 2.0)));
            assert!(close((-_1_0i).atan2(&-_0_0i), _1_0i * (-f64::consts::PI / 2.0)));
            assert!(close(_0_0i.atan2(&-_1_0i), _1_0i * f64::consts::PI));
            assert!(close((-_0_0i).atan2(&-_1_0i), _1_0i * -f64::consts::PI));
            assert!(close(_0_0i.atan2(&_0_0i), _0_0i)); // center, default to 0 instead of NaN
            assert!(close(
                (_1_0i * -1.0).atan2(&_1_0i),
                _1_0i * (-f64::consts::PI / 4.0)
            ));

            assert!(close(_0_1i.atan2(&_1_0i), Complex::new(0.0, f64::INFINITY)));
            assert!(close(_1_0i.atan2(&_0_1i), Complex::new(0.0, -f64::INFINITY)));
            assert!(close(_1_1i.atan2(&_neg1_1i), Complex::new(0.0, f64::INFINITY)));
            assert!(close(_1_1i.atan2(&_1_0i), _1_1i.atan()));
            // test approaching the branch cut from different directions
            assert!(close(Complex::real(2e-8).atan2(&-_1_0i), _1_0i * 3.1415926313853326));
            assert!(close(Complex::real(-2e-8).atan2(&-_1_0i), _1_0i * -3.1415926313853326));
            // zero signs are not properly handled here and I believe they can't be, because they don't hold enough information
            assert!(close(Complex::new(1e-16, 2e-7).atan2(&-_1_0i), Complex::new(3.141592653589793, -1.9984014441655406e-7)));
            assert!(close(Complex::new(1e-16, -2e-7).atan2(&-_1_0i), Complex::new(3.141592653589793, 1.9984014441655406e-7)));
            assert!(close(Complex::new(-1e-16, 2e-7).atan2(&-_1_0i), Complex::new(-3.141592653589793, -1.9984014441655406e-7)));
            assert!(close(Complex::new(-1e-16, -2e-7).atan2(&-_1_0i), Complex::new(-3.141592653589793, 1.9984014441655406e-7)));
            assert!(close(Complex::real(0.5e-8).atan2(&-_1_0i), _1_0i * f64::consts::PI));
            assert!(close(Complex::real(-0.5e-8).atan2(&-_1_0i), _1_0i * -f64::consts::PI));
            assert!(close(Complex::imag(0.5e-8).atan2(&-_1_0i), Complex::new(f64::consts::PI, 0.0)));
            assert!(close(Complex::imag(-0.5e-8).atan2(&-_1_0i), Complex::new(f64::consts::PI, 0.0)));
            assert!(close(Complex::new(-0.0, 0.5e-8).atan2(&-_1_0i), Complex::new(-f64::consts::PI, 0.0)));
            assert!(close(Complex::new(-0.0, -0.5e-8).atan2(&-_1_0i), Complex::new(-f64::consts::PI, 0.0)));
        }

        #[test]
        fn test_sinh() {
            assert_eq!(_0_0i.sinh(), _0_0i);
            assert_eq!((-_0_0i).sinh(), _0_0i);
            assert!((-_0_0i).sinh().re.is_sign_negative());
            assert!(close(
                _1_0i.sinh(),
                _1_0i * ((f64::consts::E - 1.0 / f64::consts::E) / 2.0)
            ));
            assert!(close(_0_1i.sinh(), _0_1i * (1.0.sin())));
            for &c in all_consts.iter() {
                // sinh(conj(z)) = conj(sinh(z))
                assert!(close(c.conj().sinh(), c.sinh().conj()));
                // sinh(-z) = -sinh(z)
                assert!(close((c * -1.0).sinh(), c.sinh() * -1.0));
            }
        }

        #[test]
        fn test_cosh() {
            assert!(close(_0_0i.cosh(), _1_0i));
            assert!(close(
                _1_0i.cosh(),
                _1_0i * ((f64::consts::E + 1.0 / f64::consts::E) / 2.0)
            ));
            assert!(close(_0_1i.cosh(), _1_0i * 1.0.cos()));
            for &c in all_consts.iter() {
                // cosh(conj(z)) = conj(cosh(z))
                assert!(close(c.conj().cosh(), c.cosh().conj()));
                // cosh(-z) = cosh(z)
                assert!(close((c * -1.0).cosh(), c.cosh()));
            }
        }

        #[test]
        fn test_tanh() {
            assert!(close(_0_0i.tanh(), _0_0i));
            assert_eq!((-_0_0i).tanh(), _0_0i);
            assert!((-_0_0i).tanh().re.is_sign_negative());
            assert!(close(
                _1_0i.tanh(),
                _1_0i * ((f64::consts::E.powi(2) - 1.0) / (f64::consts::E.powi(2) + 1.0))
            ));
            assert!(close(_0_1i.tanh(), _0_1i * (1.0.tan())));
            for &c in all_consts.iter() {
                // tanh(conj(z)) = conj(tanh(z))
                assert!(close(c.conj().tanh(), c.tanh().conj()));
                // tanh(-z) = -tanh(z)
                assert!(close((c * -1.0).tanh(), c.tanh() * -1.0));
            }
        }

        #[test]
        fn test_asinh() {
            assert!(close(_0_0i.asinh(), _0_0i));
            assert_eq!((-_0_0i).asinh(), _0_0i);
            assert!((-_0_0i).asinh().re.is_sign_negative());
            assert!(close(_1_0i.asinh(), _1_0i * (1.0 + 2.0.sqrt()).ln()));
            assert!(close(_0_1i.asinh(), _0_1i * (f64::consts::PI / 2.0)));
            assert!(close(
                _0_1i.asinh() * -1.0,
                _0_1i * (-f64::consts::PI / 2.0)
            ));
            for &c in all_consts.iter() {
                // asinh(conj(z)) = conj(asinh(z))
                assert!(close_abs(c.conj().asinh(), c.asinh().conj()));
                // asinh(-z) = -asinh(z)
                assert!(close_abs((c * -1.0).asinh(), c.asinh() * -1.0));
                // for this branch, -pi/2 <= asinh(z).im <= pi/2
                assert!(
                    -f64::consts::PI / 2.0 <= c.asinh().im && c.asinh().im <= f64::consts::PI / 2.0
                );
            }
        }

        #[test]
        fn test_acosh() {
            assert!(close(_0_0i.acosh(), _0_1i * (f64::consts::PI / 2.0)));
            assert!(close(_1_0i.acosh(), _0_0i));
            assert!(close(
                (Complex::new(-1., 0.)).acosh(),
                _0_1i * f64::consts::PI
            ));
            assert!(close((_1_0i * -1.0).acosh(), -_0_1i * f64::consts::PI)); // zero sign is used!
            for &c in all_consts.iter() {
                // acosh(conj(z)) = conj(acosh(z))
                assert!(close(c.conj().acosh(), c.acosh().conj()));
                // for this branch, -pi <= acosh(z).im <= pi and 0 <= acosh(z).re
                assert!(
                    -f64::consts::PI <= c.acosh().im
                        && c.acosh().im <= f64::consts::PI
                        && 0.0 <= c.acosh().re
                );
            }
        }

        #[test]
        fn test_atanh() {
            assert!(close(_0_0i.atanh(), _0_0i));
            assert_eq!((-_0_0i).atanh(), _0_0i);
            assert!((-_0_0i).atanh().re.is_sign_negative());
            assert!(close(_0_1i.atanh(), _0_1i * (f64::consts::PI / 4.0)));
            assert!(close(_1_0i.atanh(), Complex::new(f64::INFINITY, 0.0)));
            assert!(close(Complex::new(1.0, -0.0).atanh(), Complex::new(f64::INFINITY, 0.0)));
            assert!(close(Complex::new(-1.0, 0.0).atanh(), Complex::new(-f64::INFINITY, 0.0)));
            assert!(close(Complex::new(-1.0, -0.0).atanh(), Complex::new(-f64::INFINITY, 0.0)));
            for &c in all_consts.iter() {
                // atanh(conj(z)) = conj(atanh(z))
                assert!(close(c.conj().atanh(), c.atanh().conj()));
                // atanh(-z) = -atanh(z)
                assert!(close((c * -1.0).atanh(), c.atanh() * -1.0));
                // for this branch, -pi/2 <= atanh(z).im <= pi/2
                assert!(
                    -f64::consts::PI / 2.0 <= c.atanh().im && c.atanh().im <= f64::consts::PI / 2.0
                );
            }
        }

        #[test]
        fn test_exp_ln() {
            for &c in all_consts.iter() {
                // e^ln(z) = z
                assert!(close(c.ln().exp(), c));
            }
        }

        #[test]
        fn test_trig_to_hyperbolic() {
            for &c in all_consts.iter() {
                // sin(iz) = i sinh(z)
                assert!(close((_0_1i * c).sin(), _0_1i * c.sinh()));
                // cos(iz) = cosh(z)
                assert!(close((_0_1i * c).cos(), c.cosh()));
                // tan(iz) = i tanh(z)
                assert!(close((_0_1i * c).tan(), _0_1i * c.tanh()));
            }
        }

        #[test]
        fn test_trig_identities() {
            for &c in all_consts.iter() {
                // tan(z) = sin(z)/cos(z)
                assert!(close(c.tan(), c.sin() / c.cos()));
                // tan(z) = sin(z)/cos(z)
                assert!(close(c.tan(), &c.sin() / &c.cos()));
                // sin(z)^2 + cos(z)^2 = 1
                assert!(close(c.sin() * c.sin() + c.cos() * c.cos(), _1_0i));

                // sin(asin(z)) = z
                assert!(close_abs(c.asin().sin(), c));
                // cos(acos(z)) = z
                assert!(close_abs(c.acos().cos(), c));
                // tan(atan(z)) = z
                // i and -i are branch points
                if c != _0_1i && c != _0_1i * (-1.0) {
                    assert!(close_abs(c.atan().tan(), c));
                }

                // sin(z) = (e^(iz) - e^(-iz))/(2i)
                assert!(close_abs(
                    ((_0_1i * c).exp() - (_0_1i * c).exp().recip()) / (_0_1i * 2.0),
                    c.sin()
                ));
                // cos(z) = (e^(iz) + e^(-iz))/2
                assert!(close(
                    ((_0_1i * c).exp() + (_0_1i * c).exp().recip()) / 2.0,
                    c.cos()
                ));
                // tan(z) = i (1 - e^(2iz))/(1 + e^(2iz))
                assert!(close_abs(
                    _0_1i * (_1_0i - (_0_1i * c * 2.0).exp()) / (_1_0i + (_0_1i * c * 2.0).exp()),
                    c.tan()
                ));
            }
        }

        #[test]
        fn test_hyperbolic_identites() {
            for &c in all_consts.iter() {
                // tanh(z) = sinh(z)/cosh(z)
                assert!(close(c.tanh(), c.sinh() / c.cosh()));
                // cosh(z)^2 - sinh(z)^2 = 1
                assert!(close(c.cosh() * c.cosh() - c.sinh() * c.sinh(), _1_0i));

                // sinh(asinh(z)) = z
                assert!(close_abs(c.asinh().sinh(), c));
                // cosh(acosh(z)) = z
                assert!(close_abs(c.acosh().cosh(), c));
                // tanh(atanh(z)) = z
                // 1 and -1 are branch points
                if c != _1_0i && c != _1_0i * (-1.0) {
                    assert!(close_abs(c.atanh().tanh(), c));
                }

                // sinh(z) = (e^z - e^(-z))/2
                assert!(close_abs((c.exp() - c.exp().recip()) / 2.0, c.sinh()));
                // cosh(z) = (e^z + e^(-z))/2
                assert!(close((c.exp() + c.exp().recip()) / 2.0, c.cosh()));
                // tanh(z) = ( e^(2z) - 1)/(e^(2z) + 1)
                assert!(close_abs(
                    ((c * 2.0).exp() - _1_0i) / ((c * 2.0).exp() + _1_0i),
                    c.tanh()
                ));
            }
        }
    }

    #[test]
    fn test_string_formatting() {
        assert_fmt_eq!(format_args!("{}", _0_0i), "0+0i");
        assert_fmt_eq!(format_args!("{}", _1_0i), "1+0i");
        assert_fmt_eq!(format_args!("{}", _0_1i), "0+1i");
        assert_fmt_eq!(format_args!("{}", _1_1i), "1+1i");
        assert_fmt_eq!(format_args!("{}", _neg1_1i), "-1+1i");
        assert_fmt_eq!(format_args!("{}", -_neg1_1i), "1-1i");
        assert_fmt_eq!(format_args!("{}", _05_05i), "0.5+0.5i");
        // pretty printing
        assert_fmt_eq!(format_args!("{:#}", _0_0i), "0");
        assert_fmt_eq!(format_args!("{:#}", _1_0i), "1");
        assert_fmt_eq!(format_args!("{:#}", _0_1i), "i");
        assert_fmt_eq!(format_args!("{:#}", _1_1i), "1+i");
        assert_fmt_eq!(format_args!("{:#}", _neg1_1i), "-1+i");
        assert_fmt_eq!(format_args!("{:#}", -_neg1_1i), "1-i");
        assert_fmt_eq!(format_args!("{:#}", _05_05i), "0.5+0.5i");

        let a = Complex::new(1.23456, 123.456);
        assert_fmt_eq!(format_args!("{}", a), "1.23456+123.456i");
        assert_fmt_eq!(format_args!("{:.2}", a), "1.23+123.46i");
        assert_fmt_eq!(format_args!("{:.2e}", a), "1.23e0+1.23e2i");
        assert_fmt_eq!(format_args!("{:+.2E}", a), "+1.23E0+1.23E2i");
        assert_fmt_eq!(format_args!("{:+20.2E}", a), "     +1.23E0+1.23E2i");

        let b = Complex::new(0x80, 0xff);
        assert_fmt_eq!(format_args!("{:X}", b), "80+FFi");
        assert_fmt_eq!(format_args!("{:#x}", b), "0x80+0xffi");
        assert_fmt_eq!(format_args!("{:+#b}", b), "+0b10000000+0b11111111i");
        assert_fmt_eq!(format_args!("{:+#o}", b), "+0o200+0o377i");
        assert_fmt_eq!(format_args!("{:+#16o}", b), "   +0o200+0o377i");
        assert_fmt_eq!(format_args!("{:<+#16o}", b), "+0o200+0o377i   ");
        assert_fmt_eq!(format_args!("{:^+#17o}", b), "  +0o200+0o377i  ");
        assert_fmt_eq!(format_args!("{:-^+#17o}", b), "--+0o200+0o377i--");

        let c = Complex::new(-10, -10000);
        assert_fmt_eq!(format_args!("{}", c), "-10-10000i");
        assert_fmt_eq!(format_args!("{:16}", c), "      -10-10000i");
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_hashset() {
        use std::collections::HashSet;
        let a = Complex::new(0i32, 0i32);
        let b = Complex::new(1i32, 0i32);
        let c = Complex::new(0i32, 1i32);

        let set: HashSet<_> = [a, b, c].iter().cloned().collect();
        assert!(set.contains(&a));
        assert!(set.contains(&b));
        assert!(set.contains(&c));
        assert!(!set.contains(&(a + b + c)));
    }

    #[test]
    fn test_sum() {
        let v = [_0_1i, _1_0i];
        assert_eq!(v.iter().sum::<Complex64>(), _1_1i);
        assert_eq!(v.into_iter().sum::<Complex64>(), _1_1i);
        let v = [];
        assert_eq!(v.iter().sum::<Complex64>(), _0_0i);
        assert_eq!(v.into_iter().sum::<Complex64>(), _0_0i);
    }

    #[test]
    fn test_prod() {
        let v = [_0_1i, _1_0i];
        assert_eq!(v.iter().product::<Complex64>(), _0_1i);
        assert_eq!(v.into_iter().product::<Complex64>(), _0_1i);
        let v = [];
        assert_eq!(v.iter().product::<Complex64>(), _1_0i);
        assert_eq!(v.into_iter().product::<Complex64>(), _1_0i);
    }

    #[test]
    fn test_zero() {
        let zero = Complex64::zero();
        assert!(zero.is_zero());

        let mut c = Complex::new(1.23, 4.56);
        assert!(!c.is_zero());
        assert_eq!(c + zero, c);

        c.set_zero();
        assert!(c.is_zero());
    }

    #[test]
    fn test_one() {
        let one = Complex64::one();
        assert!(one.is_one());

        let mut c = Complex::new(1.23, 4.56);
        assert!(!c.is_one());
        assert_eq!(c * one, c);

        c.set_one();
        assert!(c.is_one());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_const() {
        const R: f64 = 12.3;
        const I: f64 = -4.5;
        const C: Complex64 = Complex::new(R, I);

        assert_eq!(C.re, 12.3);
        assert_eq!(C.im, -4.5);
    }

    #[test]
    fn test_div() {
        test_op!(_neg1_1i / _0_1i, _1_1i);
        for &c in all_consts.iter() {
            if c != Zero::zero() {
                test_op!(c / c, _1_0i);
            }
        }
        assert_eq!(_4_2i / 0.5, Complex::new(8.0, 4.0));
    }

    #[test]
    fn test_rem() {
        test_op!(_neg1_1i % _0_1i, _0_0i);
        test_op!(_4_2i % _0_1i, _0_0i);
        test_op!(_05_05i % _0_1i, _05_05i);
        test_op!(_05_05i % _1_1i, _05_05i);
        test_div_rem!(_neg1_1i, _0_1i);
        test_div_rem!(_4_2i, _05_05i);
        assert_eq!((_4_2i + _05_05i) % _0_1i, _05_05i);
        assert_eq!((_4_2i + _05_05i) % _1_1i, _05_05i);
        assert_eq!(_4_2i % 2.0, Complex::new(0.0, 0.0));
        assert_eq!(_4_2i % 3.0, Complex::new(1.0, 2.0));
        assert_eq!(_neg1_1i % 2.0, _neg1_1i);
        assert_eq!(-_4_2i % 3.0, Complex::new(-1.0, -2.0));
    }

    #[test]
    fn test_complex_complex() {
        let cc1 = Complex::new(complex!(1 + 2 i), complex!(3 + 4 i));
        let cc2 = Complex::new(complex!(5 + 6 i), complex!(7 + 8 i));
        assert_eq!(cc1 * cc2, Complex::new(complex!(4 - 36 i), complex!(-18 + 60 i)));
        assert_eq!(cc2 * cc1, Complex::new(complex!(4 - 36 i), complex!(-18 + 60 i)));
        assert_eq!(cc2 * cc1 / cc1, cc2);
        assert_eq!(cc1 * cc2 / cc2, cc1, "{}", cc1 * cc2); // testing Display impl
        assert_eq!(cc1.abs_sqr(), complex!(-10 + 28 i)); // testing Num implementation
    }
}

#[allow(non_upper_case_globals)]
#[cfg(feature = "quaternion")]
mod quaternion {
    use super::*;

    const _0: Quaternion<f64> = Quaternion::new(0.0, 0.0, 0.0, 0.0);
    const _1: Quaternion<f64> = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    const _neg1: Quaternion<f64> = Quaternion::new(-1.0, 0.0, 0.0, 0.0);
    const _2: Quaternion<f64> = Quaternion::new(2.0, 0.0, 0.0, 0.0);
    const _i: Quaternion<f64> = Quaternion::new(0.0, 1.0, 0.0, 0.0);
    const _j: Quaternion<f64> = Quaternion::new(0.0, 0.0, 1.0, 0.0);
    const _k: Quaternion<f64> = Quaternion::new(0.0, 0.0, 0.0, 1.0);
    const _ij: Quaternion<f64> = Quaternion::new(0.0, 1.0, 1.0, 0.0);
    const _jk: Quaternion<f64> = Quaternion::new(0.0, 0.0, 1.0, 1.0);
    const _ik: Quaternion<f64> = Quaternion::new(0.0, 1.0, 0.0, 1.0);
    const _1i: Quaternion<f64> = Quaternion::new(1.0, 1.0, 0.0, 0.0);
    const _1j: Quaternion<f64> = Quaternion::new(1.0, 0.0, 1.0, 0.0);
    const _1k: Quaternion<f64> = Quaternion::new(1.0, 0.0, 0.0, 1.0);
    pub const all_consts: [Quaternion<f64>; 13] =
        [_0, _neg1, _1, _2, _i, _j, _k, _ij, _jk, _ik, _1i, _1j, _1k];

    #[allow(dead_code)]
    fn close(a: Quaternion<f64>, b: Quaternion<f64>) -> bool {
        close_to_tol(a, b, 2e-15, b.abs_sqr())
    }
    #[allow(dead_code)]
    fn close_abs(a: Quaternion<f64>, b: Quaternion<f64>) -> bool {
        let s = |x| {
            if x < 0.0 {
                -1.0
            } else if x > 0.0 {
                1.0
            } else {
                0.0
            }
        };
        close_to_tol(
            a,
            b * s(
                s(s(s(a.re * b.re) + 1e-20 * a.im_i * b.im_i) + 1e-20 * a.im_j * b.im_j)
                    + 1e-20 * a.im_k * b.im_k,
            ),
            2e-15,
            1.0,
        )
    }
    #[allow(dead_code)]
    pub fn close_to_tol(a: Quaternion<f64>, b: Quaternion<f64>, tol: f64, sqr: f64) -> bool {
        // returns true if a and b are reasonably close
        let close = (a == b) || (a - b).abs_sqr() <= tol * tol * sqr;
        #[cfg(feature = "std")]
        if !close {
            std::println!("{:?} != {:?}", a, b);
        }
        close
    }

    #[test]
    fn test_consts() {
        assert_eq!(_0, Zero::zero());
        assert_eq!(_0, quaternion!(0.0f64));
        assert_eq!(_1, One::one());
        assert_eq!(_1, quaternion!(1.0f64));
        assert_eq!(_i, quaternion!(0.0f64 + i 1.0));
        assert_eq!(_j, quaternion!(0.0f64 + j 1.0));
        assert_eq!(_k, quaternion!(0.0f64 + k 1.0));
        assert_eq!(_1i, quaternion!(1.0f64 + i 1.0));
        assert_eq!(_1j, quaternion!(1.0f64 + j 1.0));
        assert_eq!(_1k, quaternion!(1.0f64 + k 1.0));
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    mod analytic {
        use super::*;
        #[test]
        fn test_sqrt1() {
            // if the imaginary value is too small compares with the real value, it will snap to the axis
            let a = Quaternion::new(-1.0, 2.0e-20, 0., 0.);
            assert!((a.sqrt() - Quaternion::new(1.0e-20, 1.0, 0., 0.)).abs() < 0.2e-20, "got {}", a.sqrt());
            let a = Quaternion::new(-4.0, 8.0e-20, 0., 0.);
            assert!((a.sqrt() - Quaternion::new(2.0e-20, 2.0, 0., 0.)).abs() < 0.2e-20);
            let a = Quaternion::new(1.0, 2.0e-20, 0., 0.);
            assert!((a.sqrt() - Quaternion::new(1.0, 1.0e-20, 0., 0.)).abs() < 0.2e-20);
            let a = Quaternion::new(4.0, 8.0e-20, 0., 0.);
            assert!((a.sqrt() - Quaternion::new(2.0, 2.0e-20, 0., 0.)).abs() < 0.2e-20);
            let a = Quaternion::new(-1.0, 2.0e-15, 0., 0.);
            assert!(
                (a.sqrt() - Quaternion::new(1.0e-15, 1.0, 0., 0.)).abs() < 0.5e-15,
                "had {}",
                a.sqrt()
            );
            let a = Quaternion::new(-1.0, 2.0e-16, 0., 0.);
            assert!((a.sqrt() - Quaternion::new(1.0e-16, 1.0, 0., 0.)).abs() < 0.5e-16);
            // im: 0.0
            let a = Quaternion::new(-4.0, 0.0f64, 0., 0.);
            assert!(
                (Quaternion::new(0.0, 2.0, 0., 0.) - a.sqrt()).abs() < 1e-15,
                "error, this is wrong: {}",
                a.sqrt()
            );
            // im: -0.0
            let a = Quaternion::new(-4.0, -0.0f64, 0., 0.);
            assert!(
                (Quaternion::new(0.0, -2.0, 0., 0.) - a.sqrt()).abs() < 1e-15,
                "error, this is wrong: {}",
                a.sqrt()
            );
        }

        #[test]
        fn test_sqrt() {
            assert!(close(_0.sqrt(), _0));
            assert!(close(_1.sqrt(), _1));
            assert!(close(Quaternion::new(-1.0, 0.0, 0.0, 0.0).sqrt(), _i));
            assert!(close(Quaternion::new(-1.0, -0.0, 0.0, 0.0).sqrt(), _i * (-1.0)));
            assert!(close(_i.sqrt(), _1i / (2.0.sqrt())));
            for &c in all_consts.iter() {
                // sqrt(conj(z() = conj(sqrt(z))
                assert!(close(c.conj().sqrt(), c.sqrt().conj()));
                // unit quaternions stay unit quaternions
                if !c.is_zero() {
                    assert!(((c / c.abs()).sqrt().abs() - 1.0).abs() < 1e-8, "{c}");
                    // for this branch, -pi/2 <= arg(sqrt(z)) <= pi/2
                    let [x, y, z] = (c / c.abs()).sqrt().to_axis_angle();
                    let len = (x.abs_sqr() + y.abs_sqr() + z.abs_sqr()).sqrt() / 2.0;
                    assert!(
                        -core::f64::consts::FRAC_PI_2 <= len && len <= core::f64::consts::FRAC_PI_2, "{len}"
                    );
                }
                // sqrt(z) * sqrt(z) = z
                assert!(close(c.sqrt() * c.sqrt(), c));
            }
        }

        #[test]
        fn test_cbrt() {
            assert!(close(_0.cbrt(), _0));
            assert!(close(_1.cbrt(), _1));
            assert!(close(
                _i.cbrt(),
                Quaternion::new(0.75.sqrt(), 0.5, 0.0, 0.0)
            ));
            assert!(close(
                _i.conj().cbrt(),
                Quaternion::new(0.75.sqrt(), -0.5, 0.0, 0.0)
            ));
            for &c in all_consts.iter() {
                // cbrt(conj(z() = conj(cbrt(z))
                assert_eq!(c.conj().cbrt(), c.cbrt().conj());
                // unit quaternions stay unit quaternions
                if !c.is_zero() {
                    assert!(((c / c.abs()).cbrt().abs() - 1.0).abs() < 1e-8);
                    // for this branch, -pi/3 <= arg(cbrt(z)) <= pi/3
                    let [x, y, z] = (c / c.abs()).cbrt().to_axis_angle();
                    let len = (x.abs_sqr() + y.abs_sqr() + z.abs_sqr()).sqrt() / 2.0;
                    assert!(
                        -core::f64::consts::FRAC_PI_3 <= len && len <= core::f64::consts::FRAC_PI_3, "{len}"
                    );
                }
                // cbrt(z) * cbrt(z) cbrt(z) = z
                assert!(close(c.cbrt() * c.cbrt() * c.cbrt(), c));
            }
        }

        #[test]
        fn test_exp_ln() {
            for &c in all_consts.iter() {
                if c.is_zero() {
                    continue;
                }
                // e^ln(z) = z
                assert!(close(c.ln().exp(), c));
            }
        }

        #[test]
        fn test_sin() {
            assert!(close(_0.sin(), _0));
            assert!(close_abs((_1 * core::f64::consts::PI * 2.0).sin(), _0));
            assert!(close(_i.sin(), _i * 1.0.sinh()));
            for &c in all_consts.iter() {
                // sin(conj(z)) = conj(sin(z))
                assert_eq!(c.conj().sin(), c.sin().conj());
                // sin(-z) = -sin(z)
                assert!(close((c * -1.0).sin(), c.sin() * -1.0));
            }
        }

        #[test]
        fn test_cos() {
            assert!(close(_0.cos(), _1));
            assert!(close((_1 * core::f64::consts::PI * 2.0).cos(), _1));
            assert!(close(_i.cos(), _1 * (1.0.cosh())));
            for &c in all_consts.iter() {
                // cos(conj(z)) = conj(cos(z))
                assert_eq!(c.conj().cos(), c.cos().conj());
                // cos(-z) = cos(z)
                assert!(close((c * -1.0).cos(), c.cos()));
            }
        }

        #[test]
        fn test_tan() {
            assert!(close(_0.tan(), _0));
            assert!(close((_1 * core::f64::consts::PI / 4.0).tan(), _1));
            assert!(close_abs((_1 * core::f64::consts::PI).tan(), _0));
            for &c in all_consts.iter() {
                // tan(conj(z)) = conj(tan(z))
                assert_eq!(c.conj().tan(), c.tan().conj());
                // tan(-z) = -tan(z)
                assert!(close((c * -1.0).tan(), c.tan() * (-1.0)));
            }
        }

        #[test]
        fn test_asin() {
            assert_eq!(_0.asin(), _0);
            assert!(close(_1.asin(), _1 * (core::f64::consts::PI / 2.0)));
            assert!(close(
                (_1 * -1.0).asin(),
                _1 * (-core::f64::consts::PI / 2.0)
            ));
            assert!(close(_i.asin(), _i * ((1.0 + 2.0.sqrt()).ln())));
            for &c in all_consts.iter() {
                // asin(conj(z)) = conj(asin(z))
                assert!(close(c.conj().asin(), c.asin().conj()));
                // asin(-z) = -asin(z)
                assert!(close((c * -1.0).asin(), c.asin() * -1.0));
                // for this branch, -pi/2 <= asin(z).re <= pi/2
                assert!(
                    -core::f64::consts::PI / 2.0 <= c.asin().re && c.asin().re <= core::f64::consts::PI / 2.0
                );
            }
        }

        #[test]
        fn test_acos() {
            assert!(close(_0.acos(), _1 * (core::f64::consts::PI / 2.0)));
            assert!(close_abs(_1.acos(), _0));
            assert!(close((_1 * -1.0).acos(), _1 * core::f64::consts::PI));
            assert!(close(
                _i.acos(),
                Quaternion::new(core::f64::consts::PI / 2.0, (2.0.sqrt() - 1.0).ln(), 0.0, 0.0)
            ));
            for &c in all_consts.iter() {
                // acos(conj(z)) = conj(acos(z))
                assert!(close(c.conj().acos(), c.acos().conj()));
                // for this branch, 0 <= acos(z).re <= pi
                assert!(0.0 <= c.acos().re && c.acos().re <= core::f64::consts::PI);
            }
        }

        #[test]
        fn test_atan() {
            assert!(close(_0.atan(), _0));
            assert!(close(_1.atan(), _1 * (core::f64::consts::PI / 4.0)));
            assert!(close(
                (_1 * -1.0).atan(),
                _1 * (-core::f64::consts::PI / 4.0)
            ));
            assert!(close(_i.atan(), Quaternion::new(0.0, f64::INFINITY, 0.0, 0.0)));
            for &c in all_consts.iter() {
                // atan(conj(z)) = conj(atan(z))
                assert!(close(c.conj().atan(), c.atan().conj()));
                // atan(-z) = -atan(z)
                assert!(close((c * -1.0).atan(), c.atan() * -1.0));
                // for this branch, -pi/2 <= atan(z).re <= pi/2
                assert!(
                    -core::f64::consts::PI / 2.0 <= c.atan().re && c.atan().re <= core::f64::consts::PI / 2.0
                );
            }
        }

        #[test]
        fn test_atan2() {
            assert!(close(_0.atan2(&_1), _0));
            assert!(close(_1.atan2(&_1), _1 * (core::f64::consts::PI / 4.0)));
            assert!(close(_1i.atan2(&_1i), _1 * (core::f64::consts::PI / 4.0)));
            assert!(close(_1.atan2(&_0), _1 * (core::f64::consts::PI / 2.0)));
            assert!(close((-_1).atan2(&_0), _1 * (-core::f64::consts::PI / 2.0)));
            assert!(close(_0.atan2(&-_1), _1 * core::f64::consts::PI));
            assert!(close((-_0).atan2(&-_1), _1 * -core::f64::consts::PI));
            assert!(close(_0.atan2(&_0), _0)); // center, default to 0 instead of NaN
            assert!(close(
                (_1 * -1.0).atan2(&_1),
                _1 * (-core::f64::consts::PI / 4.0)
            ));
            assert!(close(_i.atan2(&_1), Quaternion::new(0.0, f64::INFINITY, 0.0, 0.0)));
            assert!(close(_1i.atan2(&_1), _1i.atan()));
        }


        #[test]
        fn test_sinh() {
            assert!(close(_0.sinh(), _0));
            assert!(close(
                _1.sinh(),
                _1 * ((core::f64::consts::E - 1.0 / core::f64::consts::E) / 2.0)
            ));
            assert!(close(_i.sinh(), _i * (1.0.sin())));
            for &c in all_consts.iter() {
                // sinh(conj(z)) = conj(sinh(z))
                assert!(close(c.conj().sinh(), c.sinh().conj()));
                // sinh(-z) = -sinh(z)
                assert!(close((c * -1.0).sinh(), c.sinh() * -1.0));
            }
        }

        #[test]
        fn test_cosh() {
            assert!(close(_0.cosh(), _1));
            assert!(close(
                _1.cosh(),
                _1 * ((core::f64::consts::E + 1.0 / core::f64::consts::E) / 2.0)
            ));
            assert!(close(_i.cosh(), _1 * 1.0.cos()));
            for &c in all_consts.iter() {
                // cosh(conj(z)) = conj(cosh(z))
                assert!(close(c.conj().cosh(), c.cosh().conj()));
                // cosh(-z) = cosh(z)
                assert!(close((c * -1.0).cosh(), c.cosh()));
            }
        }

        #[test]
        fn test_tanh() {
            assert!(close(_0.tanh(), _0));
            assert!(close(
                _1.tanh(),
                _1 * ((core::f64::consts::E.powi(2) - 1.0) / (core::f64::consts::E.powi(2) + 1.0))
            ));
            assert!(close(_i.tanh(), _i * (1.0.tan())));
            for &c in all_consts.iter() {
                // tanh(conj(z)) = conj(tanh(z))
                assert!(close(c.conj().tanh(), c.tanh().conj()));
                // tanh(-z) = -tanh(z)
                assert!(close((c * -1.0).tanh(), c.tanh() * -1.0));
            }
        }

        #[test]
        fn test_asinh() {
            assert!(close(_0.asinh(), _0));
            assert!(close(_1.asinh(), _1 * (1.0 + 2.0.sqrt()).ln()));
            assert!(close(_i.asinh(), _i * (core::f64::consts::PI / 2.0)));
            assert!(close(
                _i.asinh() * -1.0,
                _i * (-core::f64::consts::PI / 2.0)
            ));
            for &c in all_consts.iter() {
                // asinh(conj(z)) = conj(asinh(z))
                assert!(close(c.conj().asinh(), c.asinh().conj()));
                // asinh(-z) = -asinh(z)
                assert!(close((c * -1.0).asinh(), c.asinh() * -1.0));
                // for this branch, -pi/2 <= asinh(z).im <= pi/2
                assert!(
                    -core::f64::consts::PI / 2.0 <= c.asinh().im_i && c.asinh().im_i <= core::f64::consts::PI / 2.0
                );
            }
        }

        #[test]
        fn test_acosh() {
            assert!(close(_0.acosh(), _i * (core::f64::consts::PI / 2.0)));
            assert!(close(_1.acosh(), _0));
            assert!(close(
                (Quaternion::new(-1., 0., 0., 0.)).acosh(),
                _i * core::f64::consts::PI
            ));
            assert!(close((_1 * -1.0).acosh(), -_i * core::f64::consts::PI)); // zero sign is used!
            for &c in all_consts.iter() {
                // acosh(conj(z)) = conj(acosh(z))
                assert!(close(c.conj().acosh(), c.acosh().conj()));
                // for this branch, -pi <= acosh(z).im <= pi and 0 <= acosh(z).re
                assert!(
                    -core::f64::consts::PI <= c.acosh().im_i
                        && c.acosh().im_i <= core::f64::consts::PI
                        && 0.0 <= c.acosh().re
                );
            }
        }

        #[test]
        fn test_atanh() {
            assert!(close(_0.atanh(), _0));
            assert!(close(_i.atanh(), _i * (core::f64::consts::PI / 4.0)));
            assert!(close(_1.atanh(), Quaternion::new(f64::INFINITY, 0.0, 0.0, 0.0)));
            for &c in all_consts.iter() {
                // atanh(conj(z)) = conj(atanh(z))
                assert!(close(c.conj().atanh(), c.atanh().conj()), "{c}");
                // atanh(-z) = -atanh(z)
                assert!(close((-c).atanh(), -c.atanh()), "{c}");
                // for this branch, -pi/2 <= atanh(z).im <= pi/2
                assert!(
                    -core::f64::consts::PI / 2.0 <= c.atanh().im_i && c.atanh().im_i <= core::f64::consts::PI / 2.0
                );
            }
        }

        #[test]
        fn test_trig_to_hyperbolic() {
            for &c in &[-0.5, 0.5, 1.0, 2.0] {
                let c = Quaternion::from(c);
                // sin(iz) = i sinh(z)
                assert!(close((_i * c).sin(), _i * c.sinh()));
                // cos(iz) = cosh(z)
                assert!(close((_i * c).cos(), c.cosh()));
                // tan(iz) = i tanh(z)
                assert!(close((_i * c).tan(), _i * c.tanh()));
                // sinh(iz) = i sin(z)
                assert!(close((_i * c).sinh(), _i * c.sin()));
                // cosh(iz) = cos(z)
                assert!(close((_i * c).cosh(), c.cos()));
                // tanh(iz) = i tan(z)
                assert!(close((_i * c).tanh(), _i * c.tan()));
            }
        }

        #[test]
        fn test_trig_identities() {
            for &c in all_consts.iter() {
                // tan(z) = sin(z)/cos(z)
                assert!(close(c.tan(), c.sin() / c.cos()));
                // tan(z) = sin(z)/cos(z)
                assert!(close(c.tan(), &c.sin() / &c.cos()));
                // sin(z)^2 + cos(z)^2 = 1
                assert!(close(c.sin() * c.sin() + c.cos() * c.cos(), _1));

                // asin(sin(z)) = z
                assert!(close(c.sin().asin().sin(), c.sin()), "with {}", c.sin());
                assert!(close(c.asin().sin(), c), "with {}", c.asin());
                // acos(cos(z)) = z
                assert!(
                    close_abs(c.cos().acos().cos(), c.cos()),
                    "with {c} -> {} and {}",
                    c.cos(),
                    c.cos().acos()
                );
                assert!(close_abs(c.acos().cos(), c));
                // atan(tan(z)) = z
                assert!(close(c.tan().atan().tan(), c.tan()), "with {}", c.tan());
                if c.im_abs() != 1.0 {
                    assert!(close(c.atan().tan(), c));
                }

                let i = c.sign_im();
                // sin(z) = (e^(iz) - e^(-iz))/(2i)
                assert!(close(
                    ((i * c).exp() - (i * c).exp().recip()) / (i * 2.0),
                    c.sin()
                ));
                // cos(z) = (e^(iz) + e^(-iz))/2
                assert!(close(
                    ((i * c).exp() + (i * c).exp().recip()) / 2.0,
                    c.cos()
                ));
                // tan(z) = i (1 - e^(2iz))/(1 + e^(2iz))
                assert!(close(
                    i * (_1 - (i * c * 2.0).exp()) / (_1 + (i * c * 2.0).exp()),
                    c.tan()
                ));
            }
        }

        #[test]
        fn test_div2() {
            let p = quaternion!(0.6349639147847361 + i 1.2984575814159773);
            let q = quaternion!(0.8337300251311491 + i 0.9888977057628651);
            let p2 = quaternion!(1. + i 1.).sinh();
            let q2 = quaternion!(1. + i 1.).cosh();
            // only close, because libm has different results from std.
            assert!(close(p2, p));
            assert!(close(q2, q));
            assert!(close(p / q, p2 / q2));
        }

        #[test]
        fn test_hyperbolic_identites() {
            for &c in all_consts.iter() {
                // tanh(z) = sinh(z)/cosh(z)
                assert!(close(c.tanh(), c.sinh() / c.cosh()), "for {c}");
                // cosh(z)^2 - sinh(z)^2 = 1
                assert!(close(c.cosh() * c.cosh() - c.sinh() * c.sinh(), _1));

                // sinh(asinh(z)) = z
                assert!(close(c.asinh().sinh(), c));
                // cosh(acosh(z)) = z
                assert!(close_abs(c.acosh().cosh(), c));
                // tanh(atanh(z)) = z
                // 1 and -1 are branch points
                if c.im_abs() != 1.0 {
                    assert!(close(c.atanh().tanh(), c));
                }

                // sinh(z) = (e^z - e^(-z))/2
                assert!(close((c.exp() - c.exp().recip()) / 2.0, c.sinh()));
                // cosh(z) = (e^z + e^(-z))/2
                assert!(close((c.exp() + c.exp().recip()) / 2.0, c.cosh()));
                // tanh(z) = ( e^(2z) - 1)/(e^(2z) + 1)
                assert!(close(
                    ((c * 2.0).exp() - _1) / ((c * 2.0).exp() + _1),
                    c.tanh()
                ));
            }
        }
    }

    #[test]
    fn test_zero() {
        let zero = Quaternion::<f64>::zero();
        assert!(zero.is_zero());

        let mut c = Quaternion::new(1.23, 4.56, 0.0, 0.0);
        assert!(!c.is_zero());
        assert_eq!(c + zero, c);

        c.set_zero();
        assert!(c.is_zero());
    }

    #[test]
    fn test_one() {
        let one = Quaternion::<f64>::one();
        assert!(one.is_one());

        let mut c = Quaternion::new(1.23, 4.56, 0.0, 0.0);
        assert!(!c.is_one());
        assert_eq!(c * one, c);

        c.set_one();
        assert!(c.is_one());
    }

    #[test]
    fn test_sum() {
        let v = [_i, _1];
        assert_eq!(v.iter().sum::<Quaternion<f64>>(), _1i);
        assert_eq!(v.into_iter().sum::<Quaternion<f64>>(), _1i);
        let v = [];
        assert_eq!(v.iter().sum::<Quaternion<f64>>(), _0);
        assert_eq!(v.into_iter().sum::<Quaternion<f64>>(), _0);
    }

    #[test]
    fn test_prod() {
        let v = [_i, _1];
        assert_eq!(v.iter().product::<Quaternion<f64>>(), _i);
        assert_eq!(v.into_iter().product::<Quaternion<f64>>(), _i);
        let v = [];
        assert_eq!(v.iter().product::<Quaternion<f64>>(), _1);
        assert_eq!(v.into_iter().product::<Quaternion<f64>>(), _1);
    }

    #[test]
    fn test_div() {
        let p = _i - _1;
        let q = _i * 2.0;
        test_op!(p / q, _1i * 0.5);
        for &c in all_consts.iter() {
            if c != Zero::zero() {
                test_op!(c / c, _1);
            }
            let p = c * _ij;
            test_op!(p / _ij, c);
        }
    }

    #[test]
    fn test_spinors() {
        use crate::quaternion as q;
        // 32 component object
        pub type DiracSpinor<T> = Quaternion<Quaternion<Complex<T>>>;

        // define the gamma matrices
        let _0c = complex!(0.+0. i);
        let _1c = complex!(1.+0. i);
        let _ic = complex!(0.+1. i);
        let id: DiracSpinor<f64> = q!(q!(_1c));
        let g0: DiracSpinor<f64> = q!((q!(_0c)) + k (q!(_ic)));
        let g1: DiracSpinor<f64> = q!((q!(_0c)) + j (q!(_0c + i (-_ic))));
        let g2: DiracSpinor<f64> = q!((q!(_0c)) + j (q!(_0c + j (-_ic))));
        let g3: DiracSpinor<f64> = q!((q!(_0c)) + j (q!(_0c + k (-_ic))));
        let g5: DiracSpinor<f64> = q!((q!(_0c)) + i (q!(_ic)));

        assert_ne!(g0, g1);
        assert_ne!(g0, g2);
        assert_ne!(g0, g3);
        assert_ne!(g0, g5);
        assert_ne!(g1, g2);
        assert_ne!(g1, g3);
        assert_ne!(g1, g5);
        assert_ne!(g2, g3);
        assert_ne!(g2, g5);
        assert_ne!(g3, g5);

        assert_eq!(g0 * g0, id);
        assert_eq!(g1 * g1, -id);
        assert_eq!(g2 * g2, -id);
        assert_eq!(g3 * g3, -id);
        assert_eq!(g5 * g5, id);
        assert_eq!(g0 * g1, -g1 * g0);
        assert_eq!(g0 * g2, -g2 * g0);
        assert_eq!(g0 * g3, -g3 * g0);
        assert_eq!(g2 * g1, -g1 * g2);
        assert_eq!(g3 * g2, -g2 * g3);
        assert_eq!(g1 * g3, -g3 * g1);
        assert_eq!(g5 * g0, -g0 * g5);
        assert_eq!(g5 * g1, -g1 * g5);
        assert_eq!(g5 * g2, -g2 * g5);
        assert_eq!(g5 * g3, -g3 * g5);
        assert_eq!(g0 * g1 * g2 * g3 * q!(_ic), g5);
        let a = g0 + g2;
        let b = g1 + g3;
        let c = g1 + g3 - g5;
        assert_eq!(a * b / b, a);
        //assert_eq!(b * a / a, b); // can not divide by a, as it's not invertible! (determinant is zero!)
        assert_eq!(b * c / c, b);
        assert_eq!(a.abs_sqr().abs_sqr().abs_sqr(), 0.0);
        assert_eq!(b.abs_sqr().abs_sqr(), 4.0.into()); // this ends up being the determinant
        assert_eq!(c.abs_sqr().abs_sqr(), 1.0.into());

        // unreadable, but still pretty
        assert_fmt_eq!(format_args!("{:#}", g0), "ik");
        assert_fmt_eq!(format_args!("{:#}", g1), "-iij");
        assert_fmt_eq!(format_args!("{:#}", g2), "-ijj");
        assert_fmt_eq!(format_args!("{:#}", g3), "-ikj");
        assert_fmt_eq!(format_args!("{:#}", g5), "ii");
    }

    #[test]
    fn test_string_formatting() {
        assert_fmt_eq!(format_args!("{}", _0), "0+0i+0j+0k");
        assert_fmt_eq!(format_args!("{}", _1), "1+0i+0j+0k");
        assert_fmt_eq!(format_args!("{}", _i), "0+1i+0j+0k");
        assert_fmt_eq!(format_args!("{}", _1i), "1+1i+0j+0k");
        assert_fmt_eq!(format_args!("{}", _neg1), "-1+0i+0j+0k");
        assert_fmt_eq!(format_args!("{}", -_i), "-0-1i-0j-0k");
        assert_fmt_eq!(format_args!("{}", -_j), "-0-0i-1j-0k");
        assert_fmt_eq!(format_args!("{}", -_jk), "-0-0i-1j-1k");
        assert_fmt_eq!(format_args!("{}", _ik/2.0), "0+0.5i+0j+0.5k");
        // pretty printing
        assert_fmt_eq!(format_args!("{:#}", _0), "0");
        assert_fmt_eq!(format_args!("{:#}", _1), "1");
        assert_fmt_eq!(format_args!("{:#}", _i), "i");
        assert_fmt_eq!(format_args!("{:#}", _1i), "1+i");
        assert_fmt_eq!(format_args!("{:#}", _neg1), "-1");
        assert_fmt_eq!(format_args!("{:#}", -_i), "-i");
        assert_fmt_eq!(format_args!("{:#}", -_j), "-j");
        assert_fmt_eq!(format_args!("{:#}", -_jk), "-j-k");
        assert_fmt_eq!(format_args!("{:#}", _ik/2.0), "0.5i+0.5k");

        let a = Quaternion::new(1.23456, 123.456, 12., 12.);
        assert_fmt_eq!(format_args!("{}", a), "1.23456+123.456i+12j+12k");
        assert_fmt_eq!(format_args!("{:.2}", a), "1.23+123.46i+12.00j+12.00k");
        assert_fmt_eq!(format_args!("{:.2e}", a), "1.23e0+1.23e2i+1.20e1j+1.20e1k");
        assert_fmt_eq!(format_args!("{:+.2E}", a), "+1.23E0+1.23E2i+1.20E1j+1.20E1k");
        assert_fmt_eq!(format_args!("{:+36.2E}", a), "     +1.23E0+1.23E2i+1.20E1j+1.20E1k");

        let b = Quaternion::new(0x80, 0xff, 0xf, 0xA);
        assert_fmt_eq!(format_args!("{:X}", b), "80+FFi+Fj+Ak");
        assert_fmt_eq!(format_args!("{:#x}", b), "0x80+0xffi+0xfj+0xak");
        assert_fmt_eq!(format_args!("{:+#b}", b), "+0b10000000+0b11111111i+0b1111j+0b1010k");
        assert_fmt_eq!(format_args!("{:+#o}", b), "+0o200+0o377i+0o17j+0o12k");
        assert_fmt_eq!(format_args!("{:+#28o}", b), "   +0o200+0o377i+0o17j+0o12k");
        assert_fmt_eq!(format_args!("{:<+#28o}", b), "+0o200+0o377i+0o17j+0o12k   ");
        assert_fmt_eq!(format_args!("{:^+#29o}", b), "  +0o200+0o377i+0o17j+0o12k  ");

        let c = Quaternion::new(-10, -10000, 0, 0);
        assert_fmt_eq!(format_args!("{}", c), "-10-10000i+0j+0k");
        assert_fmt_eq!(format_args!("{:22}", c), "      -10-10000i+0j+0k");
        assert_fmt_eq!(format_args!("{:->22}", c), "-------10-10000i+0j+0k");
    }
}

#[cfg(feature = "std")]
#[allow(dead_code)]
fn hash<T: std::hash::Hash>(x: &T) -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::*;
    let mut hasher = <RandomState as BuildHasher>::Hasher::new();
    x.hash(&mut hasher);
    hasher.finish()
}

#[cfg(feature = "rational")]
mod rational {
    use super::*;
    #[cfg(feature = "ibig")]
    use ibig::{IBig as IBig, UBig as UBig}; // note that choosing the names makes it library agnostic.
    #[cfg(feature = "ibig")]
    type BigRational = Ratio<IBig>;
    use crate::rational::Ratio;
    type Rational64 = Ratio<i64>;

    pub const _0: Rational64 = Ratio::new_raw(0, 1);
    pub const _1: Rational64 = Ratio::new_raw(1, 1);
    pub const _2: Rational64 = Ratio::new_raw(2, 1);
    pub const _NEG2: Rational64 = Ratio::new_raw(-2, 1);
    pub const _8: Rational64 = Ratio::new_raw(8, 1);
    pub const _15: Rational64 = Ratio::new_raw(15, 1);
    pub const _16: Rational64 = Ratio::new_raw(16, 1);

    pub const _1_2: Rational64 = Ratio::new_raw(1, 2);
    pub const _1_8: Rational64 = Ratio::new_raw(1, 8);
    pub const _1_15: Rational64 = Ratio::new_raw(1, 15);
    pub const _1_16: Rational64 = Ratio::new_raw(1, 16);
    pub const _3_2: Rational64 = Ratio::new_raw(3, 2);
    pub const _5_2: Rational64 = Ratio::new_raw(5, 2);
    pub const _NEG1_2: Rational64 = Ratio::new_raw(-1, 2);
    pub const _1_NEG2: Rational64 = Ratio::new_raw(1, -2);
    pub const _NEG1_NEG2: Rational64 = Ratio::new_raw(-1, -2);
    pub const _1_3: Rational64 = Ratio::new_raw(1, 3);
    pub const _NEG1_3: Rational64 = Ratio::new_raw(-1, 3);
    pub const _1_NEG3: Rational64 = Ratio::new_raw(1, -3);
    pub const _2_3: Rational64 = Ratio::new_raw(2, 3);
    pub const _NEG2_3: Rational64 = Ratio::new_raw(-2, 3);
    pub const _2_NEG3: Rational64 = Ratio::new_raw(2, -3);
    pub const _MIN: Rational64 = Ratio::new_raw(i64::MIN, 1);
    pub const _MIN_P1: Rational64 = Ratio::new_raw(i64::MIN + 1, 1);
    pub const _MAX: Rational64 = Ratio::new_raw(i64::MAX, 1);
    pub const _MAX_M1: Rational64 = Ratio::new_raw(i64::MAX - 1, 1);
    pub const _BILLION: Rational64 = Ratio::new_raw(1_000_000_000, 1);
    pub const _NAN: Rational64 = Ratio::new_raw(0, 0);
    pub const _INF: Rational64 = Ratio::new_raw(1, 0);
    pub const _NEG_INF: Rational64 = Ratio::new_raw(-1, 0);

    #[cfg(feature = "ibig")]
    pub fn to_big(n: Rational64) -> BigRational {
        Ratio::new_raw(
            IBig::from(n.numer),
            IBig::from(n.denom),
        )
    }

    #[test]
    fn test_test_constants() {
        // check our constants are what Ratio::new etc. would make.
        assert_eq!(_0, Zero::zero());
        assert_eq!(_1, One::one());
        assert_eq!(_2, Ratio::from(2));
        assert_eq!(_1_2, Ratio::new(1, 2));
        assert_eq!(_3_2, Ratio::new(3, 2));
        assert_eq!(_NEG1_2, Ratio::new(-1, 2));
        assert_eq!(_2, From::from(2));
        assert_eq!(_1_2, Ratio::new(_1, _2).trunc().numer);
        let r = Ratio::new(Complex::new(1, 1), Complex::new(1, -1)); // = 1/(-i) = i
        let r2 = Ratio::new(Complex::imag(1), Complex::real(1)); // = i
        let r3: (_, _) = r2.into();
        assert_eq!(r3, r.reduced_full().into());
        assert_eq!(r, r2);
        let r = Ratio::new(Complex::new(1, 2), Complex::new(1, -1));
        let r2 = Ratio::new(Complex::new(-1, 3), Complex::real(2)); // = (-1+3i)/2 but represented differently, as 2 is not prime
        assert_eq!(r, r2);
        assert_eq!(r2.re(), -_1_2);
        assert_eq!((r2 * Complex::i()).re(), -_3_2);
    }

    #[test]
    fn test_new_reduce() {
        assert_eq!(Ratio::new(2, 2), One::one());
        //assert_eq!(Ratio::new(0, i32::MIN), Zero::zero()); // can't work with the general trait bounds
        assert_eq!(Ratio::new(0, -i32::MAX), Zero::zero());
        assert_eq!(Ratio::new(i32::MIN, i32::MIN), One::one());
    }
    #[test]
    fn test_new_zero() {
        assert_eq!(Ratio::new(1, 0), Ratio::new_raw(1, 0));
        assert_eq!(Ratio::new(-1, 0), Ratio::new_raw(-1, 0));
    }

    #[test]
    fn test_num_complex() {
        assert_eq!(_1_2.re(), _1_2);
        let r = Ratio::new(complex!(1 + 2 i), complex!(-2 + 3 i));
        assert_eq!(r.re(), Ratio::new((r.numer * r.denom.conj()).re(), r.denom.abs_sqr()));
    }

    #[test]
    fn test_approx_float() {
        assert_eq!(
            Ratio::<i32>::from_approx(core::f32::consts::PI, 3e-7),
            Some(Ratio::new(355, 113))
        );
        assert_eq!(
            Ratio::<i64>::from_approx(core::f64::consts::PI, 3e-7),
            Some(Ratio::new(355, 113))
        );
        assert_eq!(
            Ratio::<i32>::from_approx(core::f32::consts::E, 3e-7),
            Some(Ratio::new(2721, 1001))
        );
        assert_eq!(
            Ratio::<i64>::from_approx(core::f64::consts::E, 3e-7),
            Some(Ratio::new(2721, 1001))
        );
        assert_eq!(
            Ratio::<i64>::from_approx(f32::MAX as f64, 1e-7 * f32::MAX as f64),
            None
        );
        assert_eq!(
            Ratio::<i128>::from_approx(f32::MAX as f64, 1e-7 * f32::MAX as f64),
            Ratio::try_from(f32::MAX).ok()
        );
        assert_eq!(
            Ratio::<i64>::from_approx(f64::INFINITY, 1e-7),
            Ratio::try_from(f32::INFINITY).ok()
        );
        assert_eq!(
            Ratio::<i64>::from_approx(f64::NEG_INFINITY, 1e-7),
            Ratio::try_from(f32::NEG_INFINITY).ok()
        );
        assert_eq!(
            Ratio::<i64>::from_approx(f64::NAN, 1e-7),
            Ratio::try_from(f32::NAN).ok()
        );
        // Note, these differ from the try_from representations.
        assert_eq!(
            core::f64::consts::PI,
            Ratio::<i128>::from_approx(core::f64::consts::PI, 0.0)
                .unwrap()
                .to_approx()
        );
        assert_eq!(
            core::f64::consts::E,
            Ratio::<i128>::from_approx(core::f64::consts::E, 0.0)
                .unwrap()
                .to_approx()
        );
        assert_eq!(
            core::f32::consts::PI,
            Ratio::<i64>::from_approx(core::f32::consts::PI, 0.0)
                .unwrap()
                .to_approx()
        );
        assert_eq!(
            core::f32::consts::E,
            Ratio::<i64>::from_approx(core::f32::consts::E, 0.0)
                .unwrap()
                .to_approx()
        );
    }

    #[test]
    #[allow(clippy::eq_op)]
    fn test_cmp() {
        use core::cmp::Ordering;

        assert!(_0 == -_0);
        assert!(_0 == _0 && _1 == _1);
        assert!(_0 != _1 && _1 != _0);
        assert!(_0 < _1 && !(_1 < _0));
        assert!(_1 > _0 && !(_0 > _1));
        assert_eq!(Ratio::new_raw(-1, -1), _1);
        assert_eq!(_1, Ratio::new_raw(-1, -1));
        assert_eq!(Ratio::new_raw(-1, -2), _1_2);
        assert_eq!(_1_2, Ratio::new_raw(-1, -2));
        assert_eq!(Ratio::new_raw(-1, -2).partial_cmp(&_1_2), Some(Ordering::Equal));

        assert!(_0 <= _0 && _1 <= _1);
        assert!(_0 <= _1 && !(_1 <= _0));

        assert!(_0 >= _0 && _1 >= _1);
        assert!(_1 >= _0 && !(_0 >= _1));

        let _0_2: Rational64 = Ratio::new_raw(0, 2);
        assert_eq!(_0, _0_2);

        // infinities and NaNs
        assert!(_0 <= (_1 / _0));
        assert!(Ratio::new_raw(0, -1) <= (_1 / _0));
        assert!(_0 >= ((-_1) / _0));
        assert!(((-_1) / _0) < (_1 / _0));
        assert!((_1 / _0) > ((-_1) / _0));
        assert_eq!((_2 / _0).partial_cmp(&(_1 / _0)), Some(Ordering::Equal));
        assert_eq!(
            Ratio::new_raw(2, 0).partial_cmp(&Ratio::new_raw(1, 0)),
            Some(Ordering::Equal)
        );
        assert_eq!(
            Ratio::new_raw(2, 0).partial_cmp(&Ratio::new_raw(-1, 0)),
            Some(Ordering::Greater)
        );
        assert!(!(_0 / _0).is_finite());
        assert!((_0 / _0).is_nan());
        assert_eq!((_0).partial_cmp(&(_0 / _0)), None);
        assert_eq!((_0 / _0).partial_cmp(&_0), None);
        // PartialEq with different infinities
        assert!(Ratio::new_raw(2, 0) == (_1 / _0));
        assert!((_1 / _0) != (-_1 / _0));
        assert!((_1 / _0) != (_0 / _0));
    }

    #[test]
    fn test_cmp_overflow() {
        use core::cmp::Ordering;

        // issue #7 example:
        let big = Ratio::new(128u8, 1);
        let small = big.recip();
        assert!(big > small);

        // try a few that are closer together
        // (some matching numer, some matching denom, some neither)
        let ratios = [
            Ratio::new(125_i8, 127_i8),
            Ratio::new(63_i8, 64_i8),
            Ratio::new(124_i8, 125_i8),
            Ratio::new(125_i8, 126_i8),
            Ratio::new(126_i8, 127_i8),
            Ratio::new(127_i8, 126_i8),
        ];

        fn check_cmp(a: Ratio<i8>, b: Ratio<i8>, ord: Ordering) {
            //std::println!("comparing {} and {}", a, b);
            assert_eq!(a.partial_cmp(&b), Some(ord));
            assert_eq!(b.partial_cmp(&a), Some(ord.reverse()));
            assert_eq!(a == b, ord == Ordering::Equal);
            assert_eq!(a != b, ord != Ordering::Equal);
            assert_eq!(b == a, ord == Ordering::Equal);
            assert_eq!(b != a, ord != Ordering::Equal);
        }

        for (i, &a) in ratios.iter().enumerate() {
            check_cmp(a, a, Ordering::Equal);
            check_cmp(-a, a, Ordering::Less);
            for &b in &ratios[i + 1..] {
                check_cmp(a, b, Ordering::Less);
                check_cmp(-a, -b, Ordering::Greater);
                check_cmp(a.recip(), b.recip(), Ordering::Greater);
                check_cmp(-a.recip(), -b.recip(), Ordering::Less);
            }
        }

        #[cfg(feature = "ibig")]
        {
            // test recursion limits
            let a = BigRational::new(
                "2848091240477484913831".parse().unwrap(),
                "1347298747461876457091".parse().unwrap(),
            );
            let b = BigRational::new(
                "2848091240477484913832".parse().unwrap(),
                "1347298747461876457092".parse().unwrap(),
            );
            assert!(a > b);
            assert!(a != b);
            let a = BigRational::new(
                "284809124001480758165698749156931477484913831"
                    .parse()
                    .unwrap(),
                "134729874746012938570101240147041441876457091"
                    .parse()
                    .unwrap(),
            );
            let b = BigRational::new(
                "284809124001480758165698749156931477484913832"
                    .parse()
                    .unwrap(),
                "134729874746012938570101240147041441876457092"
                    .parse()
                    .unwrap(),
            );
            assert!(a > b);
            assert!(a != b);
            let a = BigRational::new(
                "28480912408140985865921964982184619846901480758165698749156931477484913831"
                    .parse()
                    .unwrap(),
                "13472987474601293857019187419865891841240017345975201240147041441876457091"
                    .parse()
                    .unwrap(),
            );
            let b = BigRational::new(
                "28480912408140985865921964982184619846901480758165698749156931477484913832"
                    .parse()
                    .unwrap(),
                "13472987474601293857019187419865891841240017345975201240147041441876457092"
                    .parse()
                    .unwrap(),
            );
            assert!(a > b);
            assert!(a != b);
            // this next one is already 115 recursive calls deep
            let a = BigRational::new("28480912408140985865921964982184019830491750173619460128640183640194710871024710484619846901480758165698749156931477484913831".parse().unwrap(), "13472987474601293857019187419865891841240017345975201240147041247108374918659764818204710348765103865981732091731441876457091".parse().unwrap());
            let b = Ratio::new_raw(&a.numer + &IBig::one(), &a.denom + &IBig::one());
            assert!(a > b);
            assert!(a != b);
            // next: 184 recursive calls deep
            let a = BigRational::new("28480912408140985865921964982184019830491750173619460128640183640110498471864871634871264917298419374547129369126471546537312391294687154894710871024710484619846901480758165698749156931477484913831".parse().unwrap(), "13472987474601293857019187419865891841240017345975201240147041247108374918129848913648716481270321798476735481146918349817439216351940424141246876659764818204710348765103865981732091731441876457091".parse().unwrap());
            let b = Ratio::new_raw(&a.numer + &IBig::one(), &a.denom + &IBig::one());
            assert!(a > b);
            assert!(a != b);
            // 390
            let a = &a * &a;
            let b = Ratio::new_raw(&a.numer + &IBig::one(), &a.denom + &IBig::one());
            assert!(a > b);
            assert!(a != b);
            // 728
            let a = &a * &a;
            let b = Ratio::new_raw(&a.numer + &IBig::one(), &a.denom + &IBig::one());
            assert!(a > b);
            assert!(a != b);
            // probably ~1300
            let a = &a * &a;
            let b = Ratio::new_raw(&a.numer + &IBig::one(), &a.denom + &IBig::one());
            assert!(a > b);
            assert!(a != b);
        }
    }

    #[test]
    fn test_to_integer() {
        assert_eq!(_0.trunc().numer, 0);
        assert_eq!(_1.trunc().numer, 1);
        assert_eq!(_2.trunc().numer, 2);
        assert_eq!(_1_2.trunc().numer, 0);
        assert_eq!(_3_2.trunc().numer, 1);
        assert_eq!(_NEG1_2.trunc().numer, 0);
    }

    #[test]
    fn test_numer() {
        assert_eq!(_0.numer, 0);
        assert_eq!(_1.numer, 1);
        assert_eq!(_2.numer, 2);
        assert_eq!(_1_2.numer, 1);
        assert_eq!(_3_2.numer, 3);
        assert_eq!(_NEG1_2.numer, (-1));
    }
    #[test]
    fn test_denom() {
        assert_eq!(_0.denom, 1);
        assert_eq!(_1.denom, 1);
        assert_eq!(_2.denom, 1);
        assert_eq!(_1_2.denom, 2);
        assert_eq!(_3_2.denom, 2);
        assert_eq!(_NEG1_2.denom, 2);
    }

    #[test]
    fn test_is_integer() {
        assert!(_0.is_integral());
        assert!(_1.is_integral());
        assert!(_2.is_integral());
        assert!(!_1_2.is_integral());
        assert!(!_3_2.is_integral());
        assert!(!_NEG1_2.is_integral());
    }

    #[test]
    fn test_show() {
        // Test:
        // :b :o :x, :X, :?
        // alternate or not (#)
        // positive and negative
        // padding, alignment, precision

        // Note, that no_std only supports the normal form a/b and
        // will not (isn't able to) detect, when to add parenthesis.
        // However especially in a no_std environment, the simple form
        // is all, that is usually needed. If that isn't clear enough,
        // one can still use the debug output.
        assert_fmt_eq!(format_args!("{}", _2), "2");
        assert_fmt_eq!(format_args!("{:+}", _2), "+2");
        assert_fmt_eq!(format_args!("{:-}", _2), "2");
        assert_fmt_eq!(format_args!("{}", _1_2), "1/2");
        assert_fmt_eq!(format_args!("{}", -_1_2), "-1/2"); // test negatives
        assert_fmt_eq!(format_args!("{}", _0), "0");
        assert_fmt_eq!(format_args!("{}", -_2), "-2");
        assert_fmt_eq!(format_args!("{:+}", -_2), "-2");
        assert_fmt_eq!(format_args!("{:b}", _2), "10");
        assert_fmt_eq!(format_args!("{:#b}", _2), "0b10");
        assert_fmt_eq!(format_args!("{:b}", _1_2), "1/10");
        assert_fmt_eq!(format_args!("{:+b}", _1_2), "+1/10");
        assert_fmt_eq!(format_args!("{:-b}", _1_2), "1/10");
        assert_fmt_eq!(format_args!("{:b}", _0), "0");
        assert_fmt_eq!(format_args!("{:#b}", _1_2), "0b1/0b10");
        assert_fmt_eq!(format_args!("{:10b}", _1_2), "      1/10");
        assert_fmt_eq!(format_args!("{:->10b}", _1_2), "------1/10");
        assert_fmt_eq!(format_args!("{:#10b}", _1_2), "  0b1/0b10");
        assert_fmt_eq!(format_args!("{:010b}", _1_2), "0000001/10");
        assert_fmt_eq!(format_args!("{:#010b}", _1_2), "0b001/0b10");
        let half_i8: Ratio<i8> = Ratio::new(1_i8, 2_i8);
        assert_fmt_eq!(format_args!("{:b}", -half_i8), "11111111/10");
        assert_fmt_eq!(format_args!("{:#b}", -half_i8), "0b11111111/0b10");
        assert_fmt_eq!(format_args!("{:05}", Ratio::new(-1_i8, 1_i8)), "-0001");
        assert_fmt_eq!(format_args!("{:5}", Ratio::new(-1_i8, 1_i8)), "   -1");
        assert_fmt_eq!(format_args!("{:<5}", Ratio::new(-1_i8, 1_i8)), "-1   ");
        assert_fmt_eq!(format_args!("{:^5}", Ratio::new(-1_i8, 1_i8)), " -1  ");

        assert_fmt_eq!(format_args!("{:o}", _8), "10");
        assert_fmt_eq!(format_args!("{:o}", _1_8), "1/10");
        assert_fmt_eq!(format_args!("{:o}", _0), "0");
        assert_fmt_eq!(format_args!("{:#o}", _1_8), "0o1/0o10");
        assert_fmt_eq!(format_args!("{:10o}", _1_8), "      1/10");
        assert_fmt_eq!(format_args!("{:#10o}", _1_8), "  0o1/0o10");
        assert_fmt_eq!(format_args!("{:010o}", _1_8), "0000001/10");
        assert_fmt_eq!(format_args!("{:#010o}", _1_8), "0o001/0o10");
        assert_fmt_eq!(format_args!("{:o}", -half_i8), "377/2");
        assert_fmt_eq!(format_args!("{:#o}", -half_i8), "0o377/0o2");

        assert_fmt_eq!(format_args!("{:x}", _16), "10");
        assert_fmt_eq!(format_args!("{:x}", _15), "f");
        assert_fmt_eq!(format_args!("{:x}", _1_16), "1/10");
        assert_fmt_eq!(format_args!("{:x}", _1_15), "1/f");
        assert_fmt_eq!(format_args!("{:x}", _0), "0");
        assert_fmt_eq!(format_args!("{:#x}", _1_16), "0x1/0x10");
        assert_fmt_eq!(format_args!("{:010x}", _1_16), "0000001/10");
        assert_fmt_eq!(format_args!("{:#010x}", _1_16), "0x001/0x10");
        assert_fmt_eq!(format_args!("{:x}", -half_i8), "ff/2");
        assert_fmt_eq!(format_args!("{:#x}", -half_i8), "0xff/0x2");

        assert_fmt_eq!(format_args!("{:X}", _16), "10");
        assert_fmt_eq!(format_args!("{:X}", _15), "F");
        assert_fmt_eq!(format_args!("{:X}", _1_16), "1/10");
        assert_fmt_eq!(format_args!("{:X}", _1_15), "1/F");
        assert_fmt_eq!(format_args!("{:X}", _0), "0");
        assert_fmt_eq!(format_args!("{:#X}", _1_16), "0x1/0x10");
        assert_fmt_eq!(format_args!("{:010X}", _1_16), "0000001/10");
        assert_fmt_eq!(format_args!("{:#010X}", _1_16), "0x001/0x10");
        assert_fmt_eq!(format_args!("{:X}", -half_i8), "FF/2");
        assert_fmt_eq!(format_args!("{:#X}", -half_i8), "0xFF/0x2");

        assert_fmt_eq!(format_args!("{:.2e}", -_2), "-2.00e0");
        assert_fmt_eq!(format_args!("{:#.2e}", -_2), "-2.00e0");
        assert_fmt_eq!(format_args!("{:+.2e}", -_2), "-2.00e0");
        assert_fmt_eq!(format_args!("{:e}", _BILLION), "1e9");
        assert_fmt_eq!(format_args!("{:+e}", _BILLION), "+1e9");
        assert_fmt_eq!(format_args!("{:.2e}", _BILLION.recip()), "1.00e0/1.00e9");
        assert_fmt_eq!(format_args!("{:+.2e}", _BILLION.recip()), "+1.00e0/1.00e9");

        assert_fmt_eq!(format_args!("{:.2E}", -_2), "-2.00E0");
        assert_fmt_eq!(format_args!("{:#E}", -_2), "-2E0");
        assert_fmt_eq!(format_args!("{:+E}", -_2), "-2E0");
        assert_fmt_eq!(format_args!("{:E}", _BILLION), "1E9");
        assert_fmt_eq!(format_args!("{:+E}", _BILLION), "+1E9");
        assert_fmt_eq!(format_args!("{:E}", _BILLION.recip()), "1E0/1E9");
        assert_fmt_eq!(format_args!("{:+.2E}", _BILLION.recip()), "+1.00E0/1.00E9");
        assert_fmt_eq!(format_args!("{}", _NAN), "NaN");
        assert_fmt_eq!(format_args!("{}", _INF), "∞");
        assert_fmt_eq!(format_args!("{}", _NEG_INF), "-∞");
        assert_fmt_eq!(format_args!("{}", Ratio::new_raw(2, 0)), "2∞");
        assert_fmt_eq!(format_args!("{:^7}", Ratio::new_raw(2, 0)), "  2∞   "); // centering with non unicode character
        assert_fmt_eq!(format_args!("{}", Ratio::new_raw(-2, 0)), "-2∞");
        // test type combinations
        assert_fmt_eq!(format_args!("{}", Ratio::new_raw(complex!(1 + 2 i), complex!(1 - 2 i))), "(1+2i)/(1-2i)");
        let c = complex!((_1_3) + (_1_8) i);
        assert_fmt_eq!(format_args!("{}", c), "1/3+(1/8)i");
        assert_fmt_eq!(format_args!("{:+}", c), "+1/3+(1/8)i");
        assert_fmt_eq!(format_args!("{}", complex!((c) + (c) i)), "1/3+(1/8)i+(1/3+(1/8)i)i");
        let c = complex!((_1) + (_1) i);
        assert_fmt_eq!(format_args!("{:}", c), "1+1i");
        assert_fmt_eq!(format_args!("{:#5}", c), "  1+i");
        assert_fmt_eq!(format_args!("{:<#5}", c), "1+i  ");
        assert_fmt_eq!(format_args!("{:^#5}", c), " 1+i ");
        assert_fmt_eq!(format_args!("{:#010x}", c), "0x001+0x1i");
    }

    mod arith {
        use super::*;

        #[test]
        fn test_add() {
            fn test(a: Rational64, b: Rational64, c: Rational64) {
                test_op!(a + b, c);
                #[cfg(feature = "ibig")]
                assert_eq!(to_big(a) + to_big(b), to_big(c));
                if b.denom == 1 {
                    let b = b.numer;
                    test_op!(a + b, c);
                }
            }

            test(_1, _1_2, _3_2);
            test(_1, _1, _2);
            test(_1_2, _1, _3_2);
            test(_1_2, _3_2, _2);
            test(_1_2, _NEG1_2, _0);
            for x in [_INF, _NEG_INF, _NAN] {
                test(x, _1, x);
                test(_1, x, x);
                test(x, _0, x);
                test(_0, x, x);
                test(x, -_1, x);
                test(-_1, x, x);
                test(x, x, x);
            }
            test(_INF, _NEG_INF, _NAN);
            test(_NAN, _INF, _NAN);
            test(_NAN, _NEG_INF, _NAN);
        }

        #[test]
        fn test_add_overflow() {
            for_integers!({
                let _1_max = Ratio::new(1 as T, T::MAX);
                let _2_max = Ratio::new(2 as T, T::MAX);
                test_op!(_1_max + _1_max, _2_max);
            });
        }

        #[test]
        fn test_sub() {
            fn test(a: Rational64, b: Rational64, c: Rational64) {
                test_op!(a - b, c);
                #[cfg(feature = "ibig")]
                assert_eq!(to_big(a) - to_big(b), to_big(c));
                if b.denom == 1 {
                    let b = b.numer;
                    test_op!(a - b, c);
                }
            }

            test(_1, _1_2, _1_2);
            test(_3_2, _1_2, _1);
            test(_1_2, _1, _NEG1_2);
            test(_1, _NEG1_2, _3_2);
            for x in [_INF, _NEG_INF, _NAN] {
                test(x, _1, x);
                test(_1, x, -x);
                test(x, _0, x);
                test(_0, x, -x);
                test(x, -_1, x);
                test(-_1, x, -x);
                test(x, x, _NAN);
            }
            test(_INF, _NEG_INF, _INF);
            test(_NAN, _INF, _NAN);
            test(_NAN, _NEG_INF, _NAN);
        }

        #[test]
        fn test_sub_overflow() {
            for_integers!({
                let _1_max: Ratio<T> = Ratio::new(1 as T, T::MAX);
                let __0 = Ratio::zero();
                test_op!(_1_max - _1_max, __0);
            });
        }

        #[test]
        fn test_mul() {
            fn test(a: Rational64, b: Rational64, c: Rational64) {
                test_op!(a * b, c);
                #[cfg(feature = "ibig")]
                assert_eq!(to_big(a) * to_big(b), to_big(c));
                if b.denom == 1 {
                    let b = b.numer;
                    test_op!(a * b, c);
                }
            }

            test(_1, _1_2, _1_2);
            test(_1_2, _3_2, Ratio::new(3, 4));
            test(_1_2, _NEG1_2, Ratio::new(-1, 4));
            test(_1_2, _2, _1);
            for x in [_INF, _NEG_INF, _NAN] {
                test(x, _1, x);
                test(_1, x, x);
                test(x, _0, _NAN);
                test(x, -_0, _NAN);
                test(_0, x, _NAN);
                test(-_0, x, _NAN);
                test(x, -_1, -x);
                test(-_1, x, -x);
            }
            test(_INF, -_2, _NEG_INF);
            test(_NEG_INF, -_2, _INF);
            test(_INF, _INF, _INF);
            test(_NEG_INF, _NEG_INF, _INF);
            test(_INF, _NEG_INF, _NEG_INF);
            test(_NAN, _INF, _NAN);
            test(_NAN, _NEG_INF, _NAN);
            test(_NAN, _NAN, _NAN);
            assert_eq!((_0 * -_0).denom, (-_0).denom);
            assert_eq!((-_0 * _0).denom, (-_0).denom);
            assert_eq!((-(_0 * _0)).denom, (-_0).denom);
            assert_eq!((_0 * -_1).denom, (-_0).denom);
            assert_eq!((-_0 * _1).denom, (-_0).denom);
            assert_eq!((-(_0 * _1)).denom, (-_0).denom);
            assert_eq!((_0 * -1).denom, (-_0).denom);
            assert_eq!((-_0 * 1).denom, (-_0).denom);
            assert_eq!((-(_0 * 1)).denom, (-_0).denom);
        }

        #[test]
        fn test_mul_overflow() {
            for_integers!({
                // 1/big * 2/3 = 1/(max/4*3), where big is max/2
                // make big = max/2, but also divisible by 2
                let big = T::MAX / 4 * 2;
                let _1_big: Ratio<T> = Ratio::new(1, big);
                let _2_3_: Ratio<T> = Ratio::new(2, 3);
                let expected = Ratio::new(1 as T, big / 2 * 3);
                test_op!(_1_big * _2_3_, expected);

                // big/3 * 3 = big/1
                // make big = max/2, but make it indivisible by 3
                let big = T::MAX / 6 * 3 + 1;
                let big_3 = Ratio::new(big, 3);
                let expected = Ratio::new(big, 1);
                test_op!(big_3 * 3, expected);
            });
        }

        #[test]
        fn test_div() {
            fn test(a: Rational64, b: Rational64, c: Rational64) {
                test_op!(a / b, c);
                #[cfg(feature = "ibig")]
                assert_eq!(to_big(a) / to_big(b), to_big(c));
                if b.denom == 1 {
                    let b = b.numer;
                    test_op!(a / b, c);
                }
            }

            test(_1, _1_2, _2);
            test(_3_2, _1_2, _1 + _2);
            test(_1, _2, _1_2);
            test(_1, _NEG1_2, _NEG1_2 + _NEG1_2 + _NEG1_2 + _NEG1_2);
            for x in [_INF, _NEG_INF, _NAN] {
                test(x, _1, x);
                test(x, _0, x);
                test(x, -_1, -x);
            }
            test(_0, _INF, _0);
            test(_0, _NEG_INF, _0);
            test(_0, _NAN, _NAN);
            test(_1, _INF, _0);
            test(_1, _NEG_INF, _0);
            test(_1, _NAN, _NAN);
            test(-_1, _INF, _0);
            test(-_1, _NEG_INF, _0);
            test(-_1, _NAN, _NAN);

            test(_INF, _INF, _NAN);
            test(_NEG_INF, _NEG_INF, _NAN);
            test(_INF, _NEG_INF, _NAN);
            test(_NAN, _INF, _NAN);
            test(_NAN, _NEG_INF, _NAN);
            test(_NAN, _NAN, _NAN);
        }

        #[test]
        fn test_div_overflow() {
            for_integers!({
                // 1/big / 3/2 = 1/(max/4*3), where big is max/2
                // big ~ max/2, and big is divisible by 2
                let big = T::max_value() / 4 * 2;
                let _1_big: Ratio<T> = Ratio::new(1, big);
                let _3_two: Ratio<T> = Ratio::new(3, 2);
                let expected = Ratio::new(1, big / 2 * 3);
                test_op!(_1_big / _3_two, expected);

                // 3/big / 3 = 1/big where big is max/2
                // big ~ max/2, and big is not divisible by 3
                let big = T::max_value() / 6 * 3 + 1;
                let _3_big = Ratio::new(3, big);
                let expected = Ratio::new(1, big);
                test_op!(_3_big / 3, expected);
            });
        }

        #[test]
        fn test_rem() {
            fn test(a: Rational64, b: Rational64, c: Rational64) {
                test_op!(a % b, c);
                #[cfg(feature = "ibig")]
                assert_eq!(to_big(a) % to_big(b), to_big(c));
                if b.denom == 1 {
                    let b = b.numer;
                    test_op!(a % b, c);
                }
            }

            test(_3_2, _1, _1_2);
            test(_3_2, _1_2, _0);
            test(_5_2, _3_2, _1);
            test(_2, _NEG1_2, _0);
            test(_1_2, _2, _1_2);
        }

        #[test]
        fn test_rem_overflow() {
            // tests that Ratio(1,2) % Ratio(1, T::max_value()) equals 0
            // for each integer type. Previously, this calculation would overflow.
            for_integers!({
                let two = 2 as T;
                // value near to maximum, but divisible by two
                let max_div2 = T::max_value() / two * two;
                let _1_max: Ratio<T> = Ratio::new(1, max_div2);
                let _1_two: Ratio<T> = Ratio::new(1, two);
                test_op!(_1_two % _1_max, Ratio::zero());
            });
        }

        #[test]
        fn test_neg() {
            fn test(a: Rational64, b: Rational64) {
                assert_eq!(-a, b);
                #[cfg(feature = "ibig")]
                assert_eq!(-to_big(a), to_big(b))
            }

            test(_0, _0);
            test(_1_2, _NEG1_2);
            test(-_1, _1);
            assert_eq!((-_0).denom, -1);
        }
        #[test]
        fn test_zero() {
            test_op!(_0 + _0, _0);
            test_op!(_0 * _0, _0);
            test_op!(_0 * _1, _0);
            test_op!(_0 / _NEG1_2, _0);
            test_op!(_0 - _0, _0);
        }
        #[test]
        fn test_div_0() {
            let a = _1 / _0;
            assert_eq!(a.numer, 1);
            assert_eq!(a.denom, 0);
            assert_eq!(a, _INF);
            let a = (-_1) / _0;
            assert_eq!(a.numer, -1);
            assert_eq!(a.denom, 0);
            assert_eq!(a, _NEG_INF);
            let a = -(_1 / _0);
            assert_eq!(a.numer, -1);
            assert_eq!(a.denom, 0);
            assert_eq!(a, _NEG_INF);
            let a = _2 / _0;
            assert_eq!(a.numer, 1); // cancelled
            assert_eq!(a.denom, 0);
            assert_eq!(a, _INF);
            let a = _0 / _0;
            assert_eq!(a.numer, 0);
            assert_eq!(a.denom, 0);
            assert_eq!(a, _NAN);
            // negative zero
            let a = _1 / -_0;
            assert_eq!(a.numer, -1);
            assert_eq!(a.denom, 0);
            assert_eq!(a, _NEG_INF);
            let a = (-_1) / -_0;
            assert_eq!(a.numer, 1);
            assert_eq!(a.denom, 0);
            assert_eq!(a, _INF);
            let a = -(_1 / -_0);
            assert_eq!(a.numer, 1);
            assert_eq!(a.denom, 0);
            assert_eq!(a, _INF);
            let a = _2 / -_0;
            assert_eq!(a.numer, -1); // cancelled
            assert_eq!(a.denom, 0);
            assert_eq!(a, _NEG_INF);
            let a = _0 / -_0;
            assert_eq!(a.numer, 0);
            assert_eq!(a.denom, 0);
            assert_eq!(a, _NAN);
            // just a number
            let a = _1 / 0;
            assert_eq!(a.numer, 1);
            assert_eq!(a.denom, 0);
            assert_eq!(a, _INF);
            let a = (-_1) / 0;
            assert_eq!(a.numer, -1);
            assert_eq!(a.denom, 0);
            assert_eq!(a, _NEG_INF);
            let a = -(_1 / 0);
            assert_eq!(a.numer, -1);
            assert_eq!(a.denom, 0);
            assert_eq!(a, _NEG_INF);
            let a = _2 / 0;
            assert_eq!(a.numer, 1); // cancelled
            assert_eq!(a.denom, 0);
            assert_eq!(a, _INF);
            let a = _0 / 0;
            assert_eq!(a.numer, 0);
            assert_eq!(a.denom, 0);
            assert_eq!(a, _NAN);
        }

        #[test]
        #[cfg(feature = "ibig")]
        fn test_bigint_euclid() {
            let a: IBig = "81204799147741".parse().unwrap();
            let b: IBig = "137498793161553".parse().unwrap();
            let r_ref = "81204799147741".parse().unwrap();
            let r_ref2 = "56293994013812".parse().unwrap();
            let (q, r) = a.div_rem_euclid(&b);
            assert!(q.is_zero());
            assert_eq!(r, r_ref);
            let (q, r) = a.div_rem_euclid(&-&b);
            assert!(q.is_zero());
            assert_eq!(r, r_ref);
            let (q, r) = (-&a).div_rem_euclid(&-&b);
            assert!(q.is_one());
            assert_eq!(r, r_ref2);
            let (q, r) = (-&a).div_rem_euclid(&b);
            assert_eq!(q, -IBig::one());
            assert_eq!(r, r_ref2);
            assert!(a.is_valid_euclid());
            assert!(b.is_valid_euclid());
            assert!(!(-a).is_valid_euclid());
            assert!(!(-b).is_valid_euclid());
            assert!(IBig::zero().is_valid_euclid());
            let a: UBig = "81204799147741".parse().unwrap();
            let b: UBig = "137498793161553".parse().unwrap();
            let r_ref = "81204799147741".parse().unwrap();
            let (q, r) = a.div_rem_euclid(&b);
            assert!(q.is_zero());
            assert_eq!(r, r_ref);
        }
    }

    #[test]
    fn test_round() {
        let __0 = 0i64;
        let __1 = 1i64;
        assert_eq!(_1_3.ceil(), __1);
        assert_eq!(_1_3.floor(), __0);
        assert_eq!(_1_3.round(), __0);
        assert_eq!(_1_3.trunc(), _0);

        assert_eq!(_NEG1_3.ceil(), __0);
        assert_eq!(_NEG1_3.floor(), -__1);
        assert_eq!(_NEG1_3.round(), __0);
        assert_eq!(_NEG1_3.trunc(), _0);

        assert_eq!(_1_NEG3.ceil(), __0);
        assert_eq!(_1_NEG3.floor(), -__1);
        assert_eq!(_1_NEG3.round(), __0);
        assert_eq!(_1_NEG3.trunc(), _0);

        assert_eq!(_2_3.ceil(), __1);
        assert_eq!(_2_3.floor(), __0);
        assert_eq!(_2_3.round(), __1);
        assert_eq!(_2_3.trunc(), _0);

        assert_eq!(_NEG2_3.ceil(), __0);
        assert_eq!(_NEG2_3.floor(), -__1);
        assert_eq!(_NEG2_3.round(), -__1);
        assert_eq!(_NEG2_3.trunc(), _0);

        assert_eq!(_2_NEG3.ceil(), __0);
        assert_eq!(_2_NEG3.floor(), -__1);
        assert_eq!(_2_NEG3.round(), -__1);
        assert_eq!(_2_NEG3.trunc(), _0);

        assert_eq!(_1_2.ceil(), __1);
        assert_eq!(_1_2.floor(), __0);
        assert_eq!(_1_2.round(), __1);
        assert_eq!(_1_2.trunc(), _0);

        assert_eq!(_NEG1_2.ceil(), __0);
        assert_eq!(_NEG1_2.floor(), -__1);
        assert_eq!(_NEG1_2.round(), -__1);
        assert_eq!(_NEG1_2.trunc(), _0);

        assert_eq!(_1.ceil(), __1);
        assert_eq!(_1.floor(), __1);
        assert_eq!(_1.round(), __1);
        assert_eq!(_1.trunc(), _1);

        // Overflow checks

        let __0 = 0i32;
        let __1 = 1i32;
        let _large_rat1 = Ratio::new(i32::MAX, i32::MAX - 1);
        let _large_rat2 = Ratio::new(i32::MAX - 1, i32::MAX);
        let _large_rat3 = Ratio::new(i32::MIN + 2, i32::MIN + 1);
        let _large_rat4 = Ratio::new(i32::MIN + 1, i32::MIN + 2);
        let _large_rat5 = Ratio::new(i32::MIN + 2, i32::MAX);
        let _large_rat6 = Ratio::new(i32::MAX, i32::MIN + 2);
        let _large_rat7 = Ratio::new(1, i32::MIN + 1);
        let _large_rat8 = Ratio::new(1, i32::MAX);

        assert_eq!(_large_rat1.round(), __1);
        assert_eq!(_large_rat2.round(), __1);
        assert_eq!(_large_rat3.round(), __1);
        assert_eq!(_large_rat4.round(), __1);
        assert_eq!(_large_rat5.round(), -__1);
        assert_eq!(_large_rat6.round(), -__1);
        assert_eq!(_large_rat7.round(), __0);
        assert_eq!(_large_rat8.round(), __0);
    }

    #[test]
    fn test_fract() {
        assert_eq!(_1.fract(), _0);
        assert_eq!(_NEG1_2.fract(), _NEG1_2);
        assert_eq!(_1_2.fract(), _1_2);
        assert_eq!(_3_2.fract(), _1_2);
    }

    #[test]
    fn test_recip() {
        assert_eq!(_1 * _1.recip(), _1);
        assert_eq!(_2 * _2.recip(), _1);
        assert_eq!(_1_2 * _1_2.recip(), _1);
        assert_eq!(_3_2 * _3_2.recip(), _1);
        assert_eq!(_NEG1_2 * _NEG1_2.recip(), _1);

        assert_eq!(_3_2.recip(), _2_3);
        assert_eq!(_NEG1_2.recip(), _NEG2);
        //assert_eq!(_NEG1_2.recip().denom, 1); // allow negative denominators
        assert_eq!(Ratio::new(0, 1).recip(), Ratio::new_raw(1, 0));
    }

    #[test]
    fn test_pow() {
        fn test(r: Rational64, e: i64, expected: Rational64) {
            assert_eq!(r.powi(e), expected);
            #[cfg(feature = "ibig")]
            test_big(r, e, expected);
        }

        #[cfg(feature = "ibig")]
        fn test_big(r: Rational64, e: i64, expected: Rational64) {
            let r = BigRational::new_raw(r.numer.into(), r.denom.into());
            let expected = BigRational::new_raw(expected.numer.into(), expected.denom.into());
            assert_eq!((&r).powi(e), expected);
        }

        test(_1_2, 2, Ratio::new(1, 4));
        test(_1_2, -2, Ratio::new(4, 1));
        test(_1, 1, _1);
        test(_1, i64::MAX, _1);
        test(_1, i64::MIN, _1);
        test(_NEG1_2, 2, _1_2.powi(2));
        test(_NEG1_2, 3, -_1_2.powi(3));
        test(_3_2, 0, _1);
        test(_3_2, -1, _3_2.recip());
        test(_3_2, 3, Ratio::new(27, 8));
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn test_algebraic() {
        // test the abs, sqrt, cbrt, sign functions from NumAlgebraic
        // to do that, use ratios of floats
        let v = 2.0f64;
        let vr = Ratio::new_raw(6.0f64, 3.0);
        let err: f64 = (vr.sqrt() - v.sqrt()).to_approx();
        assert!(err.abs() < 1e-15f64, "error {err}");
        let err: f64 = (vr.cbrt() - v.cbrt()).to_approx();
        assert!(err.abs() < 1e-15f64, "error {err}");
        let err: f64 = (vr.abs() - v.abs()).to_approx();
        assert!(err.abs() < 1e-15f64, "error {err}");
        assert_eq!(vr.sign(), Ratio::new_raw(1., 1.));
        // test with different representation
        let vr = Ratio::new_raw(-6.0f64, -3.0);
        let err: f64 = (vr.sqrt() - v.sqrt()).to_approx();
        assert!(err.abs() < 1e-15f64, "error {err}");
        let err: f64 = (vr.cbrt() - v.cbrt()).to_approx();
        assert!(err.abs() < 1e-15f64, "error {err}");
        let err: f64 = (vr.abs() - v.abs()).to_approx();
        assert!(err.abs() < 1e-15f64, "error {err}");
        assert_eq!(vr.sign(), Ratio::new_raw(-1., -1.));
        // test with negative number
        let vr = Ratio::new_raw(-6.0f64, 3.0);
        let err: f64 = vr.sqrt().to_approx();
        assert!(err.is_nan(), "error {err}");
        let err: f64 = (vr.cbrt() + v.cbrt()).to_approx();
        assert!(err.abs() < 1e-15f64, "error {err}");
        let err: f64 = (vr.abs() - v.abs()).to_approx();
        assert!(err.abs() < 1e-15f64, "error {err}");
        assert_eq!(vr.sign(), Ratio::new_raw(-1., 1.));
        // test copysign
        assert_eq!(vr.copysign(&Ratio::new_raw(1., 1.)), Ratio::new_raw(6., 3.));
        assert_eq!(
            vr.copysign(&Ratio::new_raw(-1., 1.)),
            Ratio::new_raw(-6., 3.)
        );
        assert_eq!(
            vr.copysign(&Ratio::new_raw(1., -1.)),
            Ratio::new_raw(6., -3.)
        );
        assert_eq!(
            vr.copysign(&Ratio::new_raw(-1., -1.)),
            Ratio::new_raw(-6., -3.)
        );
    }

    /*#[test]
    #[cfg(feature = "std")]
    fn test_to_from_str() {
        use std::string::{String, ToString};
        fn test(r: Rational64, s: String) {
            assert_eq!(FromStr::from_str(&s), Ok(r));
            assert_eq!(r.to_string(), s);
        }
        test(_1, "1".to_string());
        test(_0, "0".to_string());
        test(_1_2, "1/2".to_string());
        test(_3_2, "3/2".to_string());
        test(_2, "2".to_string());
        test(_NEG1_2, "-1/2".to_string());
    }
    #[test]
    fn test_from_str_fail() {
        fn test(s: &str) {
            let rational: Result<Rational64, _> = FromStr::from_str(s);
            assert!(rational.is_err());
        }

        let xs = ["0 /1", "abc", "", "1/", "--1/2", "3/2/1", "1/0"];
        for &s in xs.iter() {
            test(s);
        }
    }*/

    #[test]
    #[cfg(feature = "std")]
    fn test_hash() {
        assert!(hash(&_0) != hash(&_1));
        assert!(hash(&_0) != hash(&_3_2));

        // a == b -> hash(a) == hash(b)
        let a = Rational64::new_raw(4, 2);
        let b = Rational64::new_raw(6, 3);
        assert_eq!(a, b);
        assert_eq!(hash(&a), hash(&b));

        let a = Rational64::new_raw(123456789, 1000);
        let b = Rational64::new_raw(123456789 * 5, 5000);
        assert_eq!(a, b);
        assert_eq!(hash(&a), hash(&b));
    }

    #[test]
    fn test_into_pair() {
        assert_eq!((0, 1), _0.into());
        assert_eq!((-2, 1), _NEG2.into());
        assert_eq!((1, -2), _1_NEG2.into());
    }

    #[test]
    fn test_from_pair() {
        assert_eq!(_0, (0, 1).into());
        assert_eq!(_1, (1, 1).into());
        assert_eq!(_NEG2, (-2, 1).into());
        assert_eq!(_1_NEG2, (1, -2).into());
    }

    #[test]
    fn ratio_iter_sum() {
        // collect into array so test works on no_std
        let mut nums = [Ratio::new(0, 1); 1000];
        for (i, r) in (0..1000).map(|n| Ratio::new(n, 500)).enumerate() {
            nums[i] = r;
        }
        let slice = &nums[..];
        let mut manual_sum = Ratio::zero();
        for ratio in slice {
            manual_sum = &manual_sum + ratio;
        }
        let sums = [manual_sum, slice.iter().sum(), slice.iter().cloned().sum()];
        assert_eq!(sums[0], sums[1]);
        assert_eq!(sums[0], sums[2]);
    }

    #[test]
    fn ratio_iter_product() {
        // collect into array so test works on no_std
        let mut nums = [Ratio::new(0, 1); 1000];
        for (i, r) in (0..1000).map(|n| Ratio::new(n, 500)).enumerate() {
            nums[i] = r;
        }
        let slice = &nums[..];
        let mut manual_prod = Ratio::one();
        for ratio in slice {
            manual_prod = &manual_prod * ratio;
        }
        let products = [
            manual_prod,
            slice.iter().product(),
            slice.iter().cloned().product(),
        ];
        assert_eq!(products[0], products[1]);
        assert_eq!(products[0], products[2]);
    }

    #[test]
    fn test_num_zero() {
        let zero = Rational64::zero();
        assert!(zero.is_zero());

        let mut r = Rational64::new(123, 456);
        assert!(!r.is_zero());
        assert_eq!(r + zero, r);

        r.set_zero();
        assert!(r.is_zero());
    }

    #[test]
    fn test_num_one() {
        let one = Rational64::one();
        assert!(one.is_one());

        let mut r = Rational64::new(123, 456);
        assert!(!r.is_one());
        assert_eq!(r * one, r);

        r.set_one();
        assert!(r.is_one());
    }

    #[test]
    fn test_const() {
        const N: Ratio<i32> = Ratio::new_raw(123, 456);

        let r = N.reduced();
        assert_eq!(r.numer, (123 / 3));
        assert_eq!(r.denom, (456 / 3));
    }

    #[test]
    fn test_from_float() {
        assert_eq!(Ratio::<i128>::try_from(f32::MAX), Err(()));
        assert_eq!(
            Ratio::<i128>::try_from(f32::MAX / 2.0),
            Ok(Ratio::new_raw((f32::MAX / 2.0) as i128, 1))
        );
        assert_eq!(
            Ratio::<i128>::try_from(-f32::MAX / 2.0),
            Ok(Ratio::new_raw(-(f32::MAX / 2.0) as i128, 1))
        );
        assert_eq!(
            Ratio::<i128>::try_from(f32::MIN_POSITIVE),
            Ok(Ratio::new_raw(1, 1 << (1 - f32::MIN_EXP)))
        );
        assert_eq!(
            Ratio::<i128>::try_from(f32::INFINITY),
            Ok(Ratio::new_raw(1, 0))
        );
        assert_eq!(
            Ratio::<i128>::try_from(f32::NEG_INFINITY),
            Ok(Ratio::new_raw(-1, 0))
        );
        assert_eq!(Ratio::<i128>::try_from(f32::NAN), Ok(Ratio::new_raw(0, 0)));
        assert_eq!(Ratio::<i128>::try_from(f64::MAX), Err(()));
        assert_eq!(
            Ratio::<i128>::try_from((f32::MAX / 2.0) as f64),
            Ok(Ratio::new_raw((f32::MAX / 2.0) as i128, 1))
        );
        assert_eq!(
            Ratio::<i128>::try_from((-f32::MAX / 2.0) as f64),
            Ok(Ratio::new_raw(-(f32::MAX / 2.0) as i128, 1))
        );
        assert_eq!(
            Ratio::<i128>::try_from(f32::MIN_POSITIVE as f64),
            Ok(Ratio::new_raw(1, 1 << (1 - f32::MIN_EXP)))
        );
        assert_eq!(
            Ratio::<i128>::try_from(f64::INFINITY),
            Ok(Ratio::new_raw(1, 0))
        );
        assert_eq!(
            Ratio::<i128>::try_from(f64::NEG_INFINITY),
            Ok(Ratio::new_raw(-1, 0))
        );
        assert_eq!(Ratio::<i128>::try_from(f64::NAN), Ok(Ratio::new_raw(0, 0)));
        // now test all simple integers
        assert_eq!(Ratio::<i128>::try_from(1.0f32), Ok(Ratio::one()));
        assert_eq!(Ratio::<i64>::try_from(1.0f32), Ok(Ratio::one()));
        assert_eq!(Ratio::<i32>::try_from(1.0f32), Ok(Ratio::one()));
        assert_eq!(Ratio::<i128>::try_from(1.0f64), Ok(Ratio::one()));
        assert_eq!(Ratio::<i64>::try_from(1.0f64), Ok(Ratio::one()));
        assert_eq!(Ratio::<i128>::try_from(-1.0f32), Ok(-Ratio::one()));
        assert_eq!(Ratio::<i64>::try_from(-1.0f32), Ok(-Ratio::one()));
        assert_eq!(Ratio::<i32>::try_from(-1.0f32), Ok(-Ratio::one()));
        assert_eq!(Ratio::<i128>::try_from(-1.0f64), Ok(-Ratio::one()));
        assert_eq!(Ratio::<i64>::try_from(-1.0f64), Ok(-Ratio::one()));
    }

    #[test]
    #[cfg(feature = "ibig")]
    fn test_big_ratio_to_f64() {
        assert_eq!(
            4115226303292181e29f64,
            BigRational::new(
                "1234567890987654321234567890987654321234567890"
                    .parse()
                    .unwrap(),
                "3".parse().unwrap()
            )
            .to_approx()
        );
        let r: Ratio::<IBig> = 5e-324.into();
        assert_eq!(5e-324f64, r.to_approx());
        assert_eq!(
            5e-324f64,
            Ratio::<IBig>::from_approx(5e-324, 0.0)
                .unwrap()
                .to_approx()
        );
        assert_eq!(
            // subnormal
            2.0f64.powi(-50).powi(21),
            BigRational::new(IBig::one(), IBig::one() << 1050).to_approx(),
        );
        assert_eq!(
            // definite underflow
            0.0,
            BigRational::new(IBig::one(), IBig::one() << 1100).to_approx(),
        );
        assert_eq!(
            core::f64::INFINITY,
            BigRational::from(IBig::one() << 1050).to_approx(),
        );
        assert_eq!(
            core::f64::NEG_INFINITY,
            BigRational::from((-IBig::one()) << 1050).to_approx(),
        );
        assert_eq!(
            1.2499999893125f64,
            BigRational::new(
                "1234567890987654321234567890".parse().unwrap(),
                "987654321234567890987654321".parse().unwrap()
            )
            .to_approx(),
        );
        assert_eq!(
            core::f64::INFINITY,
            BigRational::new_raw(IBig::one(), IBig::zero()).to_approx(),
        );
        assert_eq!(
            core::f64::NEG_INFINITY,
            BigRational::new_raw(-IBig::one(), IBig::zero()).to_approx(),
        );
        let f: f32 = BigRational::new_raw(IBig::zero(), IBig::zero()).to_approx();
        assert!(f.is_nan());
    }

    #[test]
    fn test_ratio_to_approx() {
        assert_eq!(0.5f64, Ratio::<i32>::new(1, 2).to_approx());
        assert_eq!(0.5f64, Rational64::new(1, 2).to_approx());
        assert_eq!(-0.5f64, Rational64::new(1, -2).to_approx());
        assert_eq!(0.0f64, Rational64::new(0, 2).to_approx());
        assert_eq!(-0.0f64, Rational64::new(0, -2).to_approx());
        assert_eq!(8f64, Rational64::new((1 << 57) + 1, 1 << 54).to_approx());
        assert_eq!(
            1.0000000000000002f64,
            Rational64::new((1 << 52) + 1, 1 << 52).to_approx(),
        );
        assert_eq!(
            1.0000000000000002f64,
            Rational64::new((1 << 60) + (1 << 8), 1 << 60).to_approx(),
        );
        assert_eq!(core::f64::INFINITY, Ratio::<i32>::new_raw(1, 0).to_approx(),);
        assert_eq!(
            core::f64::NEG_INFINITY,
            Ratio::<i32>::new_raw(-1, 0).to_approx(),
        );
        let v: f32 = Ratio::<i32>::new_raw(0, 0).to_approx();
        assert!(v.is_nan());
    }

    #[test]
    fn test_continued_fractions() {
        // extended version of the doctest in continued_fractions
        for end in -1..3 {
            let mut iter = [1, 2, 2].iter().continued_fraction(end);
            assert_eq!(Some(_1 + end), iter.next());
            let first = [1, 2, 2].iter().continued_fraction(end).nth(0);
            assert_eq!(Some(_1 + end), first);

            let second_expected = Some(_1 + _1 / (_1 * 2 + end));
            assert_eq!(second_expected, iter.next());
            let second = [1, 2, 2].iter().continued_fraction(end).nth(1);
            assert_eq!(second_expected, second);

            let last_expected = Some(_1 + _1 / (_1 * 2 + _1 / (_1 * 2 + end)));
            assert_eq!(last_expected, iter.next());
            let last = [1, 2, 2].iter().continued_fraction(end).last();
            assert_eq!(last_expected, last);
            let last = [1, 2, 2].iter().continued_fraction(end).nth(2);
            assert_eq!(last_expected, last);

            assert_eq!(None, iter.next());
            assert_eq!(None, [1, 2, 2].iter().continued_fraction(end).nth(3));
        }
    }
}

#[cfg(feature = "rational")]
mod extension {
    use super::*;
    use core::cmp::Ordering;

    const ZERO: SqrtExt<i32, Sqrt<i32, 5>> = SqrtExt::new(0, 0);
    const ONE: SqrtExt<i32, Sqrt<i32, 5>> = SqrtExt::new(1, 0);
    const SQRT2: SqrtExt<i32, Sqrt<i32, 2>> = SqrtExt::new(0, 1);
    const TESTNUM: SqrtExt<i32, Sqrt<i32, 2>> = SqrtExt::new(-4, -3);
    const SQRT5: SqrtExt<i32, Sqrt<i32, 5>> = SqrtExt::new(0, 1);
    const PHI: SqrtExt<i32, Sqrt<i32, 5>> = SqrtExt::new(1, 1);
    const PHI2: SqrtExt<u32, Sqrt<u32, 5>> = SqrtExt::new(1, 1); // // 1 + √5
    const RF: SqrtExt<f32, Sqrt<f32, 5>> = SqrtExt::new(1.5, -2.7);
    const R: SqrtExt<Ratio<i32>, Sqrt<Ratio<i32>, 5>> =
        SqrtExt::new(Ratio::new_raw(1, 2), Ratio::new_raw(1, 3));

    #[test]
    fn test_extension() {
        assert_eq!(PHI.abs_sqr(), (PHI + 2) * 2); // phi^2 = phi+1
        // 1 + i√5
        const C: SqrtExt<Complex<i32>, Sqrt<Complex<i32>, 5>> =
            SqrtExt::new(Complex::new(1, 0), Complex::new(0, 1));
        assert_eq!(C.abs_sqr(), 6.into());
        // 1-3i + (2+i)√5
        const X: SqrtExt<Complex<i32>, Sqrt<Complex<i32>, 5>> =
            SqrtExt::new(Complex::new(1, -3), Complex::new(2, 1));
        assert_eq!(X.abs_sqr(), SqrtExt::new(35, -2));

        // The gcd is currently not unique.
        let a = SqrtExt::<_, Sqrt<_, 2>>::new(2, 1);
        let b = SqrtExt::<_, Sqrt<_, 2>>::new(3, 0);
        let _ = gcd(a, b);
        let _ = lcm(a, b);

        let a = SqrtExt::<_, Sqrt<_, 7>>::new(2, 1);
        let b = SqrtExt::<_, Sqrt<_, 7>>::new(3, 5);
        let _ = gcd(a, b);
        let _ = lcm(a, b);
    }

    #[test]
    #[allow(dead_code)]
    fn test_extension_string_formatting() {
        // normal numbers are handled using the default formatter, so don't test these too much
        assert_fmt_eq!(format_args!("{:5}", ZERO), "    0");
        assert_fmt_eq!(format_args!("{:5}", ONE), "    1");
        assert_fmt_eq!(format_args!("{:05}", -ONE), "-0001");
        assert_fmt_eq!(format_args!("{:#b}", ZERO), "0b0");
        assert_fmt_eq!(format_args!("{:.1}", 1000), "1000"); // this works
        assert_fmt_eq!(format_args!("{:.1}", SQRT5), "√5"); // so this should also work (this fails if an alignment is given...)
        assert_fmt_eq!(format_args!("{:#x}", SQRT5), "√0x5");
        assert_fmt_eq!(format_args!("{:#X}", SQRT5), "√0x5");
        assert_fmt_eq!(format_args!("{:#3x}", SQRT5), "√0x5");
        assert_fmt_eq!(format_args!("{:#o}", SQRT5), "√0o5");
        assert_fmt_eq!(format_args!("{:b}", SQRT5), "√101");

        assert_fmt_eq!(format_args!("{:#5x}", SQRT5), " √0x5");
        assert_fmt_eq!(format_args!("{:#05x}", SQRT5), " √0x5");
        assert_fmt_eq!(format_args!("{:5}", -SQRT5), " -1√5");
        assert_fmt_eq!(format_args!("{:+5}", SQRT5), "  +√5");
        assert_fmt_eq!(format_args!("{:+5}", -SQRT5), " -1√5");
        assert_fmt_eq!(format_args!("{:>5}", SQRT5), "   √5");
        assert_fmt_eq!(format_args!("{:<05}", 1), "00001"); // this is not nonsense! (in contrast to 0<5)
        assert_fmt_eq!(format_args!("{:>05}", SQRT5), "   √5"); // so this is also not nonsense.
        assert_fmt_eq!(format_args!("{:0>5}", SQRT5), "000√5"); // zero as fill character
        assert_fmt_eq!(format_args!("{}", PHI), "1+√5");
        assert_fmt_eq!(format_args!("{}", PHI2), "1+√5");
        assert_fmt_eq!(format_args!("{}", -PHI), "-1-1√5");
        assert_fmt_eq!(format_args!("{}", R), "1/2+(1/3)√5");
        assert_fmt_eq!(format_args!("{:#}", R), "1/2+(1/3)√5");
        assert_fmt_eq!(format_args!("{}", RF), "1.5-2.7√5");
        assert_fmt_eq!(format_args!("{:#}", RF), "1.5-2.7√5");
        assert_fmt_eq!(format_args!("{:.3}", RF), "1.500-2.700√5");
        assert_fmt_eq!(format_args!("{:#.3}", RF), "1.500-2.700√5");
        assert_fmt_eq!(format_args!("{:-^8b}", SQRT5), "--√101--");
        assert_fmt_eq!(format_args!("{:-^#10b}", SQRT5), "--√0b101--");
        assert_fmt_eq!(format_args!("{:.>5}", SQRT5), "...√5");
    }

    #[test]
    fn test_rem_for_gcd() {
        let mut a = SqrtExt::<i64, Sqrt<_, 3>>::new(-1, -7);
        let mut b = SqrtExt::<_, Sqrt<_, 3>>::new(-7, -7);
        while !b.is_zero() {
            (b, a) = (a.div_rem_euclid(&b).1, b); // fails here by overflowing
            #[cfg(feature = "std")]
            std::println!("{a}, {b}");
        }
    }

    #[test]
    fn test_cmp() {
        const N: u64 = 7;
        //const S: SqrtExt<i32, Sqrt<i32, N>> = SqrtExt::new(-1, 1); // N=2 -> 0.4142, N=3 -> 0.732, otherwise > 1
        //const S: SqrtExt<i32, Sqrt<i32, N>> = SqrtExt::new(-2, 1); // N=5..9
        //const S: SqrtExt<i32, Sqrt<i32, N>> = SqrtExt::new(-5, 2); // N=7
        const S: SqrtExt<i32, Sqrt<i32, N>> = SqrtExt::new(6, -2); // N=7
        for i in -5..=5 {
            #[cfg(any(feature = "std", feature = "libm"))]
            assert_eq!(
                S.cmp(&i.into()),
                (i as f32).total_cmp(&S.to_approx()).reverse()
            );
            assert_eq!((S + i).cmp(&i.into()), Ordering::Greater);
            assert_eq!((-S * S + i).cmp(&i.into()), Ordering::Less);
            assert_eq!((-S * S * S + i).cmp(&i.into()), Ordering::Less);
            assert_eq!((-S * S * S * S + i).cmp(&i.into()), Ordering::Less);
            let x = S * S + i;
            assert_eq!((S + i).cmp(&x), Ordering::Greater);
            assert_eq!(x.cmp(&x), Ordering::Equal);
            let x_prev = x;
            let x = S * S * S + i;
            assert_eq!((S + i).cmp(&x), Ordering::Greater);
            assert_eq!(x.cmp(&x), Ordering::Equal);
            assert_eq!(x_prev.cmp(&x), Ordering::Greater);
            let x_prev = x;
            let x = S * S * S * S + i;
            assert_eq!((S + i).cmp(&x), Ordering::Greater);
            assert_eq!(x.cmp(&x), Ordering::Equal);
            assert_eq!(x_prev.cmp(&x), Ordering::Greater);
        }
        #[cfg(any(feature = "std", feature = "libm"))]
        {
            let x: SqrtExt<_, Sqrt<_, 5>> = SqrtExt::new(-5i64, -5);
            let y: SqrtExt<_, Sqrt<_, 5>> = SqrtExt::new(0, 3);
            let xf: f64 = x.to_approx();
            assert_eq!(x.cmp(&y), xf.total_cmp(&y.to_approx()));
        }
    }

    #[test]
    fn test_arithmetic() {
        const S: SqrtExt<i32, Sqrt<i32, 7>> = SqrtExt::new(6, -2);
        test_op!(S + 2, SqrtExt::new(8, -2));
        test_op!(S - 2, SqrtExt::new(4, -2));
        for ss in [S * S, &S * &S] {
            test_op!(ss - 4, (S + 2) * (S - 2));
            test_op!(ss - 9, (S + 3) * (S - 3));
            // test division
            let ss4 = ss - 4;
            test_op!(ss4 / (S + 2), S - 2);
            test_op!(ss4 / (S - 2), S + 2);
            let ss9 = ss - 9;
            test_op!(ss9 / (S + 3), S - 3);
            test_op!(ss9 / (S - 3), S + 3);
        }
        const D: SqrtExt<i32, Sqrt<i32, 7>> = SqrtExt::new(2, -5);
        assert!(!D.is_unit());
        test_div_rem!(S, D);
        test_div_rem!(S, -D);
        let s = -S;
        test_div_rem!(s, D);
        test_div_rem!(s, -D);
    }

    #[test]
    fn test_arithmetic_overflow() {
        let a: SqrtExt<i8, Sqrt<i8, 2>> = SqrtExt::new(4, 2);
        let b: SqrtExt<i8, Sqrt<i8, 2>> = SqrtExt::new(14, -6);
        let _ = a / b;
        let _ = &a / &b;
        let a: SqrtExt<i8, Sqrt<i8, 2>> = SqrtExt::new(2, 8);
        let b: SqrtExt<i8, Sqrt<i8, 2>> = SqrtExt::new(14, -6);
        let _ = a / b;
        let _ = &a / &b;
    }

    #[test]
    #[cfg(feature = "std")] // TODO remove this cfg when dynamic_sqrt_const is available in no_std
    fn test_unit() {
        //let _ = Sqrt::<i64, 9>::TEST_SQRT; // panics at compile time.
        // using over 100 types here for all numbers is a bad idea, so use the dynamic one to improve compile time.
        fn test_fn(n: u64) {
            dynamic_sqrt_const!(N, n);
            let u = SqrtExt::<i64, N<i64>>::unit();
            //std::println!("{u}");
            assert!(u.is_unit());
            assert!(!u.is_one());
        }
        for i in 2u64..=198 {
            if i == 151 || i == 166 || i == 181 {
                // 199 is the next one that fails with i64
                continue; // too big solution for i64
            }
            if i.isqrt() * i.isqrt() != i {
                test_fn(i);
            }
        }
        macro_rules! test {
            ($($N:literal),+) => {
                $(test_fn($N);)+
            };
        }
        // test for some bigger primes, which have somewhat simple solutions (617 has actually a quite large solution)
        test!(617, 2029, 21029);
        // These primes solutions are too big for the algorithm, even with i128
        //test!(6701, 12713, 20297, 59207, 60811, 79031);
    }

    #[test]
    #[cfg(feature = "ibig")]
    fn test_archimedes_cattle_problem() {
        type I = ibig::IBig;
        // originally the problem was to find the fundamental solution in √410286423278424
        // however the square free factorisation yields: 410286423278424 = 4729494 * 9314^2
        let unit = SqrtExt::<I, Sqrt<I, 4729494>>::unit();
        #[cfg(feature = "std")]
        std::println!("{}", unit);
        assert!(unit.is_unit());
        // now find a power, where y is divisible by 9314
        // this can safely be done in modulo arithmetic
        let unit_mod = SqrtExt::<i64, Sqrt<_, 4729494>>::try_from(&unit % &I::from(9314)).unwrap();
        let mut p = SqrtExt::<i64, _>::one();
        let mut power = 0;
        for i in 1..4000 {
            p *= &unit_mod;
            p %= 9314;
            if p.ext == 0 {
                #[cfg(feature = "std")]
                std::println!("{i}");
                power = i;
                break;
            }
        }
        assert_eq!(power, 2329);
        //std::println!("{}", unit.powu(power as u64)); // fills the entire terminal
    }

    #[test]
    #[cfg(feature = "std")] // TODO remove this cfg when dynamic_sqrt_const is available in no_std
    fn test_rational_sqrt() {
        fn test(n: u64, iter: u64) {
            //std::println!("start √{n}");
            dynamic_sqrt_const!(N, n);
            for n in 0..iter {
                let r = Ratio::approx_sqrt::<N<i64>>(n);
                #[cfg(feature = "std")]
                //std::println!("{}", r);
                let x = SqrtExt::<_, N<_>>::new(-r.numer, r.denom);
                assert!(x > (-1).into());
                assert!(x < 1.into());
                assert_eq!(x.floor(), if x >= 0.into() { 0 } else { -1 });
            }
            // N is neither Send nor Sync! The following panics in the thread:
            #[cfg(feature = "std")]
            std::thread::spawn(|| {
                std::panic::set_hook(std::boxed::Box::new(|_| {}));
                let _ = Ratio::approx_sqrt::<N<i64>>(3);
            })
            .join()
            .unwrap_err();
        }
        test(2, 20);
        test(3, 12);
        test(5, 12);
        test(6, 12);
        test(8, 12);
        test(26, 8);
        test(31, 8);
        let x = SqrtExt::<_, Sqrt<i64, 2>>::new(0, 2);
        for n in 0..12 {
            let _r = x.approx_rational(n);
            //std::println!("{}", _r);
        }
    }

    #[test]
    fn test_round() {
        //assert_eq!(1i32.div_floor(&2), 0);
        //assert_eq!((-1i32).div_floor(&2), -1);
        //assert_eq!(1i32.div_floor(&-2), -1);
        //assert_eq!((-1i32).div_floor(&-2), 0);

        assert_eq!(ZERO.ceil(), 0);
        assert_eq!(ZERO.floor(), 0);
        assert_eq!(ZERO.div_floor(&ONE), 0);
        assert_eq!(ZERO.round(), 0);

        assert_eq!(ONE.ceil(), 1);
        assert_eq!(ONE.floor(), 1);
        assert_eq!(ONE.div_floor(&ONE), 1);
        assert_eq!(ONE.div_floor(&-ONE), -1);
        assert_eq!(ONE.div_floor(&SQRT5), 0);
        assert_eq!(ONE.div_floor(&-SQRT5), -1);
        assert_eq!(ONE.round(), 1);

        assert_eq!((-ONE).ceil(), -1);
        assert_eq!((-ONE).floor(), -1);
        assert_eq!((-ONE).div_floor(&ONE), -1);
        assert_eq!((-ONE).div_floor(&-ONE), 1);
        assert_eq!((-ONE).div_floor(&SQRT5), -1);
        assert_eq!((-ONE).div_floor(&-SQRT5), 0);
        assert_eq!((-ONE).round(), -1);

        assert_eq!(SQRT2.ceil(), 2);
        assert_eq!(SQRT2.floor(), 1);
        assert_eq!(SQRT2.round(), 1);

        assert_eq!(TESTNUM.ceil(), -8);
        assert_eq!(TESTNUM.floor(), -9);
        assert_eq!(TESTNUM.round(), -8);

        assert_eq!(SQRT5.ceil(), 3);
        assert_eq!(SQRT5.floor(), 2);
        assert_eq!(SQRT5.div_floor(&ONE), 2);
        assert_eq!(SQRT5.div_floor(&-ONE), -3);
        assert_eq!(SQRT5.div_floor(&SQRT5), 1);
        assert_eq!(SQRT5.div_floor(&-SQRT5), -1);
        assert_eq!(SQRT5.div_floor(&PHI), 0);
        assert_eq!(SQRT5.div_floor(&-PHI), -1);
        assert_eq!(SQRT5.round(), 2);

        assert_eq!((-SQRT5).ceil(), -2);
        assert_eq!((-SQRT5).floor(), -3);
        assert_eq!((-SQRT5).div_floor(&ONE), -3);
        assert_eq!((-SQRT5).div_floor(&-ONE), 2);
        assert_eq!((-SQRT5).div_floor(&SQRT5), -1);
        assert_eq!((-SQRT5).div_floor(&-SQRT5), 1);
        assert_eq!((-SQRT5).div_floor(&PHI), -1);
        assert_eq!((-SQRT5).div_floor(&-PHI), 0);
        assert_eq!((-SQRT5).round(), -2);

        assert_eq!(PHI.ceil(), 4);
        assert_eq!(PHI.floor(), 3);
        assert_eq!(PHI.div_floor(&ONE), 3);
        assert_eq!(PHI.div_floor(&SQRT5), 1);
        assert_eq!(PHI.div_floor(&-SQRT5), -2);
        assert_eq!(PHI.round(), 3);

        assert_eq!((-PHI).ceil(), -3);
        assert_eq!((-PHI).floor(), -4);
        assert_eq!((-PHI).div_floor(&ONE), -4);
        assert_eq!((-PHI).div_floor(&SQRT5), -2);
        assert_eq!((-PHI).div_floor(&-SQRT5), 1);
        assert_eq!((-PHI).round(), -3);

        assert_eq!(PHI2.ceil(), 4);
        assert_eq!(PHI2.floor(), 3);
        assert_eq!(PHI2.round(), 3);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn test_floor_div() {
        for i in -5i64..=5 {
            for j in -5..=5 {
                for a in -5..=5 {
                    for b in -5..=5 {
                        //println!("{i} {j} {a} {b}");
                        let x = SqrtExt::<_, Sqrt<_, 2>>::new(a, b);
                        let d = SqrtExt::<_, Sqrt<_, 2>>::new(i, j);
                        // compare to float approximation result
                        let f: f64 = x.to_approx();
                        assert_eq!(x.floor(), f.floor() as i64, "failed for ({x})/({d}) ~ {f}");

                        if i != 0 || j != 0 {
                            let fi = x.div_floor(&d);
                            // compare to float approximation result
                            let f: f64 = Ratio::new_raw(x, d).to_approx();
                            assert_eq!(fi, ((f*128.0).round() / 128.0).floor() as i64, "failed for ({x})/({d}) ~ {f}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_euclid() {
        for i in -5i64..=5 {
            for j in -5..=5 {
                for a in -5..=5 {
                    for b in -5..=5 {
                        //println!("{i} {j} {a} {b}");
                        let x = SqrtExt::<_, Sqrt<_, 2>>::new(a, b);
                        let d = SqrtExt::<_, Sqrt<_, 2>>::new(i, j);
                        let (q, r) = x.div_rem_euclid(&d);
                        assert_eq!(q * d + r, x);
                        if i != 0 || j != 0 {
                            assert!(
                                r.is_valid_euclid(),
                                "({x})/({d}) -> ({q}, {r}) [numer={}, denom={}]",
                                x * d.conj_ext(),
                                d.abs_sqr_ext()
                            );
                            assert!(
                                r.abs_sqr() < d.abs_sqr(),
                                "({x})/({d}) -> ({q}, {r}) [numer={}, denom={}]",
                                x * d.conj_ext(),
                                d.abs_sqr_ext()
                            );
                            //assert!(r.abs_sqr_ext().abs() < d.abs_sqr_ext().abs());
                        }
                        else {
                            assert!(q.is_zero());
                        }

                        let x = SqrtExt::<_, Sqrt<_, 3>>::new(a, b);
                        let d = SqrtExt::<_, Sqrt<_, 3>>::new(i, j);
                        let (q, r) = x.div_rem_euclid(&d);
                        assert_eq!(q * d + r, x);
                        if i != 0 || j != 0 {
                            assert!(r.is_valid_euclid());
                            assert!(
                                r.abs_sqr() < d.abs_sqr(),
                                "({x})/({d}) -> ({q}, {r}) [numer={}, denom={}]",
                                x * d.conj_ext(),
                                d.abs_sqr_ext()
                            );
                            //assert!(r.abs_sqr_ext().abs() < d.abs_sqr_ext().abs());
                        }
                        else {
                            assert!(q.is_zero());
                        }

                        let x = SqrtExt::<_, Sqrt<_, 5>>::new(a, b);
                        let d = SqrtExt::<_, Sqrt<_, 5>>::new(i, j);
                        let (q, r) = x.div_rem_euclid(&d);
                        assert_eq!(q * d + r, x);
                        if i != 0 || j != 0 {
                            assert!(r.is_valid_euclid());
                            assert!(r.abs_sqr() < d.abs_sqr());
                        }
                        else {
                            assert!(q.is_zero());
                        }

                        let x = SqrtExt::<_, Sqrt<_, 6>>::new(a, b);
                        let d = SqrtExt::<_, Sqrt<_, 6>>::new(i, j);
                        let (q, r) = x.div_rem_euclid(&d);
                        assert_eq!(q * d + r, x);
                        if i != 0 || j != 0 {
                            assert!(r.is_valid_euclid());
                            assert!(r.abs_sqr() < d.abs_sqr());
                        }
                        else {
                            assert!(q.is_zero());
                        }

                        let x = SqrtExt::<_, Sqrt<_, 7>>::new(a, b);
                        let d = SqrtExt::<_, Sqrt<_, 7>>::new(i, j);
                        let (q, r) = x.div_rem_euclid(&d);
                        assert_eq!(q * d + r, x);
                        if i != 0 || j != 0 {
                            assert!(r.is_valid_euclid());
                            assert!(r.abs_sqr() < d.abs_sqr());
                        }
                        else {
                            assert!(q.is_zero());
                        }
                        if j != 0 && b != 0 {
                            let x = Ratio::new(a, b);
                            let d = Ratio::new(i, j);
                            let (q, r) = x.div_rem_euclid(&d);
                            assert_eq!(q * d + r, x);
                            if i != 0 {
                                assert!(r.is_valid_euclid());
                                assert!(r.abs_sqr() < d.abs_sqr());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gcd() {
        for i in -7i64..=7 {
            for j in -7..=7 {
                for a in -7..=7 {
                    for b in -7..=7 {
                        {
                            // only work for N=2,3,6,7,11  so don't ever put an SqrtExt of another type into a Ratio!
                            const N: u64 = 3;
                            let a = SqrtExt::<_, Sqrt<_, N>>::new(a, b);
                            let d = SqrtExt::<_, Sqrt<_, N>>::new(i, j);
                            //std::print!("{a}, {d} -> "); let _ = std::io::stdout().flush();
                            let _g = gcd(a, d); // may fail with overflow if the gcd doesn't converge.
                            //std::println!("{}", g);
                            let ((x, y), g2) = bezout(a, d);
                            assert_eq!(a * x + d * y, g2);
                            //assert_eq!(g, g2, "{a} {d}"); // doesn't work... that's because I decided to make the results positive
                        }
                        {
                            const N: u64 = 2;
                            let a = SqrtExt::<_, Sqrt<_, N>>::new(a, b);
                            let d = SqrtExt::<_, Sqrt<_, N>>::new(i, j);
                            let _g = gcd(a, d);
                            let ((x, y), g2) = bezout(a, d);
                            assert_eq!(a * x + d * y, g2);
                        }
                        {
                            const N: u64 = 6;
                            let a = SqrtExt::<_, Sqrt<_, N>>::new(a, b);
                            let d = SqrtExt::<_, Sqrt<_, N>>::new(i, j);
                            let _g = gcd(a, d);
                            let ((x, y), g2) = bezout(a, d);
                            assert_eq!(a * x + d * y, g2);
                        }
                        {
                            const N: u64 = 7;
                            let a = SqrtExt::<_, Sqrt<_, N>>::new(a, b);
                            let d = SqrtExt::<_, Sqrt<_, N>>::new(i, j);
                            let _g = gcd(a, d);
                            let ((x, y), g2) = bezout(a, d);
                            assert_eq!(a * x + d * y, g2);
                        }
                        {
                            const N: u64 = 11;
                            let a = SqrtExt::<_, Sqrt<_, N>>::new(a, b);
                            let d = SqrtExt::<_, Sqrt<_, N>>::new(i, j);
                            let _g = gcd(a, d);
                            let ((x, y), g2) = bezout(a, d);
                            assert_eq!(a * x + d * y, g2);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_continued_fraction() {
        // all pushed out to almost overflow
        {
            //std::println!("starting √2");
            let cf = [
                1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            ];
            let x = SqrtExt::<i64, Sqrt<i64, 2>>::new(0, 1);
            let mut iter = DevelopContinuedFraction::new(x);
            for n in 0..cf.len() {
                assert_eq!(iter.next().unwrap(), cf[n], "failed at {n}");
                let _r = cf[..n].iter().continued_fraction(1).last().unwrap();
                //std::println!("{}", _r);
            }
        }
        {
            //std::println!("starting 3√2");
            let cf = [
                4, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8,
            ];
            let x = SqrtExt::<i64, Sqrt<i64, 2>>::new(0, 3);
            let mut iter = DevelopContinuedFraction::new(x);
            for n in 0..cf.len() {
                assert_eq!(iter.next().unwrap(), cf[n], "failed at {n}");
                let _r = cf[..n].iter().continued_fraction(1).last().unwrap();
                //std::println!("{}", _r);
            }
        }
        {
            //std::println!("starting 4√2");
            let cf = [
                5, 1, 1, 1, 10, 1, 1, 1, 10, 1, 1, 1, 10, 1, 1, 1, 10, 1, 1, 1, 10, 1, 1, 1, 10,
                1, 1, 1, 10, 1, 1, 1, 10, 1, 1, 1, 10, 1, 1, 1, 10, 1, 1, 1, 10, 1, 1, 1, 10,
            ];
            let x = SqrtExt::<i64, Sqrt<i64, 2>>::new(0, 4);
            let mut iter = DevelopContinuedFraction::new(x);
            for n in 0..cf.len() {
                assert_eq!(iter.next().unwrap(), cf[n], "failed at {n}");
                let _r = cf[..n].iter().continued_fraction(1).last().unwrap();
                //std::println!("{}", _r);
            }
        }
        {
            //std::println!("starting √3");
            let cf = [
                1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
                2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
                2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
            ];
            let x = SqrtExt::<i64, Sqrt<i64, 3>>::new(0, 1);
            let mut iter = DevelopContinuedFraction::new(x);
            for n in 0..cf.len() {
                assert_eq!(iter.next().unwrap(), cf[n], "failed at {n}");
                let _r = cf[..n].iter().continued_fraction(1).last().unwrap();
                //std::println!("{}", _r);
            }
        }
        {
            //std::println!("starting √5");
            let cf = [
                2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                4, 4, 4,
            ];
            let x = SqrtExt::<i64, Sqrt<i64, 5>>::new(0, 1);
            let mut iter = DevelopContinuedFraction::new(x);
            for n in 0..cf.len() {
                assert_eq!(iter.next().unwrap(), cf[n], "failed at {n}");
                let _r = cf[..n].iter().continued_fraction(1).last().unwrap();
                //std::println!("{}", _r);
            }
        }
        {
            //std::println!("starting √24");
            let x = SqrtExt::<i64, Sqrt<i64, 24>>::new(0, 1);
            let mut iter1 = DevelopContinuedFraction::new(x);
            let mut iter2 = DevelopContinuedFraction::new(x).continued_fraction(1);
            for _ in 0..38 {
                let (_r1, _r2) = (iter1.next().unwrap(), iter2.next().unwrap());
                //std::println!("{} -> {}", _r1, _r2);
            }
        }
        {
            //std::println!("starting √31");
            let cf = [
                5, 1, 1, 3, 5, 3, 1, 1, 10, 1, 1, 3, 5, 3, 1, 1, 10, 1, 1, 3, 5, 3, 1, 1, 10, 1, 1,
                3, 5, 3, 1, 1, 10, 1, 1, 3, 5, 3, 1, 1, 10, 1, 1, 3,
            ];
            let x = SqrtExt::<i64, Sqrt<i64, 31>>::new(0, 1);
            let mut iter = DevelopContinuedFraction::new(x);
            for n in 0..cf.len() {
                assert_eq!(iter.next().unwrap(), cf[n], "failed at {n}");
                let _r = cf[..n].iter().continued_fraction(1).last().unwrap();
                //std::println!("{}", _r);
            }
        }
        //std::println!("{}", Ratio::approx_sqrt::<extension::Sqrt<i64, 31>>(n as u64));
    }

    #[test]
    fn test_ratio_sqrt() {
        // test a bunch of things with
        type S = Sqrt<Ratio<i64>, 2>;
        assert_eq!(
            SqrtExt::<_, S>::unit(),
            SqrtExt::<_, S>::new(Ratio::zero(), Ratio::one())
        ); // 1/sqrt(N) = sqrt(N)/N
        // test cf development into fractions of sqrt terms using floats (testing for infinite loops)
        let mut iter = DevelopContinuedFraction::new(core::f64::consts::PI);
        let x = SqrtExt::<f64, Sqrt<f64, 2>>::new(core::f64::consts::PI, 0.0);
        let mut iter2 = DevelopContinuedFraction::new(Ratio::from(x));
        for i in 0..20 {
            assert_eq!(iter2.next().unwrap(), iter.next().unwrap().into(), "{i}");
        }

        // check for endless loops
        let x = SqrtExt::<f64, Sqrt<f64, 10>>::new(0.0, 1.0 / core::f64::consts::PI);
        let mut iter = DevelopContinuedFraction::new(Ratio::from(x));
        for _ in 0..10 {
            let _next = iter.next().unwrap();
            // This is NaN, because the gcd is computed on a transcendent f64 number.
            //std::println!("{}", next);
            //assert!(next.value.is_finite() && next.ext.is_finite()); // would fail
        }

        // this next form gives different results including notably 22/7
        let x2 = SqrtExt::<f64, Sqrt<f64, 6>>::new(0.0, core::f64::consts::PI);
        let mut iter = DevelopContinuedFraction::new(Ratio::from(x2));
        for _ in 0..10 {
            let next = iter.next().unwrap();
            //std::println!("{}", next);
            assert!(next.value.is_finite() && next.ext.is_finite());
        }
        #[cfg(any(feature = "std", feature = "libm"))]
        {
            // mostly testing that there is no infinite loops:
            let mut iter =
                DevelopContinuedFraction::new(Ratio::from(x2)).continued_fraction(Zero::zero());
            for _ in 0..10 {
                let v = iter.next().unwrap() / SqrtExt::new(0.0, 1.0);
                let vf: f64 = v.to_approx();
                let err = ((core::f64::consts::PI - vf) / core::f64::consts::PI).abs();
                //std::println!("{v} error: {err:.2e}");
                assert!(err < 0.1, "{v} error: {err:.2e}");
            }
            // test conversions
            let phi = SqrtExt::<_, Sqrt<_, 5>>::new(Ratio::new(1, 2), Ratio::new(1, 2));
            let phi2: Ratio<SqrtExt<i64, Sqrt<_, 5>>> = phi.into();
            let f: f64 = phi.to_approx();
            assert_eq!(f, phi2.to_approx());
            assert_eq!(f, 0.5 + 0.5 * 5.0f64.sqrt());
            assert_eq!(phi, phi2.into());
            assert!(phi2.denom.is_integral());
            // test an example where a more difficult conversion is needed
            let x: Ratio<SqrtExt<i64, Sqrt<_, 5>>> =
                Ratio::new(SqrtExt::new(1, 3), SqrtExt::new(7, 9));
            let x2: SqrtExt<Ratio<i64>, Sqrt<_, 5>> = x.into();
            let f: f64 = x.to_approx();
            assert_eq!(f, x2.to_approx());
            assert_eq!(x, x2.into());
            assert!(x2.value.denom != 1);
            assert!(x2.ext.denom != 1);
        }
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn test_approx_float() {
        let f_list = [
            core::f32::consts::PI,
            core::f32::consts::E,
            core::f32::consts::LN_2,
            1.5,
            0.7,
            12500.7,
            0.0001,
        ];
        const N: u64 = 5;
        let tol = 1e-6f32;
        for f in f_list {
            let x = SqrtExt::<i64, Sqrt<_, N>>::from_approx(f, tol).unwrap();
            let xf: f32 = x.to_approx();
            assert!(
                (xf - f).abs_sqr() <= tol * tol,
                "failed with\n{} vs\n{}\n{}",
                xf,
                f,
                x
            );
            //println!("{}", x);
        }
        assert!(SqrtExt::<i32, Sqrt<_, N>>::from_approx(f32::NAN, tol).is_none());
        assert!(SqrtExt::<i32, Sqrt<_, N>>::from_approx(f32::INFINITY, tol).is_none());
        assert!(SqrtExt::<i32, Sqrt<_, N>>::from_approx(f32::NEG_INFINITY, tol).is_none());
    }

    #[test]
    fn test_complex_sqrt_ext() {
        type T = SqrtExt<Complex<i32>, Sqrt<Complex<i32>, 5>>;
        type TR = SqrtExt<i32, Sqrt<i32, 5>>;
        let x = T::new(Complex::i() * 4, Complex::i() * 4);
        let y = T::new(Complex::i() * 2, Complex::i() * 2);
        assert_eq!(x / y, Complex::from(2i32).into());
        assert!((x % y).is_zero());
        let x = T::new(Complex::i() * -4, Complex::one() * 4);
        let y = T::new(Complex::one() * 2, Complex::i() * 2);
        assert_eq!(x / y, Complex::new(0, -2i32).into());
        assert!((x % y).is_zero());
        assert_eq!(x.abs_sqr(), SqrtExt::new(16 + 16 * 5, 0));
        // let u = T::unit(); // what is the complex version of this???
        //std::println!("{}", x / y);

        // test Complex<SqrtExt<...>> conversion
        let z: Complex<TR> = x.into();
        let w: Complex<TR> = y.into();
        assert_eq!(x, z.into());
        assert_eq!(y, w.into());
        assert_eq!(z / w, Complex::new(Zero::zero(), TR::new((-2i32).into(), Zero::zero())));
        // test Complex<SqrtExt<...>> e.g. with Euclid
        let a = Complex::new(TR::new(-9, 3), TR::new(3, -2));
        let b = Complex::new(TR::new(3, -2), TR::new(1, 1));
        let (q, r) = a.div_rem_euclid(&b);
        assert_eq!(a, q * b + r);
        assert!(r.abs_sqr() < b.abs_sqr());
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_hash() {
        assert_ne!(hash(&ZERO), hash(&ONE));
        assert_eq!(hash(&PHI), hash(&PHI2)); // assuming hash is equal for u32 and i32
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_display_complex_sqrt_ext() {
        type T = SqrtExt<Complex<i32>, Sqrt<Complex<i32>, 5>>;
        let x = T::new(One::one(), One::one());
        assert_eq!("1+0i+√(5+0i)", std::format!("{x}"));
        let x = T::new(One::one(), Zero::zero());
        assert_eq!("1+0i", std::format!("{x}"));
        let x = T::new(complex!(-7 - 1 i), complex!(2 - 3 i));
        assert_eq!("-7-1i+(2-3i)√(5+0i)", std::format!("{x}"));
        let x = T::new(complex!(-7 - 1 i), complex!(-2 - 3 i));
        assert_eq!("-7-1i+(-2-3i)√(5+0i)", std::format!("{x}"));
    }
}

#[cfg(feature = "serde")]
mod serde {
    use super::*;
    
    #[test]
    fn test_serde_complex() {
        let c = complex!(1.7 + 2.4 i);
        let s = serde_yaml::to_string(&c).unwrap();
        let c2: Complex<f64> = serde_yaml::from_str(&s).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    #[cfg(feature = "quaternion")]
    fn test_serde_quaternion() {
        let c = quaternion!(1.7 + i 2.4 + j -1.3 + k 7.1);
        let s = serde_yaml::to_string(&c).unwrap();
        let c2: Quaternion<f64> = serde_yaml::from_str(&s).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    #[cfg(feature = "rational")]
    fn test_serde_rational() {
        let r = Ratio::new_raw(12, 4);
        let s = serde_yaml::to_string(&r).unwrap();
        let r2: Ratio<i32> = serde_yaml::from_str(&s).unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    #[cfg(feature = "rational")]
    fn test_serde_extension() {
        let r = SqrtExt::<_, Sqrt<_, 5>>::new(12, 4);
        let s = serde_yaml::to_string(&r).unwrap();
        let r2: SqrtExt::<_, Sqrt<_, 5>> = serde_yaml::from_str(&s).unwrap();
        assert_eq!(r, r2);
    }
}

#[cfg(feature = "bytemuck")]
mod bytemuck {
    use super::*;

    #[test]
    fn test_bytemuck_complex() {
        let c = complex!(1.2f64 + 2.7 i);
        let c_arr = [c];
        let bytes: &[u8] = ::bytemuck::cast_slice(&c_arr);
        let recovered: &[Complex<f64>] = ::bytemuck::cast_slice(bytes);
        assert_eq!(recovered[0], c);
    }
    #[test]
    #[cfg(feature = "quaternion")]
    fn test_bytemuck_quaternion() {
        let c = quaternion!(1.2f64 + i 2.7 + j 3.1 + k -1.4);
        let c_arr = [c];
        let bytes: &[u8] = ::bytemuck::cast_slice(&c_arr);
        let recovered: &[Quaternion<f64>] = ::bytemuck::cast_slice(bytes);
        assert_eq!(recovered[0], c);
    }
    #[test]
    #[cfg(feature = "rational")]
    fn test_bytemuck_rational() {
        let c = Ratio::<i32>::new_raw(100, -70);
        let c_arr = [c];
        let bytes: &[u8] = ::bytemuck::cast_slice(&c_arr);
        let recovered: &[Ratio<i32>] = ::bytemuck::cast_slice(bytes);
        assert_eq!(recovered[0], c);
    }
    #[test]
    #[cfg(feature = "rational")]
    fn test_bytemuck_extension() {
        let c = SqrtExt::<i32, Sqrt<_, 5>>::new(100, -70);
        let c_arr = [c];
        let bytes: &[u8] = ::bytemuck::cast_slice(&c_arr);
        // Note, that this is a way to change the constant. This would be the users fault.
        let recovered: &[SqrtExt::<i32, Sqrt<_, 5>>] = ::bytemuck::cast_slice(bytes);
        assert_eq!(recovered[0], c);
    }
}