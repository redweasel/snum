use core::fmt;
use core::fmt::Write;

pub struct NumberDetector {
    width: usize,
    prefix_counter: usize,
    is_alphanum: bool,
    is_number: bool,
    no_number_chars: &'static str,
    prefix: [char; 2],
    first: Option<char>,
    first_tmp: [u8; 1],
    second: Option<char>,
}

#[allow(dead_code)]
impl NumberDetector {
    pub fn new(no_number_chars: &'static str, prefix: &str) -> Self {
        Self {
            width: 0,
            prefix_counter: 0,
            is_alphanum: true,
            is_number: true,
            no_number_chars,
            prefix: core::array::from_fn(|i| prefix.chars().nth(i).unwrap_or('\0')),
            first: None,
            first_tmp: [0],
            second: None,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn is_one(&self) -> bool {
        if self.second.is_none() {
            self.first == Some('1')
        } else {
            (self.first == Some('-') || self.first == Some('+')) && self.second == Some('1')
        }
    }

    pub fn is_zero(&self) -> bool {
        if self.second.is_none() {
            self.first == Some('0')
        } else {
            (self.first == Some('-') || self.first == Some('+')) && self.second == Some('0')
        }
    }

    pub fn sign(&self) -> Option<&str> {
        self.first
            .filter(|&c| c == '-' || c == '+')
            .map(|_| core::str::from_utf8(&self.first_tmp).unwrap())
    }

    pub fn has_prefix(&self) -> bool {
        !self.prefix.is_empty() && self.prefix_counter == self.prefix.len()
    }

    pub fn is_number(&self) -> bool {
        self.is_number
    }

    pub fn is_alphanum(&self) -> bool {
        self.is_alphanum
    }

    pub fn starts_alphanumeric(&self) -> bool {
        self.first.is_some_and(char::is_alphanumeric)
    }
}

impl fmt::Write for NumberDetector {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for c in s.chars() {
            if self.first.is_some() || (c != '-' && c != '+') {
                if self.width - self.prefix_counter <= 1 {
                    if let Some(pc) = self.prefix.get(self.prefix_counter) {
                        if pc == &c {
                            self.prefix_counter += 1;
                        } else {
                            self.prefix_counter = 0;
                        }
                    }
                }
                self.is_alphanum &= c.is_ascii_alphanumeric();
                self.is_number &= !self.no_number_chars.contains(c);
            }
            self.width += 1;
            if self.first.is_some() {
                self.second.get_or_insert(c);
            } else {
                self.first = Some(c);
                self.first_tmp[0] = c as u8;
            }
        }
        Ok(())
    }
}

struct SkipWriteN<'a, 'b>(&'b mut fmt::Formatter<'a>, usize);

impl fmt::Write for SkipWriteN<'_, '_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if let Some((i, _)) = s.char_indices().nth(self.1) {
            self.0.write_str(&s[i..])?;
        }
        self.1 = self.1.saturating_sub(s.chars().count());
        Ok(())
    }
}

pub struct Parenthesis<T: Sized>(T, bool);

impl<T: fmt::Display> fmt::Display for Parenthesis<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.1 {
            write!(f, "(")?;
        }
        self.0.fmt(f)?;
        if self.1 {
            write!(f, ")")?;
        }
        Ok(())
    }
}

#[inline(never)]
pub fn pad_expr(
    f: &mut fmt::Formatter<'_>,
    prefix: &str,
    buf_args: fmt::Arguments<'_>,
) -> fmt::Result {
    if f.width().is_none() && !f.sign_plus() {
        return write!(f, "{buf_args}");
    }
    // pad by doing the formatting twice
    let mut dec = NumberDetector::new("", prefix);
    write!(&mut dec, "{buf_args}")?;
    let mut width = dec.width();

    let sign = dec.sign();
    let add_sign = if f.sign_plus() && sign.is_none() {
        width += 1;
        "+"
    } else {
        ""
    };
    let sign = sign.unwrap_or(add_sign);

    // The `width` field is more of a `min-width` parameter at this point.
    let min = f.width().unwrap_or(0);
    let align = f.align().unwrap_or(fmt::Alignment::Right);
    if width >= min {
        // We're over the minimum width, so then we can just write the bytes.
        f.write_str(add_sign)?;
        write!(f, "{buf_args}")
    } else if f.sign_aware_zero_pad() && dec.starts_alphanumeric() {
        // skip the sign and prefix -0x
        write!(f, "{sign}")?;
        let mut skip_n = sign.len() - add_sign.len();
        if dec.has_prefix() {
            write!(f, "{prefix}")?;
            skip_n += prefix.len();
        }
        // Then add the 0 padding.
        for _ in width..min {
            f.write_char('0')?;
        }
        write!(SkipWriteN(f, skip_n), "{buf_args}")
    } else {
        // Otherwise, the sign and prefix goes after the padding
        // drop the fill character to work around precision and wrong default alignment
        // (can't access the fill character from the public API)
        let fill = f.fill();
        let l = (min - width) / 2;
        let r = (min - width).div_ceil(2);
        match align {
            fmt::Alignment::Right => {
                for _ in 0..l + r {
                    f.write_char(fill)?;
                }
                write!(f, "{add_sign}{buf_args}")?;
            }
            fmt::Alignment::Center => {
                for _ in 0..l {
                    f.write_char(fill)?;
                }
                write!(f, "{add_sign}{buf_args}")?;
                for _ in 0..r {
                    f.write_char(fill)?;
                }
            }
            fmt::Alignment::Left => {
                write!(f, "{add_sign}{buf_args}")?;
                for _ in 0..l + r {
                    f.write_char(fill)?;
                }
            }
        }
        Ok(())
    }
}

#[inline(never)]
pub fn fmt_poly(
    f: &mut fmt::Formatter<'_>,
    prefix: &str,
    pretty: bool,
    monoms: &[(fmt::Arguments<'_>, &str)],
) -> fmt::Result {
    struct DisplayPoly<'a>(&'a [(fmt::Arguments<'a>, &'a str)], &'a str, bool);
    impl fmt::Display for DisplayPoly<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut first = true;
            for (args, monom) in self.0 {
                // check if args is a number and add parenthesis and sign if necessary
                let mut args_dec = NumberDetector::new("+-/*", self.1);
                write!(&mut args_dec, "{args}")?;
                if !self.2 || !args_dec.is_zero() {
                    let parens = !monom.is_empty() && !args_dec.is_number();
                    let sign = if first && !f.sign_plus() { "" } else { "+" };
                    let sign_add = if parens {
                        "+"
                    } else {
                        args_dec.sign().map_or(sign, |_| "")
                    };
                    if self.2 && !monom.is_empty() && args_dec.is_one() {
                        write!(f, "{}{monom}", args_dec.sign().unwrap_or(sign))
                    } else {
                        write!(f, "{sign_add}{}{monom}", Parenthesis(args, parens))
                    }?;
                    first = false;
                }
            }
            if first {
                write!(f, "0")?;
            }
            Ok(())
        }
    }
    pad_expr(
        f,
        prefix,
        format_args!("{}", DisplayPoly(monoms, prefix, pretty)),
    )
}

mod complex {
    use super::*;
    use crate::Complex;

    #[inline(never)]
    fn fmt_re_im(
        f: &mut fmt::Formatter<'_>,
        real: fmt::Arguments<'_>,
        imag: fmt::Arguments<'_>,
        pretty: bool,
        prefix: &str,
    ) -> fmt::Result {
        fmt_poly(f, prefix, pretty, &[(real, ""), (imag, "i")])
    }

    // string conversions
    macro_rules! impl_display {
        ($Display: ident, $s: literal, $pre: literal) => {
            impl<T: fmt::$Display> fmt::$Display for Complex<T> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    let prefix = $pre;
                    let pretty = stringify!($Display) == "Display" && f.alternate();
                    return if f.alternate() {
                        if let Some(prec) = f.precision() {
                            fmt_re_im(
                                f,
                                format_args!(concat!("{:#.1$", $s, "}"), self.re, prec),
                                format_args!(concat!("{:#.1$", $s, "}"), self.im, prec),
                                pretty,
                                prefix,
                            )
                        } else {
                            fmt_re_im(
                                f,
                                format_args!(concat!("{:#", $s, "}"), self.re),
                                format_args!(concat!("{:#", $s, "}"), self.im),
                                pretty,
                                prefix,
                            )
                        }
                    } else {
                        if let Some(prec) = f.precision() {
                            fmt_re_im(
                                f,
                                format_args!(concat!("{:.1$", $s, "}"), self.re, prec),
                                format_args!(concat!("{:.1$", $s, "}"), self.im, prec),
                                pretty,
                                prefix,
                            )
                        } else {
                            fmt_re_im(
                                f,
                                format_args!(concat!("{:", $s, "}"), self.re),
                                format_args!(concat!("{:", $s, "}"), self.im),
                                pretty,
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
}

#[cfg(feature = "quaternion")]
mod quaternion {
    use super::*;
    use crate::quaternion::Quaternion;

    #[inline(never)]
    fn fmt_quat(
        f: &mut fmt::Formatter<'_>,
        real: fmt::Arguments<'_>,
        imag_i: fmt::Arguments<'_>,
        imag_j: fmt::Arguments<'_>,
        imag_k: fmt::Arguments<'_>,
        pretty: bool,
        prefix: &str,
    ) -> fmt::Result {
        fmt_poly(
            f,
            prefix,
            pretty,
            &[(real, ""), (imag_i, "i"), (imag_j, "j"), (imag_k, "k")],
        )
    }

    // string conversions
    macro_rules! impl_display {
        ($Display: ident, $s: literal, $pre: literal) => {
            impl<T: fmt::$Display> fmt::$Display for Quaternion<T> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    let prefix = $pre;
                    let pretty = stringify!($Display) == "Display" && f.alternate();
                    return if let Some(prec) = f.precision() {
                        if f.alternate() {
                            fmt_quat(
                                f,
                                format_args!(concat!("{:#.1$", $s, "}"), self.re, prec),
                                format_args!(concat!("{:#.1$", $s, "}"), self.im_i, prec),
                                format_args!(concat!("{:#.1$", $s, "}"), self.im_j, prec),
                                format_args!(concat!("{:#.1$", $s, "}"), self.im_k, prec),
                                pretty,
                                prefix,
                            )
                        } else {
                            fmt_quat(
                                f,
                                format_args!(concat!("{:.1$", $s, "}"), self.re, prec),
                                format_args!(concat!("{:.1$", $s, "}"), self.im_i, prec),
                                format_args!(concat!("{:.1$", $s, "}"), self.im_j, prec),
                                format_args!(concat!("{:.1$", $s, "}"), self.im_k, prec),
                                pretty,
                                prefix,
                            )
                        }
                    } else {
                        if f.alternate() {
                            fmt_quat(
                                f,
                                format_args!(concat!("{:#", $s, "}"), self.re),
                                format_args!(concat!("{:#", $s, "}"), self.im_i),
                                format_args!(concat!("{:#", $s, "}"), self.im_j),
                                format_args!(concat!("{:#", $s, "}"), self.im_k),
                                pretty,
                                prefix,
                            )
                        } else {
                            fmt_quat(
                                f,
                                format_args!(concat!("{:", $s, "}"), self.re),
                                format_args!(concat!("{:", $s, "}"), self.im_i),
                                format_args!(concat!("{:", $s, "}"), self.im_j),
                                format_args!(concat!("{:", $s, "}"), self.im_k),
                                pretty,
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
}

#[cfg(feature = "rational")]
mod rational {
    use super::*;
    use crate::extension::*;
    use crate::rational::*;
    use crate::{Num, One, Zero};

    #[inline(never)]
    fn fmt_ratio(
        f: &mut fmt::Formatter<'_>,
        numer_zero: bool,
        denom_zero: bool,
        numer_args: fmt::Arguments<'_>,
        denom_args: fmt::Arguments<'_>,
        prefix: &str,
    ) -> fmt::Result {
        let mut numer_dec = NumberDetector::new("+-/", prefix);
        write!(&mut numer_dec, "{numer_args}")?;
        if denom_zero {
            if numer_zero {
                pad_expr(f, prefix, format_args!("NaN"))
            } else if numer_dec.is_one() {
                pad_expr(
                    f,
                    prefix,
                    format_args!("{}∞", numer_dec.sign().unwrap_or("")),
                )
            } else {
                pad_expr(
                    f,
                    prefix,
                    format_args!("{}∞", Parenthesis(numer_args, !numer_dec.is_number())),
                )
            }
        } else {
            // note, this is missing a lot of annotations like the precision in case of floats.
            let mut denom_dec = NumberDetector::new("+-/*", prefix);
            write!(&mut denom_dec, "{denom_args}")?;
            // Note, the signs can not be processed as strings, as the expression might be e.g. -1+i
            pad_expr(
                f,
                prefix,
                format_args!(
                    "{}/{}",
                    Parenthesis(numer_args, !numer_dec.is_number()),
                    Parenthesis(denom_args, !denom_dec.is_number())
                ),
            )
        }
    }

    // String conversions
    macro_rules! impl_formatting {
        ($Display:ident, $prefix:expr, $fmt_str:expr) => {
            impl<T: fmt::$Display + Clone + Zero + One> fmt::$Display for Ratio<T> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    if self.denom.is_one() {
                        return self.numer.fmt(f);
                    }
                    if let Some(prec) = f.precision() {
                        if f.alternate() {
                            fmt_ratio(
                                f,
                                self.numer.is_zero(),
                                self.denom.is_zero(),
                                format_args!(
                                    concat!("{:#.prec$", $fmt_str, "}"),
                                    self.numer,
                                    prec = prec
                                ),
                                format_args!(
                                    concat!("{:#.prec$", $fmt_str, "}"),
                                    self.denom,
                                    prec = prec
                                ),
                                $prefix,
                            )
                        } else {
                            fmt_ratio(
                                f,
                                self.numer.is_zero(),
                                self.denom.is_zero(),
                                format_args!(
                                    concat!("{:.prec$", $fmt_str, "}"),
                                    self.numer,
                                    prec = prec
                                ),
                                format_args!(
                                    concat!("{:.prec$", $fmt_str, "}"),
                                    self.denom,
                                    prec = prec
                                ),
                                $prefix,
                            )
                        }
                    } else {
                        if f.alternate() {
                            fmt_ratio(
                                f,
                                self.numer.is_zero(),
                                self.denom.is_zero(),
                                format_args!(concat!("{:#", $fmt_str, "}"), self.numer),
                                format_args!(concat!("{:#", $fmt_str, "}"), self.denom),
                                $prefix,
                            )
                        } else {
                            fmt_ratio(
                                f,
                                self.numer.is_zero(),
                                self.denom.is_zero(),
                                format_args!(concat!("{:", $fmt_str, "}"), self.numer),
                                format_args!(concat!("{:", $fmt_str, "}"), self.denom),
                                $prefix,
                            )
                        }
                    }
                }
            }
        };
    }

    impl_formatting!(Display, "", "");
    impl_formatting!(Octal, "0o", "o");
    impl_formatting!(Binary, "0b", "b");
    impl_formatting!(LowerHex, "0x", "x");
    impl_formatting!(UpperHex, "0x", "X");
    impl_formatting!(LowerExp, "", "e");
    impl_formatting!(UpperExp, "", "E");

    #[inline(never)]
    fn fmt_sqrtext(
        f: &mut fmt::Formatter<'_>,
        value_zero: bool,
        ext_one: bool,
        value_args: fmt::Arguments<'_>,
        ext_args: fmt::Arguments<'_>,
        sqr_args: fmt::Arguments<'_>,
        prefix: &str,
    ) -> fmt::Result {
        let mut ext_dec = NumberDetector::new("+-/", prefix);
        write!(&mut ext_dec, "{ext_args}")?;
        let mut sqr_dec = NumberDetector::new("+-/", prefix);
        write!(&mut sqr_dec, "{sqr_args}")?;
        let sqr = Parenthesis(sqr_args, !sqr_dec.is_alphanum());

        if ext_one {
            if value_zero {
                pad_expr(f, prefix, format_args!("√{sqr}"))
            } else {
                pad_expr(f, prefix, format_args!("{value_args}+√{sqr}"))
            }
        } else if value_zero {
            pad_expr(
                f,
                prefix,
                format_args!("{}√{sqr}", Parenthesis(ext_args, !ext_dec.is_number())),
            )
        } else if ext_dec.is_number() {
            let add_sign = if ext_dec.sign().is_none() { "+" } else { "" };
            pad_expr(
                f,
                prefix,
                format_args!("{value_args}{add_sign}{ext_args}√{sqr}"),
            )
        } else {
            pad_expr(f, prefix, format_args!("{value_args}+({ext_args})√{sqr}"))
        }
    }

    // String conversions
    macro_rules! impl_formatting {
        ($Display:ident, $prefix:expr, $fmt_str:expr) => {
            impl<T: Num + fmt::$Display + Clone + Zero + One, E: SqrtConst<T>> fmt::$Display
                for SqrtExt<T, E>
            {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    if self.ext.is_zero() {
                        return <T as fmt::$Display>::fmt(&self.value, f);
                    }
                    if let Some(prec) = f.precision() {
                        if f.alternate() {
                            fmt_sqrtext(
                                f,
                                self.value.is_zero(),
                                self.ext.is_one(),
                                format_args!(
                                    concat!("{:#.prec$", $fmt_str, "}"),
                                    self.value,
                                    prec = prec
                                ),
                                format_args!(
                                    concat!("{:#.prec$", $fmt_str, "}"),
                                    self.ext,
                                    prec = prec
                                ),
                                format_args!(concat!("{:#", $fmt_str, "}"), E::sqr()),
                                $prefix,
                            )
                        } else {
                            fmt_sqrtext(
                                f,
                                self.value.is_zero(),
                                self.ext.is_one(),
                                format_args!(
                                    concat!("{:.prec$", $fmt_str, "}"),
                                    self.value,
                                    prec = prec
                                ),
                                format_args!(
                                    concat!("{:.prec$", $fmt_str, "}"),
                                    self.ext,
                                    prec = prec
                                ),
                                format_args!(concat!("{:", $fmt_str, "}"), E::sqr()),
                                $prefix,
                            )
                        }
                    } else {
                        if f.alternate() {
                            fmt_sqrtext(
                                f,
                                self.value.is_zero(),
                                self.ext.is_one(),
                                format_args!(concat!("{:#", $fmt_str, "}"), self.value),
                                format_args!(concat!("{:#", $fmt_str, "}"), self.ext),
                                format_args!(concat!("{:#", $fmt_str, "}"), E::sqr()),
                                $prefix,
                            )
                        } else {
                            fmt_sqrtext(
                                f,
                                self.value.is_zero(),
                                self.ext.is_one(),
                                format_args!(concat!("{:", $fmt_str, "}"), self.value),
                                format_args!(concat!("{:", $fmt_str, "}"), self.ext),
                                format_args!(concat!("{:", $fmt_str, "}"), E::sqr()),
                                $prefix,
                            )
                        }
                    }
                }
            }
        };
    }

    impl_formatting!(Display, "", "");
    impl_formatting!(Octal, "0o", "o");
    impl_formatting!(Binary, "0b", "b");
    impl_formatting!(LowerHex, "0x", "x");
    impl_formatting!(UpperHex, "0x", "X");
    impl_formatting!(LowerExp, "", "e");
    impl_formatting!(UpperExp, "", "E");
}
