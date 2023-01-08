#![allow(non_camel_case_types)]

use std::fmt::{self, Debug};
use std::num::Wrapping;
use std::ops::*;
use std::slice;

use super::{Arch, Float, Int, PossibleArch, Simd, SupportedArch, Task};

pub struct Scalar;

impl PossibleArch for Scalar {
    #[inline]
    fn try_specialize<T: Task>() -> Option<fn(T) -> T::Result> {
        Some(Self::specialize::<T>())
    }
}

impl SupportedArch for Scalar {
    #[inline]
    fn specialize<T: Task>() -> fn(T) -> T::Result {
        T::run::<ScalarImpl>
    }
}

struct ScalarImpl;

impl Arch for ScalarImpl {
    type f32 = f32x1;
    type u32 = u32x1;
}

#[derive(Copy, Clone, Default)]
#[repr(transparent)]
struct f32x1(f32);

impl Simd for f32x1 {
    type Elem = f32;

    const LANES: usize = 1;

    #[inline]
    fn splat(value: Self::Elem) -> Self {
        f32x1(value)
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Elem] {
        slice::from_ref(&self.0)
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        slice::from_mut(&mut self.0)
    }

    #[inline]
    fn from_slice(slice: &[Self::Elem]) -> Self {
        f32x1(slice[0])
    }

    #[inline]
    fn write_to_slice(&self, slice: &mut [Self::Elem]) {
        slice[0] = self.0;
    }
}

impl Debug for f32x1 {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_slice(), fmt)
    }
}

impl Float for f32x1 {
    #[inline]
    fn abs(self) -> Self {
        f32x1(self.0.abs())
    }

    #[inline]
    fn min(self, rhs: Self) -> Self {
        f32x1(if self.0 < rhs.0 { self.0 } else { rhs.0 })
    }

    #[inline]
    fn scan_sum(self) -> Self {
        self
    }
}

impl Add for f32x1 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        f32x1(self.0 + rhs.0)
    }
}

impl Sub for f32x1 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        f32x1(self.0 - rhs.0)
    }
}

impl Mul for f32x1 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        f32x1(self.0 * rhs.0)
    }
}

impl Div for f32x1 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        f32x1(self.0 / rhs.0)
    }
}

impl Neg for f32x1 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        f32x1(-self.0)
    }
}

impl From<u32x1> for f32x1 {
    fn from(value: u32x1) -> f32x1 {
        f32x1(value.0 .0 as f32)
    }
}

#[derive(Copy, Clone, Default)]
#[repr(transparent)]
struct u32x1(Wrapping<u32>);

impl Simd for u32x1 {
    type Elem = u32;

    const LANES: usize = 1;

    #[inline]
    fn splat(value: Self::Elem) -> Self {
        u32x1(Wrapping(value))
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Elem] {
        slice::from_ref(&self.0 .0)
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        slice::from_mut(&mut self.0 .0)
    }

    #[inline]
    fn from_slice(slice: &[Self::Elem]) -> Self {
        u32x1(Wrapping(slice[0]))
    }

    #[inline]
    fn write_to_slice(&self, slice: &mut [Self::Elem]) {
        slice[0] = self.0 .0;
    }
}

impl Debug for u32x1 {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_slice(), fmt)
    }
}

impl Int for u32x1 {}

impl Shl<usize> for u32x1 {
    type Output = Self;

    #[inline]
    fn shl(self, rhs: usize) -> Self {
        u32x1(self.0 << rhs)
    }
}

impl Shr<usize> for u32x1 {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: usize) -> Self {
        u32x1(self.0 >> rhs)
    }
}

impl BitAnd for u32x1 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        u32x1(self.0 & rhs.0)
    }
}

impl BitOr for u32x1 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        u32x1(self.0 | rhs.0)
    }
}

impl From<f32x1> for u32x1 {
    fn from(value: f32x1) -> u32x1 {
        u32x1(Wrapping(value.0 as u32))
    }
}
