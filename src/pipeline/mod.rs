use crate::Color;

mod scalar;
pub use scalar::Scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "sse2")]
mod sse2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "sse2")]
pub use sse2::Sse2;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use neon::Neon;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use neon::Neon;

pub trait Pipeline {
    fn build(color: Color) -> Self;
    fn reset(&mut self);
    fn fill(&mut self, dst: &mut [u32]);
    fn fill_edge(&mut self, dst: &mut [u32], cvg: &mut [f32]);
}
