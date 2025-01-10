use crate::Color;

mod neon;
mod scalar;

pub use neon::Neon;
pub use scalar::Scalar;

pub trait Pipeline {
    fn build(color: Color) -> Self;
    fn reset(&mut self);
    fn fill(&mut self, dst: &mut [u32]);
    fn fill_edge(&mut self, dst: &mut [u32], cvg: &mut [f32]);
}
