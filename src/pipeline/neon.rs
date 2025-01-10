use std::arch::aarch64::*;

use super::Pipeline;
use crate::Color;

pub struct Neon {
    color: Color,
    a: float32x4_t,
    r: float32x4_t,
    g: float32x4_t,
    b: float32x4_t,
    accum: f32,
    cvg: f32,
}

impl Neon {
    #[inline(always)]
    fn accum(&mut self, deltas: float32x4_t) -> float32x4_t {
        unsafe {
            let zeros = vdupq_n_f32(0.0);
            let sum1 = vaddq_f32(deltas, vextq_f32(zeros, deltas, 3));
            let sum2 = vaddq_f32(sum1, vextq_f32(zeros, sum1, 2));
            let sum = vaddq_f32(vdupq_n_f32(self.accum), sum2);
            let mask = vminq_f32(vabsq_f32(sum), vdupq_n_f32(1.0));

            self.accum = vgetq_lane_f32(sum, 3);
            self.cvg = vgetq_lane_f32(mask, 3);

            mask
        }
    }

    #[inline(always)]
    fn blend(&self, dst: uint32x4_t, mask: float32x4_t) -> uint32x4_t {
        unsafe {
            let a_dst = vcvtq_f32_u32(vshrq_n_u32(dst, 24));
            let r_dst = vcvtq_f32_u32(vandq_u32(vshrq_n_u32(dst, 16), vdupq_n_u32(0xFF)));
            let g_dst = vcvtq_f32_u32(vandq_u32(vshrq_n_u32(dst, 8), vdupq_n_u32(0xFF)));
            let b_dst = vcvtq_f32_u32(vandq_u32(dst, vdupq_n_u32(0xFF)));

            let inv_a = vfmsq_f32(vdupq_n_f32(1.0), mask, vmulq_n_f32(self.a, 1.0 / 255.0));
            let a_out = vfmaq_f32(vmulq_f32(self.a, mask), inv_a, a_dst);
            let r_out = vfmaq_f32(vmulq_f32(self.r, mask), inv_a, r_dst);
            let g_out = vfmaq_f32(vmulq_f32(self.g, mask), inv_a, g_dst);
            let b_out = vfmaq_f32(vmulq_f32(self.b, mask), inv_a, b_dst);

            let out = vshlq_n_u32(vcvtq_u32_f32(a_out), 24);
            let out = vorrq_u32(out, vshlq_n_u32(vcvtq_u32_f32(r_out), 16));
            let out = vorrq_u32(out, vshlq_n_u32(vcvtq_u32_f32(g_out), 8));
            let out = vorrq_u32(out, vcvtq_u32_f32(b_out));

            out
        }
    }
}

impl Pipeline for Neon {
    #[inline(always)]
    fn build(color: Color) -> Self {
        unsafe {
            Neon {
                color,
                a: vdupq_n_f32(color.a() as f32),
                r: vdupq_n_f32(color.r() as f32),
                g: vdupq_n_f32(color.g() as f32),
                b: vdupq_n_f32(color.b() as f32),
                accum: 0.0,
                cvg: 0.0,
            }
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        unsafe {
            self.accum = 0.0;
            self.cvg = 0.0;
        }
    }

    #[inline(always)]
    fn fill(&mut self, dst: &mut [u32]) {
        if self.cvg > 254.5 / 255.0 && self.color.a() == 255 {
            dst.fill(self.color.into());
        } else if self.cvg > 0.5 / 255.0 {
            unsafe {
                let mut dst_chunks = dst.chunks_exact_mut(4);

                for dst_chunk in &mut dst_chunks {
                    let mask = vdupq_n_f32(self.cvg);
                    let dst = vld1q_u32(dst_chunk.as_ptr());
                    let out = self.blend(dst, mask);
                    vst1q_u32(dst_chunk.as_mut_ptr(), out);
                }

                let dst_rem = dst_chunks.into_remainder();
                for pixel in dst_rem {
                    let mask = vdupq_n_f32(self.cvg);
                    let dst = vld1q_lane_u32(pixel, vdupq_n_u32(0), 0);
                    let out = self.blend(dst, mask);
                    vst1q_lane_u32(pixel, out, 0);
                }
            }
        }
    }

    #[inline(always)]
    fn fill_edge(&mut self, dst: &mut [u32], cvg: &mut [f32]) {
        let mut dst_chunks = dst.chunks_exact_mut(4);
        let mut cvg_chunks = cvg.chunks_exact_mut(4);

        for (dst_chunk, cvg_chunk) in (&mut dst_chunks).zip(&mut cvg_chunks) {
            unsafe {
                let deltas = vld1q_f32(cvg_chunk.as_ptr());
                let mask = self.accum(deltas);
                vst1q_f32(cvg_chunk.as_mut_ptr(), vdupq_n_f32(0.0));

                let dst = vld1q_u32(dst_chunk.as_ptr());
                let out = self.blend(dst, mask);
                vst1q_u32(dst_chunk.as_mut_ptr(), out);
            }
        }

        let dst_rem = dst_chunks.into_remainder();
        let cvg_rem = cvg_chunks.into_remainder();
        for (pixel, delta) in std::iter::zip(dst_rem, cvg_rem) {
            unsafe {
                let deltas = vld1q_lane_f32(delta, vdupq_n_f32(0.0), 0);
                let mask = self.accum(deltas);
                vst1q_lane_f32(delta, vdupq_n_f32(0.0), 0);

                let dst = vld1q_lane_u32(pixel, vdupq_n_u32(0), 0);
                let out = self.blend(dst, mask);
                vst1q_lane_u32(pixel, out, 0);
            }
        }
    }
}
