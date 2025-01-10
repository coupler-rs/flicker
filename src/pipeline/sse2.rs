#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::Pipeline;
use crate::Color;

pub struct Sse2 {
    color: Color,
    a: __m128,
    r: __m128,
    g: __m128,
    b: __m128,
    accum: f32,
    cvg: f32,
}

impl Sse2 {
    #[inline(always)]
    fn accum(&mut self, deltas: __m128) -> __m128 {
        unsafe {
            let shifted = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(deltas), 4));
            let sum1 = _mm_add_ps(deltas, shifted);
            let shifted = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(sum1), 8));
            let sum2 = _mm_add_ps(sum1, shifted);
            let sum = _mm_add_ps(_mm_set1_ps(self.accum), sum2);

            let abs = _mm_and_ps(sum, _mm_castsi128_ps(_mm_set1_epi32(!(1 << 31))));
            let mask = _mm_min_ps(abs, _mm_set1_ps(1.0));

            self.accum = _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, 0x03));
            self.cvg = _mm_cvtss_f32(_mm_shuffle_ps(mask, mask, 0x03));

            mask
        }
    }

    #[inline(always)]
    fn blend(&self, dst: __m128i, mask: __m128) -> __m128i {
        unsafe {
            let byte_mask = _mm_set1_epi32(0xFF);
            let a_dst = _mm_cvtepi32_ps(_mm_srli_epi32(dst, 24));
            let r_dst = _mm_cvtepi32_ps(_mm_and_si128(_mm_srli_epi32(dst, 16), byte_mask));
            let g_dst = _mm_cvtepi32_ps(_mm_and_si128(_mm_srli_epi32(dst, 8), byte_mask));
            let b_dst = _mm_cvtepi32_ps(_mm_and_si128(dst, byte_mask));

            let a_unit = _mm_mul_ps(self.a, _mm_set1_ps(1.0 / 255.0));
            let inv_a = _mm_sub_ps(_mm_set1_ps(1.0), _mm_mul_ps(mask, a_unit));
            let a_out = _mm_add_ps(_mm_mul_ps(self.a, mask), _mm_mul_ps(inv_a, a_dst));
            let r_out = _mm_add_ps(_mm_mul_ps(self.r, mask), _mm_mul_ps(inv_a, r_dst));
            let g_out = _mm_add_ps(_mm_mul_ps(self.g, mask), _mm_mul_ps(inv_a, g_dst));
            let b_out = _mm_add_ps(_mm_mul_ps(self.b, mask), _mm_mul_ps(inv_a, b_dst));

            let out = _mm_slli_epi32(_mm_cvtps_epi32(a_out), 24);
            let out = _mm_or_si128(out, _mm_slli_epi32(_mm_cvtps_epi32(r_out), 16));
            let out = _mm_or_si128(out, _mm_slli_epi32(_mm_cvtps_epi32(g_out), 8));
            let out = _mm_or_si128(out, _mm_cvtps_epi32(b_out));

            out
        }
    }
}

impl Pipeline for Sse2 {
    #[inline(always)]
    fn build(color: Color) -> Self {
        unsafe {
            let a_unit = color.a() as f32 * (1.0 / 255.0);

            Sse2 {
                color,
                a: _mm_set1_ps(color.a() as f32),
                r: _mm_set1_ps(a_unit * color.r() as f32),
                g: _mm_set1_ps(a_unit * color.g() as f32),
                b: _mm_set1_ps(a_unit * color.b() as f32),
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
                    let mask = _mm_set1_ps(self.cvg);
                    let dst = _mm_loadu_si128(dst_chunk.as_ptr() as *const __m128i);
                    let out = self.blend(dst, mask);
                    _mm_storeu_si128(dst_chunk.as_mut_ptr() as *mut __m128i, out);
                }

                let dst_rem = dst_chunks.into_remainder();
                for pixel in dst_rem {
                    let mask = _mm_set1_ps(self.cvg);
                    let dst = _mm_loadu_si32(pixel as *const u32 as *const u8);
                    let out = self.blend(dst, mask);
                    _mm_storeu_si32(pixel as *mut u32 as *mut u8, out);
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
                let deltas = _mm_loadu_ps(cvg_chunk.as_ptr());
                let mask = self.accum(deltas);
                _mm_storeu_ps(cvg_chunk.as_mut_ptr(), _mm_setzero_ps());

                let dst = _mm_loadu_si128(dst_chunk.as_ptr() as *const __m128i);
                let out = self.blend(dst, mask);
                _mm_storeu_si128(dst_chunk.as_mut_ptr() as *mut __m128i, out);
            }
        }

        let dst_rem = dst_chunks.into_remainder();
        let cvg_rem = cvg_chunks.into_remainder();
        for (pixel, delta) in std::iter::zip(dst_rem, cvg_rem) {
            unsafe {
                let deltas = _mm_load_ss(delta);
                let mask = self.accum(deltas);
                _mm_store_ss(delta, _mm_set_ss(0.0));

                let dst = _mm_loadu_si32(pixel as *const u32 as *const u8);
                let out = self.blend(dst, mask);
                _mm_storeu_si32(pixel as *mut u32 as *mut u8, out);
            }
        }
    }
}
