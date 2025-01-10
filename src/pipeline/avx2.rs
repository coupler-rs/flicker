#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::Pipeline;
use crate::Color;

pub struct Avx2 {
    color: Color,
    a: __m256,
    r: __m256,
    g: __m256,
    b: __m256,
    accum: f32,
    cvg: f32,
}

impl Avx2 {
    #[inline(always)]
    fn accum(&mut self, deltas: __m256) -> __m256 {
        unsafe {
            let shifted = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(deltas), 4));
            let sum1 = _mm256_add_ps(deltas, shifted);
            let shifted = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(sum1), 8));
            let sum2 = _mm256_add_ps(sum1, shifted);
            let lower = _mm256_castps256_ps128(sum2);
            let total = _mm_shuffle_ps(lower, lower, 0xFF);
            let carry = _mm256_insertf128_ps(_mm256_setzero_ps(), total, 1);
            let sum3 = _mm256_add_ps(sum2, carry);
            let sum = _mm256_add_ps(_mm256_set1_ps(self.accum), sum3);

            let abs = _mm256_and_ps(sum, _mm256_castsi256_ps(_mm256_set1_epi32(!(1 << 31))));
            let mask = _mm256_min_ps(abs, _mm256_set1_ps(1.0));

            let sum_upper = _mm256_extractf128_ps(sum, 1);
            self.accum = _mm_cvtss_f32(_mm_shuffle_ps(sum_upper, sum_upper, 0x03));
            let mask_upper = _mm256_extractf128_ps(mask, 1);
            self.cvg = _mm_cvtss_f32(_mm_shuffle_ps(mask_upper, mask_upper, 0x03));

            mask
        }
    }

    #[inline(always)]
    fn blend(&self, dst: __m256i, mask: __m256) -> __m256i {
        unsafe {
            let byte_mask = _mm256_set1_epi32(0xFF);
            let a_dst = _mm256_cvtepi32_ps(_mm256_srli_epi32(dst, 24));
            let r_dst = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(dst, 16), byte_mask));
            let g_dst = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(dst, 8), byte_mask));
            let b_dst = _mm256_cvtepi32_ps(_mm256_and_si256(dst, byte_mask));

            let a_unit = _mm256_mul_ps(self.a, _mm256_set1_ps(1.0 / 255.0));
            let inv_a = _mm256_sub_ps(_mm256_set1_ps(1.0), _mm256_mul_ps(mask, a_unit));
            let a_out = _mm256_add_ps(_mm256_mul_ps(self.a, mask), _mm256_mul_ps(inv_a, a_dst));
            let r_out = _mm256_add_ps(_mm256_mul_ps(self.r, mask), _mm256_mul_ps(inv_a, r_dst));
            let g_out = _mm256_add_ps(_mm256_mul_ps(self.g, mask), _mm256_mul_ps(inv_a, g_dst));
            let b_out = _mm256_add_ps(_mm256_mul_ps(self.b, mask), _mm256_mul_ps(inv_a, b_dst));

            let out = _mm256_slli_epi32(_mm256_cvtps_epi32(a_out), 24);
            let out = _mm256_or_si256(out, _mm256_slli_epi32(_mm256_cvtps_epi32(r_out), 16));
            let out = _mm256_or_si256(out, _mm256_slli_epi32(_mm256_cvtps_epi32(g_out), 8));
            let out = _mm256_or_si256(out, _mm256_cvtps_epi32(b_out));

            out
        }
    }
}

impl Pipeline for Avx2 {
    #[inline(always)]
    fn build(color: Color) -> Self {
        unsafe {
            let a_unit = color.a() as f32 * (1.0 / 255.0);

            Avx2 {
                color,
                a: _mm256_set1_ps(color.a() as f32),
                r: _mm256_set1_ps(a_unit * color.r() as f32),
                g: _mm256_set1_ps(a_unit * color.g() as f32),
                b: _mm256_set1_ps(a_unit * color.b() as f32),
                accum: 0.0,
                cvg: 0.0,
            }
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.accum = 0.0;
        self.cvg = 0.0;
    }

    #[inline(always)]
    fn fill(&mut self, dst: &mut [u32]) {
        if self.cvg > 254.5 / 255.0 && self.color.a() == 255 {
            dst.fill(self.color.into());
        } else if self.cvg > 0.5 / 255.0 {
            unsafe {
                let mut dst_chunks = dst.chunks_exact_mut(8);

                for dst_chunk in &mut dst_chunks {
                    let mask = _mm256_set1_ps(self.cvg);
                    let dst = _mm256_loadu_si256(dst_chunk.as_ptr() as *const __m256i);
                    let out = self.blend(dst, mask);
                    _mm256_storeu_si256(dst_chunk.as_mut_ptr() as *mut __m256i, out);
                }

                let mut dst_rem = dst_chunks.into_remainder();

                if dst_rem.len() >= 4 {
                    let mask = _mm256_set1_ps(self.cvg);
                    let dst128 = _mm_loadu_si128(dst_rem.as_ptr() as *const __m128i);
                    let dst = _mm256_castsi128_si256(dst128);
                    let out = self.blend(dst, mask);
                    let out128 = _mm256_castsi256_si128(out);
                    _mm_storeu_si128(dst_rem.as_mut_ptr() as *mut __m128i, out128);

                    dst_rem = &mut dst_rem[4..];
                }

                for pixel in dst_rem {
                    let mask = _mm256_set1_ps(self.cvg);
                    let dst128 = _mm_loadu_si32(pixel as *const u32 as *const u8);
                    let dst = _mm256_castsi128_si256(dst128);
                    let out = self.blend(dst, mask);
                    let out128 = _mm256_castsi256_si128(out);
                    _mm_storeu_si32(pixel as *mut u32 as *mut u8, out128);
                }
            }
        }
    }

    #[inline(always)]
    fn fill_edge(&mut self, dst: &mut [u32], cvg: &mut [f32]) {
        let mut dst_chunks = dst.chunks_exact_mut(8);
        let mut cvg_chunks = cvg.chunks_exact_mut(8);

        for (dst_chunk, cvg_chunk) in (&mut dst_chunks).zip(&mut cvg_chunks) {
            unsafe {
                let deltas = _mm256_loadu_ps(cvg_chunk.as_ptr());
                let mask = self.accum(deltas);
                _mm256_storeu_ps(cvg_chunk.as_mut_ptr(), _mm256_setzero_ps());

                let dst = _mm256_loadu_si256(dst_chunk.as_ptr() as *const __m256i);
                let out = self.blend(dst, mask);
                _mm256_storeu_si256(dst_chunk.as_mut_ptr() as *mut __m256i, out);
            }
        }

        let mut dst_rem = dst_chunks.into_remainder();
        let mut cvg_rem = cvg_chunks.into_remainder();

        if dst_rem.len() >= 4 && cvg_rem.len() >= 4 {
            unsafe {
                let deltas = _mm256_castps128_ps256(_mm_loadu_ps(cvg_rem.as_ptr()));
                let mask = self.accum(deltas);
                _mm_storeu_ps(cvg_rem.as_mut_ptr(), _mm_setzero_ps());

                let dst128 = _mm_loadu_si128(dst_rem.as_ptr() as *const __m128i);
                let dst = _mm256_castsi128_si256(dst128);
                let out = self.blend(dst, mask);
                let out128 = _mm256_castsi256_si128(out);
                _mm_storeu_si128(dst_rem.as_mut_ptr() as *mut __m128i, out128);
            }

            dst_rem = &mut dst_rem[4..];
            cvg_rem = &mut cvg_rem[4..];
        }

        for (pixel, delta) in std::iter::zip(dst_rem, cvg_rem) {
            unsafe {
                let deltas = _mm256_castps128_ps256(_mm_load_ss(delta));
                let mask = self.accum(deltas);
                _mm_store_ss(delta, _mm_set_ss(0.0));

                let dst128 = _mm_loadu_si32(pixel as *const u32 as *const u8);
                let dst = _mm256_castsi128_si256(dst128);
                let out = self.blend(dst, mask);
                let out128 = _mm256_castsi256_si128(out);
                _mm_storeu_si32(pixel as *mut u32 as *mut u8, out128);
            }
        }
    }
}
