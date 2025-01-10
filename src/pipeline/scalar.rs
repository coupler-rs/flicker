use super::Pipeline;
use crate::Color;

pub struct Scalar {
    color: Color,
    a: f32,
    r: f32,
    g: f32,
    b: f32,
    accum: f32,
    cvg: f32,
}

impl Scalar {
    #[inline(always)]
    fn blend(&self, dst: u32) -> u32 {
        let a_dst = (dst >> 24) as f32;
        let r_dst = ((dst >> 16) & 0xFF) as f32;
        let g_dst = ((dst >> 8) & 0xFF) as f32;
        let b_dst = (dst & 0xFF) as f32;

        let inv_a = 1.0 - self.cvg * (1.0 / 255.0) * self.a;
        let a_out = self.cvg * self.a + inv_a * a_dst;
        let r_out = self.cvg * self.r + inv_a * r_dst;
        let g_out = self.cvg * self.g + inv_a * g_dst;
        let b_out = self.cvg * self.b + inv_a * b_dst;

        (a_out as u32) << 24 | (r_out as u32) << 16 | (g_out as u32) << 8 | b_out as u32
    }
}

impl Pipeline for Scalar {
    #[inline(always)]
    fn build(color: Color) -> Self {
        let a_unit = color.a() as f32 * (1.0 / 255.0);

        Scalar {
            color,
            a: color.a() as f32,
            r: a_unit * color.r() as f32,
            g: a_unit * color.g() as f32,
            b: a_unit * color.b() as f32,
            accum: 0.0,
            cvg: 0.0,
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
            for pixel in dst {
                *pixel = self.blend(*pixel);
            }
        }
    }

    #[inline(always)]
    fn fill_edge(&mut self, dst: &mut [u32], cvg: &mut [f32]) {
        for (pixel, delta) in std::iter::zip(dst, cvg) {
            self.accum += *delta;
            self.cvg = self.accum.abs().min(1.0);
            *delta = 0.0;

            *pixel = self.blend(*pixel);
        }
    }
}
