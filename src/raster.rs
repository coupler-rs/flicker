use std::mem;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::simd::*;
use crate::{geom::Vec2, Color};

const CELL_SIZE: usize = 4;
const CELL_SIZE_BITS: usize = 2;

const BITMASK_SIZE: usize = u64::BITS as usize;
const BITMASK_SIZE_BITS: usize = 6;

pub struct Rasterizer {
    width: usize,
    height: usize,
    coverage: Vec<f32>,
    bitmasks_width: usize,
    bitmasks: Vec<u64>,
    min_x: isize,
    min_y: isize,
    max_x: isize,
    max_y: isize,
}

impl Rasterizer {
    pub fn with_size(width: usize, height: usize) -> Rasterizer {
        // round width up to next multiple of tile size
        let width_rounded = (width + CELL_SIZE - 1) & !(CELL_SIZE - 1);

        // round up to next multiple of bitmask size
        let bitmasks_width = (width_rounded + (CELL_SIZE * BITMASK_SIZE) - 1)
            >> (CELL_SIZE_BITS + BITMASK_SIZE_BITS);
        let bitmasks_height = height;
        let bitmasks = vec![0; bitmasks_width * bitmasks_height];

        Rasterizer {
            width: width_rounded,
            height,
            coverage: vec![0.0; width_rounded * height],
            bitmasks_width,
            bitmasks,
            min_x: width_rounded as isize >> CELL_SIZE_BITS,
            min_y: height as isize,
            max_x: 0,
            max_y: 0,
        }
    }

    #[inline]
    pub fn add_line(&mut self, p1: Vec2, p2: Vec2) {
        let mut x = (p1.x + 1.0) as isize - 1;
        let mut y = (p1.y + 1.0) as isize - 1;

        let x_end = (p2.x + 1.0) as isize - 1;
        let y_end = (p2.y + 1.0) as isize - 1;

        self.min_x = self.min_x.min(x).min(x_end);
        self.min_y = self.min_y.min(y).min(y_end);
        self.max_x = self.max_x.max(x + 1).max(x_end + 1);
        self.max_y = self.max_y.max(y).max(y_end);

        let x_inc;
        let mut x_offset;
        let x_offset_end;
        let dx;
        let area_offset;
        let area_sign;
        if p2.x > p1.x {
            x_inc = 1;
            x_offset = p1.x - x as f32;
            x_offset_end = p2.x - x_end as f32;
            dx = p2.x - p1.x;
            area_offset = 2.0;
            area_sign = -1.0;
        } else {
            x_inc = -1;
            x_offset = 1.0 - (p1.x - x as f32);
            x_offset_end = 1.0 - (p2.x - x_end as f32);
            dx = p1.x - p2.x;
            area_offset = 0.0;
            area_sign = 1.0;
        }

        let y_inc;
        let mut y_offset;
        let y_offset_end;
        let dy;
        let sign;
        if p2.y > p1.y {
            y_inc = 1;
            y_offset = p1.y - y as f32;
            y_offset_end = p2.y - y_end as f32;
            dy = p2.y - p1.y;
            sign = 1.0;
        } else {
            y_inc = -1;
            y_offset = 1.0 - (p1.y - y as f32);
            y_offset_end = 1.0 - (p2.y - y_end as f32);
            dy = p1.y - p2.y;
            sign = -1.0;
        }

        let dxdy = dx / dy;
        let dydx = dy / dx;

        let mut y_offset_for_prev_x = y_offset - dydx * x_offset;
        let mut x_offset_for_prev_y = x_offset - dxdy * y_offset;

        while x != x_end || y != y_end {
            let col = x;
            let row = y;

            let x1 = x_offset;
            let y1 = y_offset;

            let x2;
            let y2;
            if y != y_end && (x == x_end || x_offset_for_prev_y + dxdy < 1.0) {
                y_offset = 0.0;
                x_offset = x_offset_for_prev_y + dxdy;
                x_offset_for_prev_y = x_offset;
                y_offset_for_prev_x -= 1.0;
                y += y_inc;

                x2 = x_offset;
                y2 = 1.0;
            } else {
                x_offset = 0.0;
                y_offset = y_offset_for_prev_x + dydx;
                x_offset_for_prev_y -= 1.0;
                y_offset_for_prev_x = y_offset;
                x += x_inc;

                x2 = 1.0;
                y2 = y_offset;
            }

            let height = sign * (y2 - y1);
            let area = 0.5 * height * (area_offset + area_sign * (x1 + x2));

            self.add_delta(col, row, height, area);
        }

        let height = sign * (y_offset_end - y_offset);
        let area = 0.5 * height * (area_offset + area_sign * (x_offset + x_offset_end));

        self.add_delta(x, y, height, area);
    }

    #[inline(always)]
    fn add_delta(&mut self, x: isize, y: isize, height: f32, area: f32) {
        if y < 0 || y >= self.height as isize || x >= self.width as isize - 1 {
            return;
        }

        if x < 0 {
            let coverage_index = y as usize * self.width;
            self.coverage[coverage_index] += height;

            let bitmask_row = y as usize * self.bitmasks_width;
            self.bitmasks[bitmask_row] |= 1 << (BITMASK_SIZE - 1);

            return;
        }

        let coverage_index = y as usize * self.width + x as usize;
        self.coverage[coverage_index] += area;
        self.coverage[coverage_index + 1] += height - area;

        let bitmask_row = y as usize * self.bitmasks_width;

        let cell_x = x as usize >> CELL_SIZE_BITS;
        let cell_bit = 1 << (BITMASK_SIZE - 1 - (cell_x & (BITMASK_SIZE - 1)));
        self.bitmasks[bitmask_row + (cell_x >> BITMASK_SIZE_BITS)] |= cell_bit;

        let cell_x = (x + 1) as usize >> CELL_SIZE_BITS;
        let cell_bit = 1 << (BITMASK_SIZE - 1 - (cell_x & (BITMASK_SIZE - 1)));
        self.bitmasks[bitmask_row + (cell_x >> BITMASK_SIZE_BITS)] |= cell_bit;
    }

    pub fn finish(&mut self, color: Color, data: &mut [u32], width: usize) {
        struct Raster<'a, 'b> {
            rasterizer: &'a mut Rasterizer,
            color: Color,
            data: &'b mut [u32],
            width: usize,
        }

        impl<'a, 'b> Task for Raster<'a, 'b> {
            type Result = ();

            #[inline(always)]
            fn run<A: Arch>(self) {
                self.rasterizer
                    .finish_inner::<A>(self.color, self.data, self.width);
            }
        }

        static INNER: AtomicPtr<()> =
            unsafe { AtomicPtr::new(mem::transmute(dispatch as fn(Raster))) };

        fn dispatch(task: Raster) {
            // Currently CELL_SIZE is hard-coded to 4
            // let inner = if let Some(avx2) = Avx2::try_specialize::<Raster>() {
            //     avx2
            // } else {
            //     Sse2::specialize::<Raster>()
            // };

            let inner = Sse2::specialize::<Raster>();

            unsafe {
                INNER.store(mem::transmute(inner), Ordering::Relaxed);
            }

            inner(task)
        }

        let inner: fn(Raster) = unsafe { mem::transmute(INNER.load(Ordering::Relaxed)) };
        inner(Raster {
            rasterizer: self,
            color,
            data,
            width,
        })
    }

    #[inline(always)]
    fn finish_inner<A: Arch>(&mut self, color: Color, data: &mut [u32], width: usize) {
        let min_x = self.min_x.max(0) as usize;
        let min_y = self.min_y.max(0) as usize;
        let max_x = self.max_x.min(self.width as isize - 1) as usize;
        let max_y = self.max_y.min(self.height as isize - 1) as usize;

        let a = A::f32::from(color.a() as f32);
        let a_unit = a * A::f32::from(1.0 / 255.0);
        let r = a_unit * A::f32::from(color.r() as f32);
        let g = a_unit * A::f32::from(color.g() as f32);
        let b = a_unit * A::f32::from(color.b() as f32);

        for y in min_y..=max_y {
            let mut accum = 0.0;
            let mut coverage = 0.0;

            let mut x = min_x;

            let bitmask_start = min_x >> (BITMASK_SIZE_BITS + CELL_SIZE_BITS);
            let bitmask_end = max_x >> (BITMASK_SIZE_BITS + CELL_SIZE_BITS);
            for bitmask_x in bitmask_start..=bitmask_end {
                let bitmask_cell_x = bitmask_x << BITMASK_SIZE_BITS;
                let mut tile = self.bitmasks[y * self.bitmasks_width + bitmask_x];
                self.bitmasks[y * self.bitmasks_width + bitmask_x] = 0;

                while x <= max_x {
                    let index = tile.leading_zeros() as usize;
                    let next_x = ((bitmask_cell_x + index) << CELL_SIZE_BITS)
                        .min(((max_x + 1) >> CELL_SIZE_BITS) << CELL_SIZE_BITS);

                    if next_x > x {
                        if coverage > 254.5 / 255.0 && color.a() == 255 {
                            let start = y * width + x;
                            let end = y * width + next_x;
                            data[start..end].fill(color.into());
                        } else if coverage > 0.5 / 255.0 {
                            let start = y * width + x;
                            let end = y * width + next_x;
                            for pixels_slice in data[start..end].chunks_exact_mut(A::f32::LANES) {
                                let mask = A::f32::from(coverage);
                                let pixels = A::u32::load(pixels_slice);

                                let dst_a = A::f32::from((pixels >> 24) & A::u32::from(0xFF));
                                let dst_r = A::f32::from((pixels >> 16) & A::u32::from(0xFF));
                                let dst_g = A::f32::from((pixels >> 8) & A::u32::from(0xFF));
                                let dst_b = A::f32::from((pixels >> 0) & A::u32::from(0xFF));

                                let inv_a = A::f32::from(1.0) - mask * a_unit;
                                let out_a = A::u32::from(mask * a + inv_a * dst_a);
                                let out_r = A::u32::from(mask * r + inv_a * dst_r);
                                let out_g = A::u32::from(mask * g + inv_a * dst_g);
                                let out_b = A::u32::from(mask * b + inv_a * dst_b);

                                let out =
                                    (out_a << 24) | (out_r << 16) | (out_g << 8) | (out_b << 0);
                                out.store(pixels_slice);
                            }
                        }
                    }

                    if index == BITMASK_SIZE {
                        break;
                    }

                    x = next_x;

                    let coverage_start = y * self.width + x;
                    let coverage_end = coverage_start + CELL_SIZE;
                    let coverage_slice = &mut self.coverage[coverage_start..coverage_end];

                    let deltas = A::f32::load(coverage_slice);
                    let accums = A::f32::from(accum) + deltas.scan_sum();
                    accum = accums.last();
                    let mask = accums.abs().min(A::f32::from(1.0));
                    coverage = mask.last();

                    coverage_slice.fill(0.0);

                    let pixels_start = y * width + x;
                    let pixels_end = pixels_start + CELL_SIZE;
                    let pixels_slice = &mut data[pixels_start..pixels_end];
                    let pixels = A::u32::load(pixels_slice);

                    let dst_a = A::f32::from((pixels >> 24) & A::u32::from(0xFF));
                    let dst_r = A::f32::from((pixels >> 16) & A::u32::from(0xFF));
                    let dst_g = A::f32::from((pixels >> 8) & A::u32::from(0xFF));
                    let dst_b = A::f32::from((pixels >> 0) & A::u32::from(0xFF));

                    let inv_a = A::f32::from(1.0) - mask * a_unit;
                    let out_a = A::u32::from(mask * a + inv_a * dst_a);
                    let out_r = A::u32::from(mask * r + inv_a * dst_r);
                    let out_g = A::u32::from(mask * g + inv_a * dst_g);
                    let out_b = A::u32::from(mask * b + inv_a * dst_b);

                    let out = (out_a << 24) | (out_r << 16) | (out_g << 8) | (out_b << 0);
                    out.store(pixels_slice);

                    tile &= !(1 << (BITMASK_SIZE - 1 - index));
                    x += CELL_SIZE;
                }
            }
        }

        self.min_x = self.width as isize >> CELL_SIZE_BITS;
        self.min_y = self.height as isize;
        self.max_x = 0;
        self.max_y = 0;
    }
}
