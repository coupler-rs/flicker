use crate::color::Color;
use crate::flatten::{flatten, stroke};
use crate::geom::{Affine, Point};
use crate::path::Path;
use crate::raster::{Rasterizer, Segment};
use crate::text::{Font, Glyph, TextLayout};

const MAX_SEGMENTS: usize = 256;

pub struct Renderer {
    segments: Vec<Segment>,
    rasterizer: Rasterizer,
}

impl Renderer {
    pub fn new() -> Renderer {
        Renderer {
            segments: Vec::with_capacity(MAX_SEGMENTS),
            rasterizer: Rasterizer::new(),
        }
    }

    pub fn canvas<'a>(
        &'a mut self,
        data: &'a mut [u32],
        width: usize,
        height: usize,
    ) -> Canvas<'a> {
        assert!(data.len() == width * height);

        Canvas {
            renderer: self,
            data,
            width,
            height,
            transform: Affine::id(),
        }
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Canvas<'a> {
    renderer: &'a mut Renderer,
    data: &'a mut [u32],
    width: usize,
    height: usize,
    transform: Affine,
}

impl<'a> Canvas<'a> {
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn with_transform<F, R>(&mut self, transform: Affine, f: F) -> R
    where
        F: FnOnce(&mut Canvas) -> R,
    {
        let saved = self.transform;
        self.transform = saved * transform;

        let result = f(self);

        self.transform = saved;

        result
    }

    pub fn clear(&mut self, color: Color) {
        for pixel in self.data.iter_mut() {
            *pixel = color.into();
        }
    }

    fn add_segment(&mut self, p1: Point, p2: Point) {
        self.renderer.segments.push(Segment { p1, p2 });

        if self.renderer.segments.len() == self.renderer.segments.capacity() {
            self.drain_segments();
        }
    }

    fn drain_segments(&mut self) {
        self.renderer.rasterizer.add_segments(&self.renderer.segments);
        self.renderer.segments.clear();
    }

    pub fn fill_path(&mut self, path: &Path, transform: Affine, color: Color) {
        if path.is_empty() {
            return;
        }

        let transform = self.transform * transform;

        let mut min = Point::new(self.width as f32, self.height as f32);
        let mut max = Point::new(0.0, 0.0);
        for &point in &path.points {
            let transformed = transform * point;
            min = min.min(transformed);
            max = max.max(transformed);
        }

        let min_x = (min.x as isize).max(0).min(self.width as isize) as usize;
        let min_y = (min.y as isize).max(0).min(self.height as isize) as usize;
        let max_x = ((max.x + 1.0) as isize).max(0).min(self.width as isize) as usize;
        let max_y = ((max.y + 1.0) as isize).max(0).min(self.height as isize) as usize;

        if max_x <= min_x || max_y <= min_y {
            return;
        }

        let path_width = max_x - min_x;
        let path_height = max_y - min_y;

        let offset = Point::new(min_x as f32, min_y as f32);

        self.renderer.rasterizer.set_size(path_width, path_height);

        flatten(path, transform, &mut |p1, p2| {
            self.add_segment(p1 - offset, p2 - offset);
        });

        self.drain_segments();

        let data_start = min_y * self.width + min_x;
        self.renderer.rasterizer.finish(color, &mut self.data[data_start..], self.width);
    }

    pub fn stroke_path(&mut self, path: &Path, width: f32, transform: Affine, color: Color) {
        if path.is_empty() {
            return;
        }

        let transform = self.transform * transform;

        let dilate_x = transform.linear() * width * Point::new(0.5, 0.0);
        let dilate_y = transform.linear() * width * Point::new(0.0, 0.5);
        let dilate_min = dilate_x.min(dilate_y);
        let dilate_max = dilate_x.max(dilate_y);

        let mut min = Point::new(self.width as f32, self.height as f32);
        let mut max = Point::new(0.0, 0.0);
        for &point in &path.points {
            let transformed = transform * point;
            min = min.min(transformed + dilate_min);
            max = max.max(transformed + dilate_max);
        }

        let min_x = (min.x as isize).max(0).min(self.width as isize) as usize;
        let min_y = (min.y as isize).max(0).min(self.height as isize) as usize;
        let max_x = ((max.x + 1.0) as isize).max(0).min(self.width as isize) as usize;
        let max_y = ((max.y + 1.0) as isize).max(0).min(self.height as isize) as usize;

        if max_x <= min_x || max_y <= min_y {
            return;
        }

        let path_width = max_x - min_x;
        let path_height = max_y - min_y;

        let offset = Point::new(min_x as f32, min_y as f32);

        self.renderer.rasterizer.set_size(path_width, path_height);

        stroke(path, width, transform, &mut |p1, p2| {
            self.add_segment(p1 - offset, p2 - offset);
        });

        self.drain_segments();

        let data_start = min_y * self.width + min_x;
        self.renderer.rasterizer.finish(color, &mut self.data[data_start..], self.width);
    }

    pub fn fill_glyphs(
        &mut self,
        glyphs: &[Glyph],
        font: &Font,
        size: f32,
        transform: Affine,
        color: Color,
    ) {
        use rustybuzz::ttf_parser::{GlyphId, OutlineBuilder};

        struct Builder {
            path: Path,
            ascent: f32,
        }

        impl OutlineBuilder for Builder {
            fn move_to(&mut self, x: f32, y: f32) {
                self.path.move_to(Point::new(x, self.ascent - y));
            }

            fn line_to(&mut self, x: f32, y: f32) {
                self.path.line_to(Point::new(x, self.ascent - y));
            }

            fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
                self.path.quadratic_to(
                    Point::new(x1, self.ascent - y1),
                    Point::new(x, self.ascent - y),
                );
            }

            fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
                self.path.cubic_to(
                    Point::new(x1, self.ascent - y1),
                    Point::new(x2, self.ascent - y2),
                    Point::new(x, self.ascent - y),
                );
            }

            fn close(&mut self) {
                self.path.close();
            }
        }

        let scale = size / font.face.units_per_em() as f32;

        for glyph in glyphs {
            let mut builder = Builder {
                path: Path::new(),
                ascent: font.face.ascender() as f32,
            };
            font.face.outline_glyph(GlyphId(glyph.id), &mut builder);

            let transform = transform * Affine::translate(glyph.x, glyph.y) * Affine::scale(scale);

            self.fill_path(&builder.path, transform, color);
        }
    }

    pub fn fill_text(
        &mut self,
        text: &str,
        font: &Font,
        size: f32,
        transform: Affine,
        color: Color,
    ) {
        let layout = TextLayout::new(text, font, size);
        self.fill_glyphs(layout.glyphs(), font, size, transform, color);
    }
}
