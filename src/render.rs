use crate::bbox::{self, Bbox};
use crate::color::Color;
use crate::flatten::{self, Line};
use crate::geom::{Affine, Point};
use crate::path::Path;
use crate::raster::Rasterizer;
use crate::text::{Font, Glyph, TextLayout};

pub struct Renderer {
    lines: Vec<Line>,
    rasterizer: Rasterizer,
}

impl Renderer {
    pub fn new() -> Renderer {
        Renderer {
            lines: Vec::new(),
            rasterizer: Rasterizer::new(),
        }
    }

    pub fn attach<'a>(
        &'a mut self,
        data: &'a mut [u32],
        width: usize,
        height: usize,
    ) -> RenderTarget<'a> {
        assert!(data.len() == width * height);

        RenderTarget {
            data,
            width,
            height,
            transform: Affine::id(),

            lines: &mut self.lines,
            rasterizer: &mut self.rasterizer,
        }
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RenderTarget<'a> {
    data: &'a mut [u32],
    width: usize,
    height: usize,
    transform: Affine,

    lines: &'a mut Vec<Line>,
    rasterizer: &'a mut Rasterizer,
}

impl<'a> RenderTarget<'a> {
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn with_transform<F, R>(&mut self, transform: Affine, f: F) -> R
    where
        F: FnOnce(&mut RenderTarget) -> R,
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

    pub fn fill_path(&mut self, path: &Path, transform: Affine, color: Color) {
        let transform = self.transform * transform;

        let clip = Bbox {
            x0: 0,
            y0: 0,
            x1: self.width as i32,
            y1: self.height as i32,
        };
        let bbox = bbox::fill(path, transform, clip);

        if bbox.is_empty() {
            return;
        }

        let path_width = (bbox.x1 - bbox.x0) as usize;
        let path_height = (bbox.y1 - bbox.y0) as usize;
        self.rasterizer.set_size(path_width, path_height);

        let offset = Affine::translate(-bbox.x0 as f32, -bbox.y0 as f32);
        flatten::fill(path, offset * transform, &mut self.lines);

        self.rasterizer.rasterize(&self.lines);
        self.lines.clear();

        let data_start = bbox.y0 as usize * self.width + bbox.x0 as usize;
        self.rasterizer.finish(color, &mut self.data[data_start..], self.width);
    }

    pub fn stroke_path(&mut self, path: &Path, width: f32, transform: Affine, color: Color) {
        let transform = self.transform * transform;

        let clip = Bbox {
            x0: 0,
            y0: 0,
            x1: self.width as i32,
            y1: self.height as i32,
        };
        let bbox = bbox::stroke(path, width, transform, clip);

        if bbox.is_empty() {
            return;
        }

        let path_width = (bbox.x1 - bbox.x0) as usize;
        let path_height = (bbox.y1 - bbox.y0) as usize;
        self.rasterizer.set_size(path_width, path_height);

        let offset = Affine::translate(-bbox.x0 as f32, -bbox.y0 as f32);
        flatten::stroke(path, width, offset * transform, &mut self.lines);

        self.rasterizer.rasterize(&self.lines);
        self.lines.clear();

        let data_start = bbox.y0 as usize * self.width + bbox.x0 as usize;
        self.rasterizer.finish(color, &mut self.data[data_start..], self.width);
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
