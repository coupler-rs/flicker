use std::iter::zip;

use rustybuzz::{Face, UnicodeBuffer};

pub struct Font {
    pub(crate) face: Face<'static>,
}

impl Font {
    pub fn from_bytes(data: &'static [u8], index: usize) -> Option<Font> {
        let face = Face::from_slice(data, index as u32)?;

        Some(Self { face })
    }
}

#[derive(Copy, Clone)]
pub struct Glyph {
    pub id: u16,
    pub x: f32,
    pub y: f32,
}

#[derive(Clone)]
pub struct TextLayout {
    glyphs: Vec<Glyph>,
}

impl TextLayout {
    pub fn empty() -> TextLayout {
        TextLayout { glyphs: Vec::new() }
    }

    pub fn new(text: &str, font: &Font, size: f32) -> TextLayout {
        let mut buf = UnicodeBuffer::new();
        buf.push_str(text);
        let glyph_buf = rustybuzz::shape(&font.face, &[], buf);

        let scale = size / font.face.units_per_em() as f32;

        let mut offset = 0.0;
        let mut glyphs = Vec::with_capacity(glyph_buf.len());
        for (info, glyph_pos) in zip(glyph_buf.glyph_infos(), glyph_buf.glyph_positions()) {
            glyphs.push(Glyph {
                id: info.glyph_id as u16,
                x: offset + scale * glyph_pos.x_offset as f32,
                y: scale * glyph_pos.y_offset as f32,
            });

            offset += scale * glyph_pos.x_advance as f32;
        }

        TextLayout { glyphs }
    }

    pub fn glyphs(&self) -> &[Glyph] {
        &self.glyphs
    }
}
