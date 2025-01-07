use crate::geom::{Affine, Point};
use crate::path::Path;

#[derive(Copy, Clone)]
pub struct Bbox {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Bbox {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }
}

pub fn fill(path: &Path, transform: Affine, clip: Bbox) -> Bbox {
    let mut min = Point::new(clip.x1 as f32, clip.y1 as f32);
    let mut max = Point::new(clip.x0 as f32, clip.y0 as f32);
    for &point in &path.points {
        let transformed = transform * point;
        min = min.min(transformed);
        max = max.max(transformed);
    }

    Bbox {
        x0: (min.x as i32).max(clip.x0).min(clip.x1),
        y0: (min.y as i32).max(clip.y0).min(clip.y1),
        x1: ((max.x + 1.0) as i32).max(clip.x0).min(clip.x1),
        y1: ((max.y + 1.0) as i32).max(clip.y0).min(clip.y1),
    }
}

pub fn stroke(path: &Path, width: f32, transform: Affine, clip: Bbox) -> Bbox {
    let dilate_x = transform.linear() * width * Point::new(0.5, 0.0);
    let dilate_y = transform.linear() * width * Point::new(0.0, 0.5);
    let dilate_min = dilate_x.min(dilate_y).min(-dilate_x).min(-dilate_y);
    let dilate_max = dilate_x.max(dilate_y).max(-dilate_x).max(-dilate_y);

    let mut min = Point::new(clip.x1 as f32, clip.y1 as f32);
    let mut max = Point::new(clip.x0 as f32, clip.y0 as f32);
    for &point in &path.points {
        let transformed = transform * point;
        min = min.min(transformed + dilate_min);
        max = max.max(transformed + dilate_max);
    }

    Bbox {
        x0: (min.x as i32).max(clip.x0).min(clip.x1),
        y0: (min.y as i32).max(clip.y0).min(clip.y1),
        x1: ((max.x + 1.0) as i32).max(clip.x0).min(clip.x1),
        y1: ((max.y + 1.0) as i32).max(clip.y0).min(clip.y1),
    }
}
