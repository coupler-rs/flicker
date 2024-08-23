use crate::geom::{Affine, Point};
use crate::path::{Path, Verb};

const TOLERANCE: f32 = 0.2;
const MAX_SEGMENTS: usize = 100;

trait Curve {
    fn transform(&self, transform: Affine) -> Self;

    fn start(&self) -> Point;
    fn end(&self) -> Point;
    fn eval(&self, t: f32) -> Point;

    fn tangent(&self, t: f32) -> Point;
    fn start_tangent(&self) -> Point;
    fn end_tangent(&self) -> Point;

    fn segments_for_tolerance(&self, tolerance: f32) -> usize;
}

#[derive(Copy, Clone)]
struct Line {
    p0: Point,
    p1: Point,
}

impl Curve for Line {
    #[inline]
    fn transform(&self, transform: Affine) -> Self {
        Line {
            p0: transform * self.p0,
            p1: transform * self.p1,
        }
    }

    #[inline]
    fn start(&self) -> Point {
        self.p0
    }

    #[inline]
    fn end(&self) -> Point {
        self.p1
    }

    #[inline]
    fn eval(&self, t: f32) -> Point {
        Point::lerp(t, self.p0, self.p1)
    }

    #[inline]
    fn start_tangent(&self) -> Point {
        self.p1 - self.p0
    }

    #[inline]
    fn end_tangent(&self) -> Point {
        self.p1 - self.p0
    }

    #[inline]
    fn tangent(&self, _t: f32) -> Point {
        self.p1 - self.p0
    }

    #[inline]
    fn segments_for_tolerance(&self, _tolerance: f32) -> usize {
        1
    }
}

#[derive(Copy, Clone)]
struct Quadratic {
    p0: Point,
    p1: Point,
    p2: Point,
}

impl Curve for Quadratic {
    #[inline]
    fn transform(&self, transform: Affine) -> Self {
        Quadratic {
            p0: transform * self.p0,
            p1: transform * self.p1,
            p2: transform * self.p2,
        }
    }

    #[inline]
    fn start(&self) -> Point {
        self.p0
    }

    #[inline]
    fn end(&self) -> Point {
        self.p2
    }

    #[inline]
    fn eval(&self, t: f32) -> Point {
        let p01 = Point::lerp(t, self.p0, self.p1);
        let p12 = Point::lerp(t, self.p1, self.p2);
        Point::lerp(t, p01, p12)
    }

    #[inline]
    fn start_tangent(&self) -> Point {
        self.p1 - self.p0
    }

    #[inline]
    fn end_tangent(&self) -> Point {
        self.p2 - self.p1
    }

    #[inline]
    fn tangent(&self, t: f32) -> Point {
        Point::lerp(t, self.p1 - self.p0, self.p2 - self.p1)
    }

    #[inline]
    fn segments_for_tolerance(&self, tolerance: f32) -> usize {
        let dt = ((4.0 * tolerance) / (self.p0 - 2.0 * self.p1 + self.p2).length()).sqrt();

        dt.recip().ceil() as usize
    }
}

#[derive(Copy, Clone)]
struct Cubic {
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
}

impl Curve for Cubic {
    #[inline]
    fn transform(&self, transform: Affine) -> Self {
        Cubic {
            p0: transform * self.p0,
            p1: transform * self.p1,
            p2: transform * self.p2,
            p3: transform * self.p3,
        }
    }

    #[inline]
    fn start(&self) -> Point {
        self.p0
    }

    #[inline]
    fn end(&self) -> Point {
        self.p3
    }

    #[inline]
    fn eval(&self, t: f32) -> Point {
        let p01 = Point::lerp(t, self.p0, self.p1);
        let p12 = Point::lerp(t, self.p1, self.p2);
        let p23 = Point::lerp(t, self.p2, self.p3);
        let p012 = Point::lerp(t, p01, p12);
        let p123 = Point::lerp(t, p12, p23);
        Point::lerp(t, p012, p123)
    }

    #[inline]
    fn tangent(&self, t: f32) -> Point {
        let t1 = Point::lerp(t, self.p1 - self.p0, self.p2 - self.p1);
        let t2 = Point::lerp(t, self.p2 - self.p1, self.p3 - self.p2);
        Point::lerp(t, t1, t2)
    }

    #[inline]
    fn start_tangent(&self) -> Point {
        self.p1 - self.p0
    }

    #[inline]
    fn end_tangent(&self) -> Point {
        self.p3 - self.p2
    }

    #[inline]
    fn segments_for_tolerance(&self, tolerance: f32) -> usize {
        let a = -1.0 * self.p0 + 3.0 * self.p1 - 3.0 * self.p2 + self.p3;
        let b = 3.0 * (self.p0 - 2.0 * self.p1 + self.p2);
        let conc = b.length().max((a + b).length());
        let dt = ((8.0f32.sqrt() * tolerance) / conc).sqrt();

        dt.recip().ceil() as usize
    }
}

#[inline]
fn flatten_curve<C: Curve>(curve: &C, transform: Affine, sink: &mut impl FnMut(Point, Point)) {
    let curve = curve.transform(transform);

    let segments = curve.segments_for_tolerance(TOLERANCE).clamp(1, MAX_SEGMENTS);
    let dt = 1.0 / segments as f32;

    let mut prev = curve.start();
    let mut t = dt;
    for _ in 0..segments {
        let point = curve.eval(t);
        sink(prev, point);

        prev = point;
        t += dt;
    }
}

#[inline]
pub fn flatten(path: &Path, transform: Affine, sink: &mut impl FnMut(Point, Point)) {
    let mut points = path.points.iter();

    let mut first = Point::new(0.0, 0.0);
    let mut prev = Point::new(0.0, 0.0);
    for verb in &path.verbs {
        match *verb {
            Verb::Move => {
                first = *points.next().unwrap();
                prev = first;
            }
            Verb::Line => {
                let line = Line {
                    p0: prev,
                    p1: *points.next().unwrap(),
                };
                prev = line.end();

                flatten_curve(&line, transform, sink);
            }
            Verb::Quadratic => {
                let quadratic = Quadratic {
                    p0: prev,
                    p1: *points.next().unwrap(),
                    p2: *points.next().unwrap(),
                };
                prev = quadratic.end();

                flatten_curve(&quadratic, transform, sink);
            }
            Verb::Cubic => {
                let cubic = Cubic {
                    p0: prev,
                    p1: *points.next().unwrap(),
                    p2: *points.next().unwrap(),
                    p3: *points.next().unwrap(),
                };
                prev = cubic.end();

                flatten_curve(&cubic, transform, sink);
            }
            Verb::Close => {
                if prev != first {
                    flatten_curve(
                        &Line {
                            p0: prev,
                            p1: first,
                        },
                        transform,
                        sink,
                    );
                }
                prev = first;
            }
        }
    }

    if prev != first {
        flatten_curve(
            &Line {
                p0: prev,
                p1: first,
            },
            transform,
            sink,
        );
    }
}

struct Stroker<S> {
    width: f32,
    transform: Affine,
    first_right: Point,
    first_left: Point,
    prev_right: Point,
    prev_left: Point,
    closed: bool,
    sink: S,
}

impl<S: FnMut(Point, Point)> Stroker<S> {
    #[inline]
    fn new(width: f32, transform: Affine, sink: S) -> Stroker<S> {
        Stroker {
            width,
            transform,
            first_right: Point::new(0.0, 0.0),
            first_left: Point::new(0.0, 0.0),
            prev_right: Point::new(0.0, 0.0),
            prev_left: Point::new(0.0, 0.0),
            closed: true,
            sink,
        }
    }

    #[inline]
    fn cap_begin(&mut self) {
        (self.sink)(self.first_left, self.first_right);
    }

    #[inline]
    fn cap_end(&mut self) {
        (self.sink)(self.prev_right, self.prev_left);
    }

    #[inline]
    fn join(&mut self, right: Point, left: Point) {
        (self.sink)(self.prev_right, right);
        (self.sink)(left, self.prev_left);
    }

    #[inline]
    fn stroke_curve<C: Curve>(&mut self, curve: &C) {
        if curve.start() == curve.end() {
            return;
        }

        let curve_transformed = curve.transform(self.transform);

        let segments = curve_transformed.segments_for_tolerance(TOLERANCE).clamp(1, MAX_SEGMENTS);
        let dt = 1.0 / segments as f32;

        let start = curve_transformed.start();
        let start_tangent = curve.tangent(dt.min(0.5));
        let start_normal = Point::new(-start_tangent.y, start_tangent.x);
        let normal_len = start_normal.length();
        let offset = if normal_len.abs() < 1e-6 {
            Point::new(0.0, 0.0)
        } else {
            0.5 * self.width * (1.0 / normal_len) * start_normal
        };
        let offset_transformed = self.transform.linear() * offset;
        let right = start + offset_transformed;
        let left = start - offset_transformed;

        if self.closed {
            self.first_right = right;
            self.first_left = left;
            self.closed = false;
        } else {
            self.join(right, left);
        }

        self.prev_right = right;
        self.prev_left = left;

        let mut t = dt;
        for _ in 0..segments {
            let point = curve_transformed.eval(t);
            let tangent = curve.tangent(t.min(1.0 - dt));
            let normal = Point::new(-tangent.y, tangent.x);
            let normal_len = normal.length();
            let offset = if normal_len.abs() < 1e-6 {
                Point::new(0.0, 0.0)
            } else {
                0.5 * self.width * (1.0 / normal_len) * normal
            };
            let offset_transformed = self.transform.linear() * offset;
            let right = point + offset_transformed;
            let left = point - offset_transformed;

            (self.sink)(self.prev_right, right);
            (self.sink)(left, self.prev_left);

            self.prev_right = right;
            self.prev_left = left;
            t += dt;
        }

        self.closed = false;
    }

    #[inline]
    fn close(&mut self) {
        self.join(self.first_right, self.first_left);
        self.closed = true;
    }

    #[inline]
    fn finish(&mut self) {
        if !self.closed {
            self.cap_begin();
            self.cap_end();
            self.closed = true;
        }
    }
}

#[inline]
pub fn stroke(path: &Path, width: f32, transform: Affine, sink: &mut impl FnMut(Point, Point)) {
    let mut stroker = Stroker::new(width, transform, sink);

    let mut points = path.points.iter();
    let mut prev = Point::new(0.0, 0.0);
    for verb in &path.verbs {
        match *verb {
            Verb::Move => {
                stroker.finish();
                prev = *points.next().unwrap();
            }
            Verb::Line => {
                let p1 = *points.next().unwrap();
                stroker.stroke_curve(&Line { p0: prev, p1 });
                prev = p1;
            }
            Verb::Quadratic => {
                let p1 = *points.next().unwrap();
                let p2 = *points.next().unwrap();
                stroker.stroke_curve(&Quadratic { p0: prev, p1, p2 });
                prev = p2;
            }
            Verb::Cubic => {
                let p1 = *points.next().unwrap();
                let p2 = *points.next().unwrap();
                let p3 = *points.next().unwrap();
                stroker.stroke_curve(&Cubic {
                    p0: prev,
                    p1,
                    p2,
                    p3,
                });
                prev = p3;
            }
            Verb::Close => {
                stroker.close();
            }
        }
    }

    stroker.finish();
}
