use std::ops;

/// A 2-dimensional point.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    /// Constructs a 2-dimensional point.
    #[inline]
    pub fn new(x: f32, y: f32) -> Point {
        Point { x: x, y: y }
    }

    /// Computes the dot product between two points (treated as vectors).
    #[inline]
    pub fn dot(self, other: Point) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Considering the two given points as 3-dimensional vectors lying in the XY-plane, finds the
    /// z-coordinate of their cross product.
    #[inline]
    pub fn cross(self, other: Point) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// Computes the distance between two points.
    #[inline]
    pub fn distance(self, other: Point) -> f32 {
        (other - self).length()
    }

    /// Computes the distance of a point from the origin.
    #[inline]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Finds the vector with the same direction and a length of 1.
    #[inline]
    pub fn normalized(self) -> Point {
        (1.0 / self.length()) * self
    }

    /// Linearly interpolates between two points by the parameter `t`.
    #[inline]
    pub fn lerp(t: f32, a: Point, b: Point) -> Point {
        (1.0 - t) * a + t * b
    }

    /// Finds the componentwise minimum of two points.
    #[inline]
    pub fn min(self, other: Point) -> Point {
        Point {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }

    /// Finds the componentwise maximum of two points.
    #[inline]
    pub fn max(self, other: Point) -> Point {
        Point {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
        }
    }
}

impl ops::Add for Point {
    type Output = Point;

    #[inline]
    fn add(self, rhs: Point) -> Point {
        Point {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl ops::AddAssign for Point {
    #[inline]
    fn add_assign(&mut self, other: Point) {
        *self = *self + other;
    }
}

impl ops::Sub for Point {
    type Output = Point;

    #[inline]
    fn sub(self, rhs: Point) -> Point {
        Point {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl ops::SubAssign for Point {
    #[inline]
    fn sub_assign(&mut self, other: Point) {
        *self = *self - other;
    }
}

impl ops::Mul<Point> for f32 {
    type Output = Point;

    #[inline]
    fn mul(self, rhs: Point) -> Point {
        Point {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

impl ops::Mul<f32> for Point {
    type Output = Point;

    #[inline]
    fn mul(self, rhs: f32) -> Point {
        rhs * self
    }
}

impl ops::MulAssign<f32> for Point {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        *self = rhs * *self;
    }
}

/// A 2-dimensional affine transformation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Affine([f32; 6]);

impl Affine {
    /// Constructs an affine transformation from a set of coefficients.
    ///
    /// The coefficients are interpreted as the first two rows of a 3×3 affine transformation matrix
    /// in row-major order.
    #[inline]
    pub fn new(coeffs: [f32; 6]) -> Affine {
        Affine(coeffs)
    }

    /// Gets the coefficients of the transformation.
    ///
    /// The coefficients are the first two rows of the corresponding 3×3 affine transformation
    /// matrix in row-major order.
    #[inline]
    pub fn coeffs(self) -> [f32; 6] {
        self.0
    }

    /// Constructs an identity transformation.
    #[inline]
    pub fn id() -> Affine {
        Affine([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    }

    /// Constructs a translation.
    #[inline]
    pub fn translate(x: f32, y: f32) -> Affine {
        Affine([1.0, 0.0, x, 0.0, 1.0, y])
    }

    /// Constructs a uniform scaling.
    #[inline]
    pub fn scale(scale: f32) -> Affine {
        Affine([scale, 0.0, 0.0, 0.0, scale, 0.0])
    }

    /// Constructs a rotation.
    #[inline]
    pub fn rotate(angle: f32) -> Affine {
        let cos = angle.cos();
        let sin = angle.sin();

        Affine([cos, sin, 0.0, -sin, cos, 0.0])
    }

    // Gets the linear part of the affine transformation, i.e. without the translation.
    #[inline]
    pub fn linear(&self) -> Affine {
        Affine([self.0[0], self.0[1], 0.0, self.0[3], self.0[4], 0.0])
    }
}

impl ops::Mul<Affine> for Affine {
    type Output = Affine;

    #[inline]
    fn mul(self, rhs: Affine) -> Affine {
        Affine([
            self.0[0] * rhs.0[0] + self.0[1] * rhs.0[3],
            self.0[0] * rhs.0[1] + self.0[1] * rhs.0[4],
            self.0[0] * rhs.0[2] + self.0[1] * rhs.0[5] + self.0[2],
            self.0[3] * rhs.0[0] + self.0[4] * rhs.0[3],
            self.0[3] * rhs.0[1] + self.0[4] * rhs.0[4],
            self.0[3] * rhs.0[2] + self.0[4] * rhs.0[5] + self.0[5],
        ])
    }
}

impl ops::MulAssign<Affine> for Affine {
    #[inline]
    fn mul_assign(&mut self, rhs: Affine) {
        *self = *self * rhs;
    }
}

impl ops::Mul<Point> for Affine {
    type Output = Point;

    #[inline]
    fn mul(self, rhs: Point) -> Point {
        Point {
            x: self.0[0] * rhs.x + self.0[1] * rhs.y + self.0[2],
            y: self.0[3] * rhs.x + self.0[4] * rhs.y + self.0[5],
        }
    }
}

impl ops::Mul<Affine> for Point {
    type Output = Point;

    #[inline]
    fn mul(self, rhs: Affine) -> Point {
        rhs * self
    }
}

impl ops::MulAssign<Affine> for Point {
    #[inline]
    fn mul_assign(&mut self, rhs: Affine) {
        *self = rhs * *self;
    }
}

impl ops::Mul<Affine> for f32 {
    type Output = Affine;
    #[inline]
    fn mul(self, rhs: Affine) -> Affine {
        Affine([
            self * rhs.0[0],
            self * rhs.0[1],
            self * rhs.0[2],
            self * rhs.0[3],
            self * rhs.0[4],
            self * rhs.0[5],
        ])
    }
}

impl ops::Mul<f32> for Affine {
    type Output = Affine;
    #[inline]
    fn mul(self, rhs: f32) -> Affine {
        rhs * self
    }
}

impl ops::MulAssign<f32> for Affine {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        *self = rhs * *self;
    }
}
