use std::ops;

/// A 2-dimensional vector.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    /// Constructs a 2-dimensional vector.
    #[inline]
    pub fn new(x: f32, y: f32) -> Point {
        Point { x: x, y: y }
    }

    /// Computes the dot product between two vectors.
    #[inline]
    pub fn dot(self, other: Point) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Considering the two given vectors as 3-dimensional vectors lying in the
    /// XY-plane, finds the z-coordinate of their cross product.
    #[inline]
    pub fn cross(self, other: Point) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// Computes the distance between two points.
    #[inline]
    pub fn distance(self, other: Point) -> f32 {
        (other - self).length()
    }

    /// Computes the length of a vector.
    #[inline]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Finds the vector with the same direction and a length of 1.
    #[inline]
    pub fn normalized(self) -> Point {
        (1.0 / self.length()) * self
    }

    /// Linearly interpolates between two vectors by the parameter `t`.
    #[inline]
    pub fn lerp(t: f32, a: Point, b: Point) -> Point {
        (1.0 - t) * a + t * b
    }

    /// Finds the componentwise minimum of two vectors.
    #[inline]
    pub fn min(self, other: Point) -> Point {
        Point {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }

    /// Finds the componentwise maximum of two vectors.
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

/// A 2×2 matrix, in row-major order.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mat2x2(pub [f32; 4]);

impl Mat2x2 {
    /// Constructs a 2×2 matrix. Arguments are given in row-major order.
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Mat2x2 {
        Mat2x2([a, b, c, d])
    }

    /// Constructs an identity matrix.
    pub fn id() -> Mat2x2 {
        Mat2x2([1.0, 0.0, 0.0, 1.0])
    }

    /// Constructs a uniform scaling matrix.
    pub fn scale(scale: f32) -> Mat2x2 {
        Mat2x2([scale, 0.0, 0.0, scale])
    }

    /// Constructs a rotation matrix.
    pub fn rotate(angle: f32) -> Mat2x2 {
        Mat2x2([angle.cos(), angle.sin(), -angle.sin(), angle.cos()])
    }

    /// Computes the determinant of the matrix.
    pub fn determinant(&self) -> f32 {
        self.0[0] * self.0[3] - self.0[1] * self.0[2]
    }
}

impl ops::Mul<Mat2x2> for Mat2x2 {
    type Output = Mat2x2;
    #[inline]
    fn mul(self, rhs: Mat2x2) -> Mat2x2 {
        Mat2x2([
            self.0[0] * rhs.0[0] + self.0[1] * rhs.0[2],
            self.0[0] * rhs.0[1] + self.0[1] * rhs.0[3],
            self.0[2] * rhs.0[0] + self.0[3] * rhs.0[2],
            self.0[2] * rhs.0[1] + self.0[3] * rhs.0[3],
        ])
    }
}

impl ops::Mul<Point> for Mat2x2 {
    type Output = Point;
    #[inline]
    fn mul(self, rhs: Point) -> Point {
        Point {
            x: self.0[0] * rhs.x + self.0[1] * rhs.y,
            y: self.0[2] * rhs.x + self.0[3] * rhs.y,
        }
    }
}

impl ops::Mul<Mat2x2> for f32 {
    type Output = Mat2x2;
    #[inline]
    fn mul(self, rhs: Mat2x2) -> Mat2x2 {
        Mat2x2([
            self * rhs.0[0],
            self * rhs.0[1],
            self * rhs.0[2],
            self * rhs.0[3],
        ])
    }
}

impl ops::Mul<f32> for Mat2x2 {
    type Output = Mat2x2;
    #[inline]
    fn mul(self, rhs: f32) -> Mat2x2 {
        rhs * self
    }
}

/// A 2-dimensional affine transformation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Affine {
    pub matrix: Mat2x2,
    pub offset: Point,
}

impl Affine {
    /// Constructs an affine transformation from the given transformation
    /// matrix and translation vector.
    pub fn new(matrix: Mat2x2, offset: Point) -> Affine {
        Affine { matrix, offset }
    }

    /// Constructs an identity transformation.
    pub fn id() -> Affine {
        Affine {
            matrix: Mat2x2::id(),
            offset: Point::new(0.0, 0.0),
        }
    }

    /// Constructs a translation.
    pub fn translate(x: f32, y: f32) -> Affine {
        Affine {
            matrix: Mat2x2::id(),
            offset: Point::new(x, y),
        }
    }

    /// Constructs a uniform scaling.
    pub fn scale(scale: f32) -> Affine {
        Affine {
            matrix: Mat2x2::scale(scale),
            offset: Point::new(0.0, 0.0),
        }
    }

    /// Constructs a rotation.
    pub fn rotate(angle: f32) -> Affine {
        Affine {
            matrix: Mat2x2::rotate(angle),
            offset: Point::new(0.0, 0.0),
        }
    }

    /// Sequentially composes two affine transformations.
    pub fn then(self, other: Affine) -> Affine {
        Affine {
            matrix: other.matrix * self.matrix,
            offset: other.matrix * self.offset + other.offset,
        }
    }

    /// Applies the affine transformation to the given vector.
    pub fn apply(self, vec: Point) -> Point {
        self.matrix * vec + self.offset
    }
}
