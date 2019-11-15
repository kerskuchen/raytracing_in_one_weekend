use rand::prelude::*;
use std::fmt::Write;
use std::fs;

struct Camera {
    origin: Vec3,
    left_bottom_corner: Vec3,
    dim_horizontal: Vec3,
    dim_vertical: Vec3,
}

impl Camera {
    pub const fn new(origin: Vec3) -> Camera {
        Camera {
            origin,
            left_bottom_corner: Vec3::new(-2.0, -1.0, -1.0),
            dim_horizontal: Vec3::new(4.0, 0.0, 0.0),
            dim_vertical: Vec3::new(0.0, 2.0, 0.0),
        }
    }

    fn get_ray(&self, u: f32, v: f32) -> Ray {
        Ray::new(
            self.origin,
            self.left_bottom_corner + u * self.dim_horizontal + v * self.dim_vertical,
        )
    }
}

struct HitRecord {
    t: f32,
    position: Vec3,
    normal: Vec3,
}

struct Sphere {
    center: Vec3,
    radius: f32,
}

fn ray_hit_sphere(sphere: &Sphere, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
    let dir_center_to_origin = ray.origin - sphere.center;
    let a = Vec3::dot(ray.direction, ray.direction);
    let b = Vec3::dot(dir_center_to_origin, ray.direction);
    let c = Vec3::dot(dir_center_to_origin, dir_center_to_origin) - sphere.radius * sphere.radius;
    let discriminant = b * b - a * c;

    if discriminant < 0.0 {
        return None;
    }

    let t_hit = (-b - f32::sqrt(discriminant)) / a;
    if t_min < t_hit && t_hit < t_max {
        let t = t_hit;
        let position = ray.point_at_parameter(t_hit);
        let normal = (position - sphere.center) / sphere.radius;
        return Some(HitRecord {
            t,
            position,
            normal,
        });
    }

    let t_hit = (-b + f32::sqrt(discriminant)) / a;
    if t_min < t_hit && t_hit < t_max {
        let t = t_hit;
        let position = ray.point_at_parameter(t_hit);
        let normal = (position - sphere.center) / sphere.radius;
        return Some(HitRecord {
            t,
            position,
            normal,
        });
    }

    return None;
}

#[derive(Copy, Clone)]
enum Material {
    Lambertian { albedo: Color },
    Metal { albedo: Color, fuzz: f32 },
}

fn ray_material_scatter(
    ray: &Ray,
    material: &Material,
    hit: &HitRecord,
    out_attenuation: &mut Color,
    out_scattered: &mut Ray,
) -> bool {
    match material {
        Material::Lambertian { albedo } => {
            let target = hit.position + hit.normal + Vec3::random_point_in_unit_sphere();
            *out_scattered = Ray::new(hit.position, target - hit.position);
            *out_attenuation = *albedo;
            true
        }
        Material::Metal { albedo, fuzz } => {
            let fuzz = f32::max(0.0, f32::min(1.0, *fuzz));
            let reflected = ray.direction.normalized().reflected_on_normal(hit.normal);
            *out_scattered = Ray::new(
                hit.position,
                reflected + fuzz * Vec3::random_point_in_unit_sphere(),
            );
            *out_attenuation = *albedo;

            Vec3::dot(out_scattered.direction, hit.normal) > 0.0
        }
    }
}

enum Hittable {
    Sphere(Sphere, Material),
}

fn ray_hit_color(ray: &Ray, hittables: &[Hittable], depth: u32) -> Color {
    if depth >= 50 {
        Color::black();
    }

    let t_min = 0.001;
    let mut t_max = std::f32::MAX;
    let mut current_hit = None;
    let mut current_material = Material::Lambertian {
        albedo: Color::black(),
    };

    for hittable in hittables {
        let (maybe_hit, material) = match hittable {
            Hittable::Sphere(sphere, material) => {
                (ray_hit_sphere(&sphere, ray, t_min, t_max), material)
            }
        };

        match maybe_hit {
            Some(new_hit) => {
                t_max = new_hit.t;
                current_hit.replace(new_hit);
                current_material = *material;
            }
            None => {}
        }
    }

    match current_hit {
        Some(hit) => {
            let mut scattered = Ray::new(Vec3::zero(), Vec3::zero());
            let mut attenuation = Color::black();

            if ray_material_scatter(
                ray,
                &current_material,
                &hit,
                &mut attenuation,
                &mut scattered,
            ) {
                return attenuation * ray_hit_color(&scattered, hittables, depth + 1);
            } else {
                return Color::black();
            }
        }
        None => {
            let ray_unit_direction = ray.direction.normalized();
            let t = 0.5 * (ray_unit_direction.y + 1.0);
            Color::from_mix(Color::white(), Color::new(0.5, 0.7, 1.0), t)
        }
    }
}

fn main() {
    let image_width = 200;
    let image_height = 100;
    let mut ppm_data = format!("P3\n{} {}\n255\n", image_width, image_height);

    let samplecount = 1000;
    let camera = Camera::new(Vec3::new(0.0, 0.0, 0.0));

    let hittables = vec![
        Hittable::Sphere(
            Sphere {
                center: Vec3::new(0.0, 0.0, -1.0),
                radius: 0.5,
            },
            Material::Lambertian {
                albedo: Color::new(0.8, 0.3, 0.3),
            },
        ),
        Hittable::Sphere(
            Sphere {
                center: Vec3::new(0.0, -100.5, -1.0),
                radius: 100.0,
            },
            Material::Lambertian {
                albedo: Color::new(0.8, 0.8, 0.0),
            },
        ),
        Hittable::Sphere(
            Sphere {
                center: Vec3::new(1.0, 0.0, -1.0),
                radius: 0.5,
            },
            Material::Metal {
                albedo: Color::new(0.8, 0.6, 0.2),
                fuzz: 1.0,
            },
        ),
        Hittable::Sphere(
            Sphere {
                center: Vec3::new(-1.0, 0.0, -1.0),
                radius: 0.5,
            },
            Material::Metal {
                albedo: Color::new(0.8, 0.8, 0.8),
                fuzz: 0.3,
            },
        ),
    ];

    for y in (0..image_height).rev() {
        for x in 0..image_width {
            let mut color = Color::black();
            for _ in 0..samplecount {
                let u = (x as f32 + random::<f32>()) / image_width as f32;
                let v = (y as f32 + random::<f32>()) / image_height as f32;

                let ray = camera.get_ray(u, v);
                color += ray_hit_color(&ray, &hittables, 0);
            }
            color /= samplecount as f32;
            // Gamma correction
            color = Color::new(f32::sqrt(color.r), f32::sqrt(color.g), f32::sqrt(color.b));

            let ir = f32::round(255.0 * color.r) as i32;
            let ig = f32::round(255.0 * color.g) as i32;
            let ib = f32::round(255.0 * color.b) as i32;

            write!(&mut ppm_data, "{} {} {}\n", ir, ig, ib).unwrap();
        }
    }

    fs::write("output.ppm", ppm_data).expect("Unable to write file");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Vector 3

use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::DivAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Default for Vec3 {
    fn default() -> Vec3 {
        Vec3::zero()
    }
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    pub const fn zero() -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub const fn ones() -> Vec3 {
        Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        }
    }

    pub const fn unit_x() -> Vec3 {
        Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub const fn unit_y() -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    }

    pub const fn unit_z() -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    }

    pub fn length(self) -> f32 {
        f32::sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    }

    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalized(self) -> Vec3 {
        self / self.length()
    }

    pub fn normalize(&mut self) {
        *self = *self / self.length();
    }

    pub fn dot(a: Vec3, b: Vec3) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
        Vec3::new(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        )
    }

    pub fn reflected_on_normal(self, normal: Vec3) -> Vec3 {
        self - 2.0 * Vec3::dot(self, normal) * normal
    }

    pub fn random_point_in_unit_rect() -> Vec3 {
        Vec3::new(random(), random(), random())
    }

    pub fn random_point_in_unit_sphere() -> Vec3 {
        let mut result = Vec3::random_point_in_unit_rect();
        while result.length_squared() >= 1.0 {
            result = Vec3::random_point_in_unit_rect();
        }
        result
    }
}

// ---------------------------------------------------------------------------------------------
// Negation
//

impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Element-wise addition
//
impl Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}
impl Add<f32> for Vec3 {
    type Output = Vec3;

    fn add(self, scalar: f32) -> Vec3 {
        Vec3 {
            x: self.x + scalar,
            y: self.y + scalar,
            z: self.z + scalar,
        }
    }
}
impl Add<Vec3> for f32 {
    type Output = Vec3;

    fn add(self, vec: Vec3) -> Vec3 {
        Vec3 {
            x: vec.x + self,
            y: vec.y + self,
            z: vec.z + self,
        }
    }
}

impl AddAssign<Vec3> for Vec3 {
    fn add_assign(&mut self, other: Vec3) {
        *self = Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl AddAssign<f32> for Vec3 {
    fn add_assign(&mut self, scalar: f32) {
        *self = Vec3 {
            x: self.x + scalar,
            y: self.y + scalar,
            z: self.z + scalar,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Element-wise subtraction
//
impl Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}
impl Sub<f32> for Vec3 {
    type Output = Vec3;

    fn sub(self, scalar: f32) -> Vec3 {
        Vec3 {
            x: self.x - scalar,
            y: self.y - scalar,
            z: self.z - scalar,
        }
    }
}

impl SubAssign<Vec3> for Vec3 {
    fn sub_assign(&mut self, other: Vec3) {
        *self = Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}
impl SubAssign<f32> for Vec3 {
    fn sub_assign(&mut self, scalar: f32) {
        *self = Vec3 {
            x: self.x - scalar,
            y: self.y - scalar,
            z: self.z - scalar,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Element-wise multiplication
//
impl Mul<Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}
impl Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(self, scalar: f32) -> Vec3 {
        Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}
impl Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, vec: Vec3) -> Vec3 {
        Vec3 {
            x: vec.x * self,
            y: vec.y * self,
            z: vec.z * self,
        }
    }
}
impl MulAssign<Vec3> for Vec3 {
    fn mul_assign(&mut self, other: Vec3) {
        *self = Vec3 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}
impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, scalar: f32) {
        *self = Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Element-wise division
//
impl Div<Vec3> for Vec3 {
    type Output = Vec3;

    fn div(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
        }
    }
}
impl Div<f32> for Vec3 {
    type Output = Vec3;

    fn div(self, scalar: f32) -> Vec3 {
        Vec3 {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

impl DivAssign<Vec3> for Vec3 {
    fn div_assign(&mut self, other: Vec3) {
        *self = Vec3 {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
        }
    }
}
impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, scalar: f32) {
        *self = Vec3 {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Color

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl From<Vec3> for Color {
    fn from(v: Vec3) -> Self {
        Color {
            r: v.x,
            g: v.y,
            b: v.z,
        }
    }
}

impl Default for Color {
    fn default() -> Color {
        Color::black()
    }
}

impl Color {
    pub const fn new(r: f32, g: f32, b: f32) -> Color {
        Color { r, g, b }
    }

    pub fn from_mix(first: Color, second: Color, percentage: f32) -> Color {
        (1.0 - percentage) * first + percentage * second
    }

    pub const fn black() -> Color {
        Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
        }
    }

    pub const fn white() -> Color {
        Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
        }
    }

    pub const fn red() -> Color {
        Color {
            r: 1.0,
            g: 0.0,
            b: 0.0,
        }
    }

    pub const fn green() -> Color {
        Color {
            r: 0.0,
            g: 1.0,
            b: 0.0,
        }
    }

    pub const fn blue() -> Color {
        Color {
            r: 0.0,
            g: 0.0,
            b: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Negation
//

impl Neg for Color {
    type Output = Color;

    fn neg(self) -> Color {
        Color {
            r: -self.r,
            g: -self.g,
            b: -self.b,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Element-wise addition
//
impl Add<Color> for Color {
    type Output = Color;

    fn add(self, other: Color) -> Color {
        Color {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }
}
impl Add<f32> for Color {
    type Output = Color;

    fn add(self, scalar: f32) -> Color {
        Color {
            r: self.r + scalar,
            g: self.g + scalar,
            b: self.b + scalar,
        }
    }
}
impl Add<Color> for f32 {
    type Output = Color;

    fn add(self, vec: Color) -> Color {
        Color {
            r: vec.r + self,
            g: vec.g + self,
            b: vec.b + self,
        }
    }
}

impl AddAssign<Color> for Color {
    fn add_assign(&mut self, other: Color) {
        *self = Color {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }
}

impl AddAssign<f32> for Color {
    fn add_assign(&mut self, scalar: f32) {
        *self = Color {
            r: self.r + scalar,
            g: self.g + scalar,
            b: self.b + scalar,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Element-wise subtraction
//
impl Sub<Color> for Color {
    type Output = Color;

    fn sub(self, other: Color) -> Color {
        Color {
            r: self.r - other.r,
            g: self.g - other.g,
            b: self.b - other.b,
        }
    }
}
impl Sub<f32> for Color {
    type Output = Color;

    fn sub(self, scalar: f32) -> Color {
        Color {
            r: self.r - scalar,
            g: self.g - scalar,
            b: self.b - scalar,
        }
    }
}

impl SubAssign<Color> for Color {
    fn sub_assign(&mut self, other: Color) {
        *self = Color {
            r: self.r - other.r,
            g: self.g - other.g,
            b: self.b - other.b,
        }
    }
}
impl SubAssign<f32> for Color {
    fn sub_assign(&mut self, scalar: f32) {
        *self = Color {
            r: self.r - scalar,
            g: self.g - scalar,
            b: self.b - scalar,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Element-wise multiplication
//
impl Mul<Color> for Color {
    type Output = Color;

    fn mul(self, other: Color) -> Color {
        Color {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
        }
    }
}
impl Mul<f32> for Color {
    type Output = Color;

    fn mul(self, scalar: f32) -> Color {
        Color {
            r: self.r * scalar,
            g: self.g * scalar,
            b: self.b * scalar,
        }
    }
}
impl Mul<Color> for f32 {
    type Output = Color;

    fn mul(self, vec: Color) -> Color {
        Color {
            r: vec.r * self,
            g: vec.g * self,
            b: vec.b * self,
        }
    }
}
impl MulAssign<Color> for Color {
    fn mul_assign(&mut self, other: Color) {
        *self = Color {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
        }
    }
}
impl MulAssign<f32> for Color {
    fn mul_assign(&mut self, scalar: f32) {
        *self = Color {
            r: self.r * scalar,
            g: self.g * scalar,
            b: self.b * scalar,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Element-wise division
//
impl Div<Color> for Color {
    type Output = Color;

    fn div(self, other: Color) -> Color {
        Color {
            r: self.r / other.r,
            g: self.g / other.g,
            b: self.b / other.b,
        }
    }
}
impl Div<f32> for Color {
    type Output = Color;

    fn div(self, scalar: f32) -> Color {
        Color {
            r: self.r / scalar,
            g: self.g / scalar,
            b: self.b / scalar,
        }
    }
}
impl DivAssign<Color> for Color {
    fn div_assign(&mut self, other: Color) {
        *self = Color {
            r: self.r / other.r,
            g: self.g / other.g,
            b: self.b / other.b,
        }
    }
}
impl DivAssign<f32> for Color {
    fn div_assign(&mut self, scalar: f32) {
        *self = Color {
            r: self.r / scalar,
            g: self.g / scalar,
            b: self.b / scalar,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Ray

pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub const fn new(origin: Vec3, direction: Vec3) -> Ray {
        Ray { origin, direction }
    }

    pub fn point_at_parameter(&self, t: f32) -> Vec3 {
        self.origin + t * self.direction
    }
}
