#pragma once

#include <cuda_runtime.h>
#include "linear_math.h"

#define PiOver4 0.78539816339
#define PiOver2 1.57079632679

using Point2f = Vec2f;
using Vector2f = Vec2f;
using Vector3f = Vec3f;
using Float = float;

__device__ inline Vec2f& operator*(float a, Vec2f& v){ v.x *= a; v.y *= a;           return v; }
__device__ inline Vec3f& operator*(float a, Vec3f& v){ v.x *= a; v.y *= a; v.z *= a; return v; }

__device__ inline Point2f ConcentricSampleDisk(Point2f &u) {
    // Map uniform random numbers to $[-1,1]^2$
    Point2f uOffset = 2.f * u - Vector2f(1, 1);

    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0) return Point2f(0, 0);

    // Apply concentric mapping to point
    Float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }
    return r * Point2f(cos(theta), sin(theta));
}

__device__ inline  Vector3f CosineSampleHemisphere(Point2f &u) {
    Point2f d = ConcentricSampleDisk(u);
    Float z = sqrt(max((Float)0, 1 - d.x * d.x - d.y * d.y));
    return Vector3f(d.x, d.y, z);
}

__device__ inline void lambertianReflection(float r1, float r2, Vec3f& ray, Vec3f& normal) {
    Vec3f s = CosineSampleHemisphere(Vec2f(r1, r2));

    Vec3f w = normal; w.normalize();
    Vec3f u = cross((fabs(w.x) > .1 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)), w); u.normalize();
    Vec3f v = cross(w, u);

    ray = s.x * u + s.z * v + s.y * w;
    ray.normalize();
}