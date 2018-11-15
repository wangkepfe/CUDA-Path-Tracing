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

// ------------------------------- perfect diffuse: lambertian reflection --------------------------------------

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

    Vec3f w = normal;
    Vec3f u = cross((fabs(w.x) > .1 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)), w);
    Vec3f v = cross(w, u);

    ray = s.x * u + s.z * v + s.y * w;
    ray.normalize();
}

// ------------------------------- smooth glass : specular reflection and transmission --------------------------------

__device__ inline void specularGlass (
    float r1,
    bool into, 
    Vec3f& raydir, 
    Vec3f& nextdir,
    Vec3f& n,
    bool& refl)
{
    const float etaI = 1.0f;
    const float etaT = 1.4f;
    const float eta = into ? etaI/etaT : etaT/etaI;

    Float cosThetaI = abs(dot(n, raydir)); // _cosThetaT_ needs to be positive
    Float sin2ThetaI = max(Float(0), Float(1.0f - cosThetaI * cosThetaI));
    Float sin2ThetaT = eta * eta * sin2ThetaI;

    if (sin2ThetaT >= 1) { // total internal reflection
        nextdir = raydir - n * 2.0f * dot(n, raydir);
        refl = true;
    } else { // transmission or reflection decided by fresnel equation
        // Compute _cosThetaT_ using Snell's law
        Float cosThetaT = sqrt(max((Float)0, 1 - sin2ThetaT));

        // fresnel for dialectric
        Float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                      ((etaT * cosThetaI) + (etaI * cosThetaT));
        Float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                      ((etaI * cosThetaI) + (etaT * cosThetaT));

        // square average
        float fresnel = (Rparl * Rparl + Rperp * Rperp) / 2;

        if (r1 > fresnel) { // transmission ray
            nextdir = eta * raydir + (eta * cosThetaI - cosThetaT) * n;
            refl = false;
        } else { // reflection ray
            nextdir = raydir - n * 2.0f * dot(n, raydir);
            refl = true;
        }
    }

    nextdir.normalize();
}