#pragma once

#include <cuda_runtime.h>
#include "linear_math.h"

#define PiOver4 0.78539816339
#define PiOver2 1.57079632679
#define Pi                    3.1415926535897932384626422832795028841971
#define TWO_PI				  6.2831853071795864769252867665590057683943

#define Epsilon 1e-5

#define E                     2.7182818284590452353602874713526624977572
#define SQRT_OF_ONE_THIRD     0.5773502691896257645091487805019574556476

using Point2f = Vec2f;
using Vector2f = Vec2f;
using Vector3f = Vec3f;
using Float = float;

__device__ inline Vec2f& operator*(float a, Vec2f& v){ v.x *= a; v.y *= a;           return v; }
__device__ inline Vec3f& operator*(float a, Vec3f& v){ v.x *= a; v.y *= a; v.z *= a; return v; }

// ------------------------------- perfect diffuse: lambertian reflection --------------------------------------

__device__ inline Point2f ConcentricSampleDisk(Point2f u) {
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

__device__ inline  Vector3f CosineSampleHemisphere(Point2f u) {
    Point2f d = ConcentricSampleDisk(u);
    Float z = sqrt(max((Float)0, 1 - d.x * d.x - d.y * d.y));
    return Vector3f(d.x, z, d.y);
}

__device__ inline void localizeSample(Vec3f& normal, Vec3f& u, Vec3f& v) {
    normal.normalize();
    Vec3f w;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) { 
		w = Vec3f(1, 0, 0);
	} else if (abs(normal.y) < SQRT_OF_ONE_THIRD) { 
		w = Vec3f(0, 1, 0);
	} else {
		w = Vec3f(0, 0, 1);
    }
    u = cross(normal, w).normalize();
    v = cross(normal, u);
}

__device__ inline void lambertianReflection(float r1, float r2, Vec3f& ray, Vec3f& normal) {
    Vec3f s = CosineSampleHemisphere(Vec2f(r1, r2));

    Vec3f u, v;
    localizeSample(normal, u, v);

    ray = s.x * u + s.z * v + s.y * normal;
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

// --------------------------- media -------------------------------------

__device__ inline Vector3f UniformSampleSphere(Point2f u) {
    Float z = 1 - 2 * u.x;
    Float r = sqrt(max((Float)0, (Float)1 - z * z));
    Float phi = TWO_PI * u.y;
    return Vector3f(r * cos(phi), z, r * sin(phi));
}

__device__ inline void CoordinateSystem(Vector3f &v1, Vector3f &v2, Vector3f &v3) {
    if (abs(v1.x) > abs(v1.z))
        v2 = Vector3f(-v1.y, v1.x, 0) / sqrt(v1.x * v1.x + v1.y * v1.y);
    else
        v2 = Vector3f(0, -v1.z, v1.y) / sqrt(v1.z * v1.z + v1.y * v1.y);
    v3 = cross(v1, v2);
}

__device__ inline Vector3f HenyeyGreensteinSample(Point2f u, Float g, Vector3f &raydir) {
    Float cosTheta;

    if (abs(g) < 1e-3)
        cosTheta = 1 - 2 * u.x;
    else {
        Float sqrTerm = (1 - g * g) / (1 - g + 2 * g * u.x);
        cosTheta = (1 + g * g - sqrTerm * sqrTerm) / (2 * g);
    }

    Float sinTheta = sqrt(max((Float)0, 1 - cosTheta * cosTheta));
    Float phi = 2 * Pi * u.y;

    Vector3f v1, v2;
    //CoordinateSystem(raydir, v1, v2);
    localizeSample(raydir, v1, v2);

    return  sinTheta * cos(phi) * v1 
          + sinTheta * sin(phi) * v2 
          + cosTheta            * raydir;
}

__device__ inline void HomogeneousMedium (
    float r1, 
    float r2,
    float r3,
    float r4,

    Vec3f& color,  
    Vec3f sigmaT,
    Vec3f& sigmaS,
    float g,

    float sceneT,

    Vec3f& rayorig,
    Vec3f& raydir,

    Vec3f& hitpoint,
    Vec3f& nextdir,

    bool& sampledMedium
    )
{
    // sample a channel (r g b)
    int channel = r1 * 3.0f;

    // sample a distance along the ray
    Float dist = - logf(1.0f - r2) / sigmaT._v[channel];

    // sampled medium or hit surface
    sampledMedium = dist < sceneT;

    // ray length
    Float t;
    if (sampledMedium) {
        t = dist;
    } else {
        t = sceneT;  
    }
    t = min(t, 1e20f);

    // transmission
    Vec3f Tr = expf(sigmaT * (-1.f) *  t);

    if (sampledMedium) {
        // scatter
        hitpoint = rayorig + t * raydir;
        //nextdir = UniformSampleSphere(Vec2f(r3, r4));
        nextdir = HenyeyGreensteinSample(Vec2f(r3, r4), g, raydir);
        //nextdir = raydir;
        nextdir.normalize();

        // absorb
        Vec3f density = sigmaT * Tr;
        Float pdf = (density._v[0] + density._v[1] + density._v[2]) / 3.0f;
        color *= Tr * sigmaS / pdf;
        //color *= expf(sigmaS * (-1.f) *  t);
    } else {
        // absorb
        Vec3f density = Tr;
        Float pdf = (density._v[0] + density._v[1] + density._v[2]) / 3.0f;
        color *= Tr / pdf;
        //color *= expf(sigmaS * (-1.f) *  t);
    }
}