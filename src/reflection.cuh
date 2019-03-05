#pragma once

#include <cuda_runtime.h>
#include "linear_math.h"
#include "mathDefine.h"

// ------------------------------- helper functions -----------------------------------

__device__ inline float pow5(float e) {
    float e2 = e * e;
    return e2 * e2 * e;
}

__device__ inline Vec3f fresnelShlick(Vec3f& F0, float cosTheta) {
    return F0 + (Vec3f(1.0f) - F0) * pow5(1.0f - cosTheta);
}

__device__ inline float fresnelShlick(float F0, float cosTheta) {
    return F0 + (1.0f - F0) * pow5(1.0f - cosTheta);
}

__device__ inline float fresnelDielectric(float cosThetaI, float etaI, float etaT) {
    float eta = etaI / etaT;
    float cosThetaT = sqrt(1.0f - (1.0f - cosThetaI * cosThetaI) * eta * eta);
    float r1 = etaT * cosThetaI;
    float r2 = etaI * cosThetaT;
    float r3 = etaI * cosThetaI;
    float r4 = etaT * cosThetaT;
    float rp = (r1 - r2) / (r1 + r2);
    float rs = (r3 - r4) / (r3 + r4);
    return (rp * rp + rs * rs) / 2.0f;
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

// ------------------------------- perfect diffuse: lambertian reflection --------------------------------------

__device__ inline Vec2f ConcentricSampleDisk(Vec2f u) {
    // Map uniform random numbers to $[-1,1]^2$
    Vec2f uOffset = 2.f * u - Vec2f(1, 1);

    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0) return Vec2f(0, 0);

    // Apply concentric mapping to point
    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }
    return r * Vec2f(cos(theta), sin(theta));
}

__device__ inline  Vec3f CosineSampleHemisphere(Vec2f u) {
    Vec2f d = ConcentricSampleDisk(u);
    float z = sqrt(max((float)0, 1 - d.x * d.x - d.y * d.y));
    return Vec3f(d.x, z, d.y);
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
    bool& refl,
    float etaT)
{
    const float etaI = 1.0f;
    const float eta = into ? etaI/etaT : etaT/etaI;

    float cosThetaI = abs(dot(n, raydir)); // _cosThetaT_ needs to be positive
    float sin2ThetaI = max(float(0), float(1.0f - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    if (sin2ThetaT >= 1) { // total internal reflection
        nextdir = raydir - n * 2.0f * dot(n, raydir);
        refl = true;
    } else { // transmission or reflection decided by fresnel equation
        // Snell's law
        float cosThetaT = sqrt(max((float)0, 1 - sin2ThetaT));

        // Fresnel for dialectric
        float R1 = etaT * cosThetaI, R2 = etaI * cosThetaT;
        float R3 = etaI * cosThetaI, R4 = etaT * cosThetaT;
        float Rparl = (R1 - R2) / (R1 + R2);
        float Rperp = (R3 - R4) / (R3 + R4);
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

__device__ inline Vec3f HenyeyGreensteinSample(Vec2f u, float g, Vec3f &raydir) {
    float cosTheta;

    if (abs(g) < 1e-3)
        cosTheta = 1 - 2 * u.x;
    else {
        float sqrTerm = (1 - g * g) / (1 - g + 2 * g * u.x);
        cosTheta = (1 + g * g - sqrTerm * sqrTerm) / (2 * g);
    }

    float sinTheta = sqrt(max((float)0, 1 - cosTheta * cosTheta));
    float phi = 2 * Pi * u.y;

    Vec3f v1, v2;
    localizeSample(raydir, v1, v2);

    return  sinTheta * cos(phi) * v1 
          + sinTheta * sin(phi) * v2 
          + cosTheta            * raydir;
}

__device__ inline void HomogeneousMedium (
    float r1, float r2, float r3, float r4,

    Vec3f& color,  
    Vec3f sigmaT,
    Vec3f& sigmaS,
    float g,
    float sceneT,

    Vec3f& rayorig,
    Vec3f& raydir,
    Vec3f& hitpoint,
    Vec3f& nextdir,

    bool& sampledMedium)
{
    // sample a channel (r g b)
    int channel = r1 * 3.0f;

    // sample a distance along the ray
    float dist = - logf(1.0f - r2) / sigmaT[channel];

    // sampled medium or hit surface
    sampledMedium = dist < sceneT;

    // ray length
    float t =  min(sampledMedium ? dist : sceneT, 1e20f);

    // transmission (Beer's Law)
    Vec3f Tr = expf(sigmaT * (-1.f) *  t);

    // scatter
    if (sampledMedium) {
        hitpoint = rayorig + t * raydir;
        nextdir = HenyeyGreensteinSample(Vec2f(r3, r4), g, raydir);
        nextdir.normalize();
    }

    // absorption
    Vec3f density = sampledMedium ? (sigmaT * Tr) : Tr;

    float pdf = (density[0] + density[1] + density[2]) / 3.0f;
    if (pdf < 1e-4) pdf = 1.0f;

    color *= sampledMedium ? (Tr * sigmaS / pdf) : (Tr / pdf);
}

// --------------------------- microfacet -------------------------------------

__device__ inline void macrofacetReflection (
    float r1, float r2,
    Vec3f& raydir, Vec3f& nextdir, Vec3f& normal, Vec3f& tangent,
    Vec3f& beta, Vec3f& F0,
    float alphax, float alphay)
{
    // isotropic
    bool isotropic = alphax == alphay;

    // precalculate
    float alphax2 = alphax * alphax;
    float alphay2 = alphay * alphay;

    // sample normal
    Vec3f sampledNormalLocal;
    if (isotropic) {
        float cosTheta = 1.0f / sqrt(1.0f + alphax2 * r1 / (1.0f - r1));
        float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
        float phi = TWO_PI * r2;
        sampledNormalLocal = Vec3f(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
    } else {
        float phi = atan(alphay / alphax * tan(TWO_PI * r1 + PiOver2));
        if (r1 > 0.5f) phi += Pi;
        float sinPhi = sin(phi);
        float cosPhi = cos(phi);
        float cosTheta = 1 / sqrt(1 + 1.0f / ((cosPhi*cosPhi) / alphax2 + (sinPhi*sinPhi) / alphay2) * r2 / (1 - r2));
        float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
        sampledNormalLocal = Vec3f(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
    }

    // local to world space
    Vec3f t, b;
    if (isotropic) {
        localizeSample(normal, t, b);
    } else {
        normal.normalize();
        tangent.normalize();
        t = tangent;
        b = cross(normal, tangent);
    }
    Vec3f sampledNormal = sampledNormalLocal.x * t + sampledNormalLocal.z * b + sampledNormalLocal.y * normal;
    sampledNormal.normalize();

    // reflect
    nextdir = raydir - sampledNormal * dot(sampledNormal, raydir) * 2.0f;
    nextdir.normalize();

    // Fresnel
    float cosThetaWoWh = max(0.01f, abs(dot(sampledNormal, nextdir)));
    Vec3f F = fresnelShlick(F0, cosThetaWoWh);

    // Smith's Mask-shadowing function G
    float G;
    float cosThetaWo = abs(dot(nextdir, normal));
    float cosThetaWi = max(0.01f, abs(dot(raydir, normal)));
    float tanThetaWo = sqrt(1.0f - cosThetaWo * cosThetaWo) / cosThetaWo;
    if (isotropic) { 
        G = 1.0f / (1.0f + (sqrtf(1.0f + alphax2 * tanThetaWo * tanThetaWo) - 1.0f) / 2.0f);
    } else {
        float cos2PhiWo = dot(cross(nextdir, normal), b); cos2PhiWo *= cos2PhiWo;
        float alpha = sqrt(cos2PhiWo * alphax2 + (1.0f - cos2PhiWo) * alphay2);
        float alphaTan = alpha * tanThetaWo;
        G = 1.0f / (1.0f + (sqrtf(1.0f + alphaTan * alphaTan) - 1.0f) / 2.0f);
    }

    // color
    float cosThetaWh = max(0.01f, dot(sampledNormal, normal));
    beta = minf3f(1.0f, F * G * cosThetaWoWh / cosThetaWi / cosThetaWh);
}

__device__ inline void microfacetSampling(
    float r1, float r2, bool into, 
    Vec3f& raydir, Vec3f& normal, bool& refl, 
    float etaT, float alpha, 
    Vec3f &sampledNormal, Vec3f& beta, Vec3f &nextdir
) {
    // sample normal
    float alpha2 = alpha * alpha;
    float cosTheta = 1.0f / sqrt(1.0f + alpha2 * r1 / (1.0f - r1));
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    float phi = TWO_PI * r2;
    Vec3f sampledNormalLocal = Vec3f(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));

    // local to world space
    Vec3f tangent, bitangent;
    localizeSample(normal, tangent, bitangent);
    sampledNormal = sampledNormalLocal.x * tangent + sampledNormalLocal.z * bitangent + sampledNormalLocal.y * normal;
    sampledNormal.normalize();

    // Fresnel for dialectric
    const float etaI = 1.0f;
    const float eta = into ? etaI/etaT : etaT/etaI;

    float cosThetaI = abs(dot(sampledNormal, raydir));
    float sin2ThetaI = max(float(0), float(1.0f - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI; // Snell's law
    // float cosThetaT = sqrt(max(0.0f, 1.0f - sin2ThetaT));
    // float R1 = etaT * cosThetaI, R2 = etaI * cosThetaT;
    // float R3 = etaI * cosThetaI, R4 = etaT * cosThetaT;
    // float Rparl = (R1 - R2) / (R1 + R2);
    // float Rperp = (R3 - R4) / (R3 + R4);
    // float fresnel = (Rparl * Rparl + Rperp * Rperp) / 2;
    float fresnel = fresnelShlick(0.03f, cosThetaI);

    // total internal reflection, fresnel decide transmission or reflection
    refl = sin2ThetaT >= 1 || (sin2ThetaT < 1 && r1 < fresnel);

    if (refl) {
        nextdir = raydir - sampledNormal * 2.0f * dot(sampledNormal, raydir);
    } else {
        return;
    }
    nextdir.normalize();

    // Smith's Mask-shadowing function G
    float cosThetaWo = abs(dot(nextdir, normal));
    float cosThetaWi = max(0.01f, abs(dot(raydir, normal)));
    float tanThetaWo = sqrt(1.0f - cosThetaWo * cosThetaWo) / cosThetaWo;
    float G = 1.0f / (1.0f + (sqrtf(1.0f + alpha2 * tanThetaWo * tanThetaWo) - 1.0f) / 2.0f);

    // color
    float cosThetaWh = max(0.01f, dot(sampledNormal, normal));
    beta = minf3f(1.0f, G * cosThetaI / cosThetaWi / cosThetaWh);
}

__device__ inline void macrofacetGlass (
    float r1, float r2, float r3,
    bool into,
    Vec3f& beta,
    Vec3f& raydir, 
    Vec3f& nextdir,
    Vec3f& normal,
    bool& refl,
    float etaT, 
    float alpha)
{
    // sample normal
    float alpha2 = alpha * alpha;
    float cosTheta = 1.0f / sqrt(1.0f + alpha2 * r1 / (1.0f - r1));
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    float phi = TWO_PI * r2;
    Vec3f sampledNormalLocal = Vec3f(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));

    // local to world space
    Vec3f tangent, bitangent;
    localizeSample(normal, tangent, bitangent);
    Vec3f sampledNormal = sampledNormalLocal.x * tangent + sampledNormalLocal.z * bitangent + sampledNormalLocal.y * normal;
    sampledNormal.normalize();

    // Fresnel for dialectric
    const float etaI = 1.0f;
    const float eta = into ? etaI/etaT : etaT/etaI;

    float cosThetaI = abs(dot(sampledNormal, raydir));
    float sin2ThetaI = max(float(0), float(1.0f - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI; // Snell's law
    float cosThetaT = sqrt(max(0.0f, 1.0f - sin2ThetaT));
    float R1 = etaT * cosThetaI, R2 = etaI * cosThetaT;
    float R3 = etaI * cosThetaI, R4 = etaT * cosThetaT;
    float Rparl = (R1 - R2) / (R1 + R2);
    float Rperp = (R3 - R4) / (R3 + R4);
    float fresnel = (Rparl * Rparl + Rperp * Rperp) / 2;        

    // total internal reflection, fresnel decide transmission or reflection
    refl = sin2ThetaT >= 1 || (sin2ThetaT < 1 && r1 < fresnel);

    if (refl) {
        nextdir = raydir - sampledNormal * 2.0f * dot(sampledNormal, raydir);
    } else {
        nextdir = eta * raydir + (eta * cosThetaI - cosThetaT) * sampledNormal;
    }
    nextdir.normalize();

    // Smith's Mask-shadowing function G
    float cosThetaWo = abs(dot(nextdir, normal));
    float cosThetaWi = max(0.01f, abs(dot(raydir, normal)));
    float tanThetaWo = sqrt(1.0f - cosThetaWo * cosThetaWo) / cosThetaWo;
    float G = 1.0f / (1.0f + (sqrtf(1.0f + alpha2 * tanThetaWo * tanThetaWo) - 1.0f) / 2.0f);

    // color
    float cosThetaWh = max(0.01f, dot(sampledNormal, normal));
    beta = minf3f(1.0f, G * cosThetaI / cosThetaWi / cosThetaWh);
}

__device__ inline void fresnelBlend (
    float r1, float r2, float r3,
    Vec3f& raydir, Vec3f& nextdir, Vec3f& normal,
    Vec3f& beta, 
    Vec3f Rd, Vec3f& Rs, float alpha) 
{
    float alpha2 = alpha * alpha;
    Vec3f wh;

    if (r3 < 0.5f) { // diffuse
        lambertianReflection(r1, r2, nextdir, normal);
        wh = nextdir - raydir;
    } else { // reflection

        // sample normal
        float cosTheta = 1.0f / sqrt(1.0f + alpha2 * r1 / (1.0f - r1));
        float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
        float phi = TWO_PI * r2;
        Vec3f sampledNormalLocal = Vec3f(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));

        // local to world space
        Vec3f tangent, bitangent;
        localizeSample(normal, tangent, bitangent);
        wh = sampledNormalLocal.x * tangent + sampledNormalLocal.z * bitangent + sampledNormalLocal.y * normal;

        // reflect
        nextdir = raydir - wh * dot(wh, raydir) * 2.0f;
    }
    normal.normalize();
    wh.normalize();
    nextdir.normalize();

    Vec3f wo = raydir.normalize();
    float cosThetaWi = abs(dot(nextdir, normal));
    float cosThetaWo = min(0.01f, abs(dot(wo, normal)));
    float cosThetaWh = min(0.01f, abs(dot(wh, normal)));

    // distribution D
    float cos2ThetaWh = cosThetaWh * cosThetaWh;
    float tan2ThetaWh = (1.0f - cos2ThetaWh) / cos2ThetaWh;
    float cos4ThetaWh = cos2ThetaWh * cos2ThetaWh;
    float e = 1.0f + tan2ThetaWh / alpha2;
    float D = 1.0f / (Pi * alpha2 * cos4ThetaWh * e * e);

    // f(wi, wo)
    float dotWiWh = min(0.01f, abs(dot(nextdir, wh)));
    Vec3f diff = (28.0f / (23.0f * Pi)) * Rd * (Vec3f(1.0f) - Rs) * (1.0f - pow5(1.0f - 0.5f * cosThetaWi)) * (1.0f - pow5(1.0f - 0.5f * cosThetaWo));
    Vec3f spec = D / (4.0f * dotWiWh * max(cosThetaWi, cosThetaWo)) * fresnelShlick(Rs, dotWiWh);
    Vec3f f = spec + diff;

    // pdf
    float pdf = 0.5f * (cosThetaWi * InvPi + D / (4.0f * dotWiWh));

    // beta
    beta = f * cosThetaWi / pdf;
}