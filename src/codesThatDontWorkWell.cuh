#pragma once

#include <cuda_runtime.h>
#include "linear_math.h"

#define PiOver4 0.78539816339
#define PiOver2 1.57079632679
#define Pi                    3.1415926535897932384626422832795028841971

#ifndef TWO_PI
#define TWO_PI 6.2831853071795864769252867665590057683943
#endif

#define invPi                 0.31830988618f

#define Epsilon 1e-5

#define E                     2.7182818284590452353602874713526624977572
#define SQRT_OF_ONE_THIRD     0.5773502691896257645091487805019574556476

using Point2f = Vec2f;
using Vector2f = Vec2f;
using Vector3f = Vec3f;
using Float = float;
using Spectrum = Vec3f;

__device__ inline void CoordinateSystem(Vector3f &v1, Vector3f &v2, Vector3f &v3) {
    if (abs(v1.x) > abs(v1.z))
        v2 = Vector3f(-v1.y, v1.x, 0) / sqrt(v1.x * v1.x + v1.y * v1.y);
    else
        v2 = Vector3f(0, -v1.z, v1.y) / sqrt(v1.z * v1.z + v1.y * v1.y);
    v3 = cross(v1, v2);
}

__device__ inline Vector3f UniformSampleSphere(Point2f u) {
    Float z = 1 - 2 * u.x;
    Float r = sqrt(max((Float)0, (Float)1 - z * z));
    Float phi = TWO_PI * u.y;
    return Vector3f(r * cos(phi), z, r * sin(phi));
}


__device__ inline Vector3f SphericalDirection(Float sinTheta, Float cosTheta, Float phi) { return Vector3f(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta); }
__device__ inline Float CosTheta(const Vector3f &w) { return w.z; }
__device__ inline Float Cos2Theta(const Vector3f &w) { return w.z * w.z; }
__device__ inline Float AbsCosTheta(const Vector3f &w) { return abs(w.z); }
__device__ inline bool SameHemisphere(const Vector3f &w, const Vector3f &wp) { return w.z * wp.z > 0; }
__device__ inline Float Sin2Theta(const Vector3f &w) { return max((Float)0, (Float)1 - Cos2Theta(w)); }
__device__ inline Float SinTheta(const Vector3f &w) { return sqrt(Sin2Theta(w)); }
__device__ inline Float CosPhi(const Vector3f &w) { Float sinTheta = SinTheta(w); return (sinTheta == 0) ? 1 : clamp(w.x / sinTheta, -1, 1); }
__device__ inline Float SinPhi(const Vector3f &w) { Float sinTheta = SinTheta(w); return (sinTheta == 0) ? 0 : clamp(w.y / sinTheta, -1, 1); }
__device__ inline Float Cos2Phi(const Vector3f &w) { return CosPhi(w) * CosPhi(w); }
__device__ inline Float Sin2Phi(const Vector3f &w) { return SinPhi(w) * SinPhi(w); }
__device__ inline Float TanTheta(const Vector3f &w) { return SinTheta(w) / CosTheta(w); }
__device__ inline Float Tan2Theta(const Vector3f &w) { return Sin2Theta(w) / Cos2Theta(w); }

__device__ inline Float TrowbridgeReitzDistributionRoughnessToAlpha(Float roughness) {
    roughness = max(roughness, (Float)1e-3);
    Float x = log(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}

__device__ inline Float FrDielectric(Float cosThetaI, Float etaI, Float etaT) {
    cosThetaI = clamp(cosThetaI, -1, 1);
    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        swap(etaI, etaT);
        cosThetaI = abs(cosThetaI);
    }

    // Compute _cosThetaT_ using Snell's law
    Float sinThetaI = sqrt(max((Float)0, 1 - cosThetaI * cosThetaI));
    Float sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    if (sinThetaT >= 1) return 1;
    Float cosThetaT = sqrt(max((Float)0, 1 - sinThetaT * sinThetaT));
    Float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                  ((etaT * cosThetaI) + (etaI * cosThetaT));
    Float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                  ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2;
}

__device__ inline void TrowbridgeReitzSample11(Float cosTheta, Float U1, Float U2, Float *slope_x, Float *slope_y) {
    // special case (normal incidence)
    if (cosTheta > .9999f) {
        Float r = sqrt(U1 / (1 - U1));
        Float phi = 6.28318530718 * U2;
        *slope_x = r * cos(phi);  
        *slope_y = r * sin(phi);
        return;
    }

    Float sinTheta = sqrt(max((Float)0, (Float)1 - cosTheta * cosTheta));
    Float tanTheta = sinTheta / cosTheta;
    Float a = 1 / tanTheta;
    Float G1 = 2 / (1 + sqrt(1.f + 1.f / (a * a)));

    // sample slope_x
    Float A = 2 * U1 / G1 - 1;
    Float tmp = 1.f / (A * A - 1.f);
    if (tmp > 1e10) tmp = 1e10;
    Float B = tanTheta;
    Float D = sqrt(max(Float(B * B * tmp * tmp - (A * A - B * B) * tmp), Float(0)));
    Float slope_x_1 = B * tmp - D;
    Float slope_x_2 = B * tmp + D;
    *slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

    // sample slope_y
    Float S;
    if (U2 > 0.5f) {
        S = 1.f;
        U2 = 2.f * (U2 - .5f);
    } else {
        S = -1.f;
        U2 = 2.f * (.5f - U2);
    }
    Float z = (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) / (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
    *slope_y = S * z * sqrt(1.f + *slope_x * *slope_x);
}

__device__ inline Vector3f TrowbridgeReitzSample(const Vector3f &wi, Float alpha_x, Float alpha_y, Float U1, Float U2) {
    // 1. stretch wi
    Vector3f wiStretched = Vector3f(alpha_x * wi.x, alpha_y * wi.y, wi.z).normalize();

    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    Float slope_x, slope_y;
    TrowbridgeReitzSample11(CosTheta(wiStretched), U1, U2, &slope_x, &slope_y);

    // 3. rotate
    Float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
    slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
    slope_x = tmp;

    // 4. unstretch
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;

    // 5. compute normal
    return Vector3f(-slope_x, -slope_y, 1.).normalize();
}

__device__ inline Vector3f TrowbridgeReitzDistributionSampleNormal(
    float alphax,
    float alphay,
    Vector3f &wo, 
    Point2f u) 
{
    Vector3f wh;
    const bool sampleVisibleArea = true;
    if (!sampleVisibleArea) {
        Float cosTheta = 0;
        Float phi = (2 * Pi) * u[1];

        if (alphax == alphay) {
            Float tanTheta2 = alphax * alphax * u[0] / (1.0f - u[0]);
            cosTheta = 1 / sqrt(1 + tanTheta2);
        } else {
            phi = atan(alphay / alphax * tan(2 * Pi * u[1] + .5f * Pi));
            if (u[1] > .5f) {
                phi += Pi;
            }
            Float sinPhi = sin(phi), cosPhi = cos(phi);
            const Float alphax2 = alphax * alphax;
            const Float alphay2 = alphay * alphay;
            const Float alpha2 = 1 / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
            Float tanTheta2 = alpha2 * u[0] / (1 - u[0]);
            cosTheta = 1 / sqrt(1 + tanTheta2);
        }
        Float sinTheta = sqrt(max((Float)0., (Float)1. - cosTheta * cosTheta));
        wh = SphericalDirection(sinTheta, cosTheta, phi);
        if (!SameHemisphere(wo, wh)) wh = -wh;
    } else {
        bool flip = wo.z < 0;

        wh = TrowbridgeReitzSample(flip ? -wo : wo, alphax, alphay, u[0], u[1]);

        if (flip) {
            wh = -wh;
        }
    }
    return wh;
}

__device__ inline Float TrowbridgeReitzDistributionLambda(float alphax, float alphay, Vector3f &w) {
    Float absTanTheta = abs(TanTheta(w));
    if (isinf(absTanTheta)) return 0.;
    // Compute _alpha_ for direction _w_
    Float alpha = sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
    Float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
    return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
}

__device__ inline Float TrowbridgeReitzDistributionG(float alphax, float alphay, Vector3f &wo, Vector3f &wi) {
    return 1 / (1 + TrowbridgeReitzDistributionLambda(alphax, alphay, wo) + TrowbridgeReitzDistributionLambda(alphax, alphay, wi));
}

__device__ inline Float TrowbridgeReitzDistributionG1(float alphax, float alphay, Vector3f &w) {
    return 1 / (1 + TrowbridgeReitzDistributionLambda(alphax, alphay, w));
}

__device__ inline Float TrowbridgeReitzDistributionD(float alphax, float alphay, Vector3f &wh)  {
    Float tan2Theta = Tan2Theta(wh);
    if (isinf(tan2Theta)) return 0.;
    const Float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
    Float e = (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay)) * tan2Theta;
    return 1 / (Pi * alphax * alphay * cos4Theta * (1 + e) * (1 + e));
}

__device__ inline Float MicrofacetDistributionPdf(float alphax, float alphay, Vector3f &wo, Vector3f &wh) {
    const bool sampleVisibleArea = true;
    if (sampleVisibleArea)
        return TrowbridgeReitzDistributionD(alphax, alphay, wh) * TrowbridgeReitzDistributionG1(alphax, alphay, wo) * abs(dot(wo, wh)) / AbsCosTheta(wo);
    else
        return TrowbridgeReitzDistributionD(alphax, alphay, wh) * AbsCosTheta(wh);
}

__device__ inline Float MicrofacetReflectionPdf(float alphax, float alphay, Vector3f &wo, Vector3f &wi) { // not get called
    Vector3f wh = normalize(wo + wi);
    return MicrofacetDistributionPdf(alphax, alphay, wo, wh) / (4 * dot(wo, wh));
}

__device__ inline Vec3f MicrofacetReflectionF(float alphax, float alphay, Vec3f& R, Vector3f &wo, Vector3f &wi) {
    Float cosThetaO = AbsCosTheta(wo);
    Float cosThetaI = AbsCosTheta(wi);
    Vector3f wh = wi + wo;
    // Handle degenerate cases for microfacet reflection
    if (cosThetaI == 0 || cosThetaO == 0) return Vec3f(0.f);
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Vec3f(0.f);
    wh = normalize(wh);
    float F = FrDielectric(dot(wi, wh), 1.5f, 1.0f);
    float D = TrowbridgeReitzDistributionD(alphax, alphay, wh);
    float G = TrowbridgeReitzDistributionG(alphax, alphay, wo, wi);
    return R * (D * G * F / (4 * cosThetaI * cosThetaO));
}

__device__ inline void microfacetReflection2 (
    float r1,
    float r2,

    Vec3f& raydir,
    Vec3f& nextdir,
    Vec3f& normal,
    Vec3f& tangent,

    Vec3f& surfaceColor,
    Vec3f& colorMask,

    float roughnessX,
    float roughnessY)
{
    float alphax = TrowbridgeReitzDistributionRoughnessToAlpha(roughnessX);
    float alphay = (roughnessX == roughnessY) ? alphax : TrowbridgeReitzDistributionRoughnessToAlpha(roughnessY);

    // world space to local space
    Vec3f bitangent = cross(normal, tangent).normalize();

    Vec3f transposeTNB[3];
    transposeTNB[0] = Vec3f(tangent.x, normal.x, bitangent.x);
    transposeTNB[1] = Vec3f(tangent.y, normal.y, bitangent.y);
    transposeTNB[2] = Vec3f(tangent.z, normal.z, bitangent.z);

    Vec3f raydirLocal = transposeTNB[0] * raydir.x + transposeTNB[1] * raydir.y + transposeTNB[2] * raydir.z;

    // swizzle, flip
    Vec3f localOutfaceRayDir = Vec3f(-raydirLocal.x, -raydirLocal.z, -raydirLocal.y);
    
    // sample normal
    Vec3f sampledNormalLocal = TrowbridgeReitzDistributionSampleNormal(alphax, alphay, localOutfaceRayDir, Vec2f(r1, r2));

    // reflect
    Vec3f localNextdir = (-localOutfaceRayDir) - sampledNormalLocal * 2.0f * dot(sampledNormalLocal, (-localOutfaceRayDir));

    // pdf
    float pdf = MicrofacetDistributionPdf(alphax, alphay, localOutfaceRayDir, sampledNormalLocal) / (4 * dot(localOutfaceRayDir, sampledNormalLocal));

    // color
    colorMask *= MicrofacetReflectionF(alphax, alphay, surfaceColor, localOutfaceRayDir, localNextdir) * dot(localOutfaceRayDir, sampledNormalLocal) / pdf; 

    // swizzle
    localNextdir = Vec3f(localNextdir.x, localNextdir.z, localNextdir.y);

    // local space to world space
    nextdir = tangent * localNextdir.x + normal * localNextdir.y + bitangent * localNextdir.z;
}


__device__ inline void plasticReflection (
    float r1,
    float r2,

    Vec3f& raydir,
    Vec3f& nextdir,
    Vec3f& normal,

    Vec3f& surfaceColor,
    Vec3f& colorMask,

    float alpha,
    float kd,
    float ks)
{
    // roughnes to alpha: (0,1) -> (0.5, 1.5)
    // roughness = max(roughness, 1e-3f);
    // float x = log(roughness);
    // float x2 = x * x;
    // float x3 = x2 * x;
    // float x4 = x3 * x;
    // float alpha = 1.62142f + 0.819955f * x + 0.1734f * x2 + 0.0171201f * x3 + 0.000640711f * x4;

    // sample normal
    float cosTheta = 1.0f / sqrt(1.0f + alpha * alpha * r1 / (1.0f - r1));
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    float phi = TWO_PI * r2;
    Vec3f sampledNormalLocal = Vec3f(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));

    // local to world space
    Vec3f tangent, bitangent;
    localizeSample(normal, tangent, bitangent);
    Vec3f sampledNormal = sampledNormalLocal.x * tangent + sampledNormalLocal.z * bitangent + sampledNormalLocal.y * normal;
    sampledNormal.normalize();

    // reflect
    nextdir = raydir - sampledNormal * dot(sampledNormal, raydir) * 2.0f;

    // preparation for polar coord sys calculation
    Vec3f wo = normalize(-raydir);
    Vec3f wi = normalize(nextdir);
    Vec3f& n = normal;
    Vec3f& wh = sampledNormal;

    float cosThetaWi = dot(wi, n);
    float cosThetaWo = dot(wo, n);
    float cosThetaWh = dot(wh, n);

    if (cosThetaWi < 1e-2f || cosThetaWh < 1e-2f || cosThetaWo < 1e-2f) { // bad cases
        colorMask = 0.0f;
        return;
    }

    float sinThetaWi = sqrtf(1.0f - cosThetaWi * cosThetaWi);
    float tanThetaWi = sinThetaWi / cosThetaWi;

    float sinThetaWo = sqrtf(1.0f - cosThetaWo * cosThetaWo);
    float tanThetaWo = sinThetaWo / cosThetaWo;

    float cos2ThetaWh = cosThetaWh * cosThetaWh;
    float sin2ThetaWh = 1.0f - cos2ThetaWh;
    float tan2ThetaWh = sin2ThetaWh / cos2ThetaWh;
    float cos4ThetaWh = cos2ThetaWh * cos2ThetaWh;

    float dotHalfOut = dot(wh, wo);

    // Trowbridge Reitz halfway vector distribution function D
    float alpha2 = alpha * alpha;
    float e = tan2ThetaWh / alpha2;
    float D = 1.0f / (Pi * alpha2 * cos4ThetaWh * (1 + e) * (1 + e));

    // Smith's Mask-shadowing function G
    float lambdaWo = (-1.0f + sqrtf(1.0f + (alpha * tanThetaWo) * (alpha * tanThetaWo))) / 2.0f;
    float lambdaWi = (-1.0f + sqrtf(1.0f + (alpha * tanThetaWi) * (alpha * tanThetaWi))) / 2.0f;
    float G = 1.0f / (1.0f + lambdaWo + lambdaWi);

    // Fresnel F
    float etaI = 1.0f;
    float etaT = 1.5f;
    float eta = etaI / etaT;
    float cosThetaI = dot(wi, wh);
    float cosThetaT = sqrt(max(0.0f, 1.0f - eta * eta * max(0.0f, 1.0f - cosThetaI * cosThetaI)));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    float F = (Rparl * Rparl + Rperp * Rperp) / 2.0f;

    // BSDF function f(wo, wi)
    Vec3f fd = kd * surfaceColor * invPi;
    float fs = ks * min((D * G * F) / (4.0f * cosThetaWo * cosThetaWi), 1.0f);
    Vec3f f = fd + Vec3f(fs, fs, fs);

    // pdf
    float pdf = (D * cosThetaWh) / (4.0f * dotHalfOut);

    // color    
    Vec3f beta = f * cosThetaWo / pdf;

    colorMask *= beta;
}


// ---------------------------- BSSRDF -------------------------------

// __device__ inline Float FrDielectric(Float cosThetaI, Float etaI, Float etaT) {
//     cosThetaI = clamp(cosThetaI, -1, 1);
//     // Potentially swap indices of refraction
//     bool entering = cosThetaI > 0.f;
//     if (!entering) {
//         swap(etaI, etaT);
//         cosThetaI = abs(cosThetaI);
//     }

//     // Compute _cosThetaT_ using Snell's law
//     Float sinThetaI = sqrt(max((Float)0, 1 - cosThetaI * cosThetaI));
//     Float sinThetaT = etaI / etaT * sinThetaI;

//     // Handle total internal reflection
//     if (sinThetaT >= 1) return 1;
//     Float cosThetaT = sqrt(max((Float)0, 1 - sinThetaT * sinThetaT));
//     Float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
//                   ((etaT * cosThetaI) + (etaI * cosThetaT));
//     Float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
//                   ((etaI * cosThetaI) + (etaT * cosThetaT));
//     return (Rparl * Rparl + Rperp * Rperp) / 2;
// }

// struct BSSRDFTable {
//     const int nRhoSamples = 100;
//     const int nRadiusSamples = 64;

//     Float[100] rhoSamples;
//     Float[64] radiusSamples;
//     Float[6400] profile;
//     Float[100] rhoEff;
//     Float[] profileCDF;
// };

// void ComputeBeamDiffusionBSSRDF(Float g, Float eta, BSSRDFTable *t) {
//     // Choose radius values of the diffusion profile discretization
//     t->radiusSamples[0] = 0;
//     t->radiusSamples[1] = 2.5e-3f;
//     for (int i = 2; i < t->nRadiusSamples; ++i)
//         t->radiusSamples[i] = t->radiusSamples[i - 1] * 1.2f;

//     // Choose albedo values of the diffusion profile discretization
//     for (int i = 0; i < t->nRhoSamples; ++i) {
//         t->rhoSamples[i] = (1 - exp(-8 * i / (Float)(t->nRhoSamples - 1))) / (1 - exp(-8));

//         // Compute the diffusion profile for the _i_th albedo sample

//         // Compute scattering profile for chosen albedo $\rho$
//         for (int j = 0; j < t->nRadiusSamples; ++j) {
//             Float rho = t->rhoSamples[i], r = t->radiusSamples[j];
//             t->profile[i * t->nRadiusSamples + j] =
//                 2 * Pi * r * (BeamDiffusionSS(rho, 1 - rho, g, eta, r) +
//                               BeamDiffusionMS(rho, 1 - rho, g, eta, r));
//         }

//         // Compute effective albedo $\rho_{\roman{eff}}$ and CDF for importance
//         // sampling
//         t->rhoEff[i] =
//             IntegrateCatmullRom(t->nRadiusSamples, t->radiusSamples.get(),
//                                 &t->profile[i * t->nRadiusSamples],
//                                 &t->profileCDF[i * t->nRadiusSamples]);
//     }
    
// }

// __device__ inline Float TabulatedBSSRDF_Sample_Sr(
//     int ch, 
//     Float u,
//     Vec3f& sigma_t,
//     Vec3f& rho,
//     BSSRDFTable& table
// ) {
//     if (sigma_t[ch] == 0) return -1;
//     return SampleCatmullRom2D(table.nRhoSamples, 
//         table.nRadiusSamples,
//         table.rhoSamples.get(), 
//         table.radiusSamples.get(),
//         table.profile.get(), 
//         table.profileCDF.get(),
//         rho[ch], u) / sigma_t[ch];
// }

// // BSSRDF
// // integral of S(wi, pi, wo, po) * L(wo, wi) * cos(dot(n,wo))

// // separatable BSSRDF
// // S = (1 - Fr(cos(wo))) * Sp * Sw
// // Sw = Fr(cos(wi)) / c

// // tabulated BSSRDF
// // Sp(pi, po) = Sr(|po - pi|) = Sr(r)
// //
// // tabulate Sr(radius, albedo) as a lookup table

// __device__ inline bool BSSRDF_FresnelRefl( // decide ray refract into surface or reflect out
//     float r1,
//     Vec3f& raydir,
//     Vec3f& normal,
//     float eta
// ) {
//     raydir.normalize();
//     normal.normalize();
//     float fresnel = FrDielectric(dot(raydir, normal), 1.0f, eta);
//     return r1 < fresnel;
// }

// __device__ inline void BSSRDF_SrProbeRay( // sample axis and radius, get a probe ray
//     float u1,
//     float u2,
//     float u3,
//     Vec3f& normal,
// ) {
//     Vector3f ss, ts;
//     localizeSample(normal, ss, ts);

//     Vector3f vx, vy, vz;
//     if (u1 < .5f) {
//         vx = ss;
//         vy = normal;
//         vz = ts;
//         u1 *= 2;
//     } else if (u1 < .75f) {
//         vx = ts;
//         vy = ss;
//         vz = normal;
//         u1 = (u1 - .5f) * 4;
//     } else {
//         vx = normal;
//         vy = ss;
//         vz = ts;
//         u1 = (u1 - .75f) * 4;
//     }

//     // Choose spectral channel for BSSRDF sampling
//     int ch = u1 * 3.0f;
//     u1 = u1 * 3.0f - ch;

//     // Sample BSSRDF profile in polar coordinates
//     Float r = TabulatedBSSRDF_Sample_Sr(ch, u2);
//     if (r < 0) return Spectrum(0.f);
//     Float phi = 2 * Pi * u3;

//     // Compute BSSRDF profile bounds and intersection height
//     Float rMax = TabulatedBSSRDF_Sample_Sr(ch, 0.999f);
//     if (r >= rMax) return Spectrum(0.f);
//     Float l = 2 * sqrt(rMax * rMax - r * r);


// }

// __device__ inline void BSSRDF_Sw( // for sampled position on surface, sample a direction out
//     float r1, 
//     float r2, 
//     Vec3f& ray,
//     Vec3f& normal,
//     Vec3f& color,
//     float eta
// ) {
//     lambertianReflection(r1, r2, ray, normal);
//     Float c = 1 - 2 * FresnelMoment1(1 / eta);
//     color *= (1.0f - FrDielectric(dot(ray, normal), 1.0f, eta)) / (c * Pi) * eta * eta;
// }

		// if (geomtype == GEO_SPHERE) {
		// 	Sphere &hitsphere = spheres[hitSphereIdx]; // hit object with closest intersection
		// 	n = hitpoint - hitsphere.pos;	// normal
		// 	objcol = hitsphere.col;   // object colour
		// 	emit = hitsphere.emi;  // object emission
		// 	refltype = hitsphere.refl;
		// 	objMedium = hitsphere.medium;
        // } 
        
        		// spheres
		// Sphere spheres[] = {
		// 	//{ 0.78f, { 0.0f, 0.0f, -3.0f }, { 0.0, 0.0, 0.0 }, { 1.0f, 1.0f, 1.0f }, MAT_GLASS, 1},
		// 	{ 0.0f, { 0.0f, 0.0f, 0.0f }, { 0.0, 0.0, 0.0 }, { 0.0f, 0.0f, 0.0f }, MAT_DIFF, MEDIUM_NO}, // null
		// 	//{ 1.0f, lightBallPos, { 2.0, 2.0, 2.0 }, { 0.0f, 0.0f, 0.0f }, MAT_EMIT, MEDIUM_NO} // light
		// };
		// float numspheres = sizeof(spheres) / sizeof(Sphere);
		// for (int i = int(numspheres); i--;){  // for all spheres in scene
		// 	if ((hitSphereDist = spheres[i].intersect(Ray(rayorig, raydir)))  // keep track of distance from origin to closest intersection point
		// 	&& hitSphereDist < sceneT && hitSphereDist > RAY_MIN) { 
		// 		sceneT = hitSphereDist; 
		// 		hitSphereIdx = i; 
		// 		geomtype = GEO_SPHERE; 
		// 	}
		// }

    // int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

            // threadId == 460800
    // if (threadId > 300000 && threadId < 700000 && beta.x < 0.01f) {
    //     printf("BSSRDF: threadId = %d\n", threadId);
    //     printf("BSSRDF: entering kernel ......\n");
    //     printf("BSSRDF: previous normal = (%.5f, %.5f, %.5f), next normal = (%.5f, %.5f, %.5f)\n", ns.x, ns.y, ns.z, normalNext.x, normalNext.y, normalNext.z);
    //     printf("BSSRDF: d = (%.5f, %.5f, %.5f), d_direction = (%.5f, %.5f, %.5f)\n", d.x, d.y, d.z, d_direction.x, d_direction.y, d_direction.z);
    //     // printf("BSSRDF: sampleAxis = %d, sampledRadius = %.5f\n", sampleAxis, sampledRadius);
    //     printf("BSSRDF: sigmaT = (%.5f, %.5f, %.5f), rho = (%.5f, %.5f, %.5f)\n", sigmaT.x, sigmaT.y, sigmaT.z, rho.x, rho.y, rho.z);
    //     printf("BSSRDF: radius projection = (%.5f, %.5f, %.5f), radius = %.5f\n", radiusProjection[0], radiusProjection[1], radiusProjection[2], radius);
    //     printf("BSSRDF: axis channel pdf = (%.5f, %.5f, %.5f)\n", axisChannelPdf[0], axisChannelPdf[1], axisChannelPdf[2]);
    //     printf("BSSRDF: pdf = %.5f, Sr = (%.5f, %.5f, %.5f), outS = %.5f, beta = (%.5f, %.5f, %.5f)\n", 
    //     pdf, Sr.x, Sr.y, Sr.z, outS, beta.x, beta.y, beta.z);
    //     printf("\n\n");
    // }