#pragma once

#include <cuda_runtime.h>
#include "linear_math.h"
#include "mathDefine.h"
#include "bssrdfTable.h"

#define USE_SOE true

// --------------------------  Photon Beam Diffusion  ----------------------------

__device__ inline float EvalProfile(const BSSRDF& table, int rhoIndex, int radiusIndex) {
    return table.profile[rhoIndex * table.radiusNum + radiusIndex];
}

template <typename Predicate>
__device__ inline int FindInterval(int size, const Predicate &pred) {
    int first = 0, len = size;
    while (len > 0) {
        int half = len >> 1, middle = first + half;
        // Bisect range based on value of _pred_ at _middle_
        if (pred(middle)) {
            first = middle + 1;
            len -= half + 1;
        } else
            len = half;
    }
    return clamp(first - 1, 0, size - 2);
}

__device__ inline bool CatmullRomWeights(int size, const float *nodes, float x, int *offset, float *weights) {
    // Return _false_ if _x_ is out of bounds
    if (!(x >= nodes[0] && x <= nodes[size - 1])) return false;

    // Search for the interval _idx_ containing _x_
    int idx = FindInterval(size, [&] __device__ (int i) { return nodes[i] <= x; });
    *offset = idx - 1;
    float x0 = nodes[idx], x1 = nodes[idx + 1];

    // Compute the $t$ parameter and powers
    float t = (x - x0) / (x1 - x0), t2 = t * t, t3 = t2 * t;

    // Compute initial node weights $w_1$ and $w_2$
    weights[1] = 2 * t3 - 3 * t2 + 1;
    weights[2] = -2 * t3 + 3 * t2;

    // Compute first node weight $w_0$
    if (idx > 0) {
        float w0 = (t3 - 2 * t2 + t) * (x1 - x0) / (x1 - nodes[idx - 1]);
        weights[0] = -w0;
        weights[2] += w0;
    } else {
        float w0 = t3 - 2 * t2 + t;
        weights[0] = 0;
        weights[1] -= w0;
        weights[2] += w0;
    }

    // Compute last node weight $w_3$
    if (idx + 2 < size) {
        float w3 = (t3 - t2) * (x1 - x0) / (nodes[idx + 2] - x0);
        weights[1] -= w3;
        weights[3] = w3;
    } else {
        float w3 = t3 - t2;
        weights[1] -= w3;
        weights[2] += w3;
        weights[3] = 0;
    }
    return true;
}

__device__ inline float InvertCatmullRom(int n, const float *x, const float *values, float u) {
    // Stop when _u_ is out of bounds
    if (!(u > values[0]))
        return x[0];
    else if (!(u < values[n - 1]))
        return x[n - 1];

    // Map _u_ to a spline interval by inverting _values_
    int i = FindInterval(n, [&](int i) { return values[i] <= u; });

    // Look up $x_i$ and function values of spline segment _i_
    float x0 = x[i], x1 = x[i + 1];
    float f0 = values[i], f1 = values[i + 1];
    float width = x1 - x0;

    // Approximate derivatives using finite differences
    float d0, d1;
    if (i > 0)
        d0 = width * (f1 - values[i - 1]) / (x1 - x[i - 1]);
    else
        d0 = f1 - f0;
    if (i + 2 < n)
        d1 = width * (values[i + 2] - f0) / (x[i + 2] - x0);
    else
        d1 = f1 - f0;

    // Invert the spline interpolant using Newton-Bisection
    float a = 0, b = 1, t = .5f;
    float Fhat, fhat;
    while (true) {
        // Fall back to a bisection step when _t_ is out of bounds
        if (!(t > a && t < b)) t = 0.5f * (a + b);

        // Compute powers of _t_
        float t2 = t * t, t3 = t2 * t;

        // Set _Fhat_ using Equation (8.27)
        Fhat = (2 * t3 - 3 * t2 + 1) * f0 + (-2 * t3 + 3 * t2) * f1 +
               (t3 - 2 * t2 + t) * d0 + (t3 - t2) * d1;

        // Set _fhat_ using Equation (not present)
        fhat = (6 * t2 - 6 * t) * f0 + (-6 * t2 + 6 * t) * f1 +
               (3 * t2 - 4 * t + 1) * d0 + (3 * t2 - 2 * t) * d1;

        // Stop the iteration if converged
        if (std::abs(Fhat - u) < 1e-6f || b - a < 1e-6f) break;

        // Update bisection bounds using updated _t_
        if (Fhat - u < 0)
            a = t;
        else
            b = t;

        // Perform a Newton step
        t -= (Fhat - u) / fhat;
    }
    return x0 + t * width;
}

__device__ inline Vec3f SubsurfaceFromDiffuse(const BSSRDF& table, const Vec3f& rhoEff) {
    Vec3f rho;
    for (int c = 0; c < 3; ++c) {
        rho[c] = InvertCatmullRom(table.rhoNum, table.rho, table.rhoEff, rhoEff[c]);
    }
    return rho;
}

__device__ inline float SampleCatmullRom2D(int size1, int size2, 
    const float *nodes1, const float *nodes2, const float *values, const float *cdf, 
    float alpha, float u, float *fval = NULL, float *pdf = NULL) {
    // Determine offset and coefficients for the _alpha_ parameter
    int offset;
    float weights[4];
    if (!CatmullRomWeights(size1, nodes1, alpha, &offset, weights)) {
        return 0;
    }

    // Define a lambda function to interpolate table entries
    auto interpolate = [&] __device__ (const float *array, int idx) {
        float value = 0;
        for (int i = 0; i < 4; ++i)
            if (weights[i] != 0)
                value += array[(offset + i) * size2 + idx] * weights[i];
        return value;
    };

    // Map _u_ to a spline interval by inverting the interpolated _cdf_
    float maximum = interpolate(cdf, size2 - 1);
    u *= maximum;
    int idx = FindInterval(size2, [&] __device__ (int i) { return interpolate(cdf, i) <= u; });

    // Look up node positions and interpolated function values
    float f0 = interpolate(values, idx), f1 = interpolate(values, idx + 1);
    float x0 = nodes2[idx], x1 = nodes2[idx + 1];
    float width = x1 - x0;
    float d0, d1;

    // Re-scale _u_ using the interpolated _cdf_
    u = (u - interpolate(cdf, idx)) / width;

    // Approximate derivatives using finite differences of the interpolant
    if (idx > 0) {
        d0 = width * (f1 - interpolate(values, idx - 1)) / (x1 - nodes2[idx - 1]);
    } else {
        d0 = f1 - f0;
    }
    if (idx + 2 < size2) {
        d1 = width * (interpolate(values, idx + 2) - f0) / (nodes2[idx + 2] - x0);
    } else {
        d1 = f1 - f0;
    }

    // Invert definite integral over spline segment and return solution

    // Set initial guess for $t$ by importance sampling a linear interpolant
    float t;
    if (f0 != f1) {
        t = (f0 - sqrt(max((float)0, f0 * f0 + 2 * u * (f1 - f0)))) / (f0 - f1);
    } else {
        t = u / f0;
    }
    float a = 0, b = 1, Fhat, fhat;
    while (true) {
        // Fall back to a bisection step when _t_ is out of bounds
        if (!(t >= a && t <= b)) {
            t = 0.5f * (a + b);
        }

        // Evaluate target function and its derivative in Horner form
        Fhat = t * (f0 +
               t * (.5f * d0 +
               t * ((1.f / 3.f) * (-2 * d0 - d1) + f1 - f0 +
               t * (.25f * (d0 + d1) + .5f * (f0 - f1)))));

        fhat = f0 +
               t * (d0 +
               t * (-2 * d0 - d1 + 3 * (f1 - f0) +
               t * (d0 + d1 + 2 * (f0 - f1))));

        // Stop the iteration if converged
        if (abs(Fhat - u) < 1e-6f || b - a < 1e-6f) {
            break;
        }

        // Update bisection bounds using updated _t_
        if (Fhat - u < 0)
            a = t;
        else
            b = t;

        // Perform a Newton step
        t -= (Fhat - u) / fhat;
    }

    // Return the sample position and function value
    if (fval) *fval = fhat;
    if (pdf) *pdf = fhat / maximum;
    return x0 + width * t;
}

__device__ inline float sampleBSSRDFtable(const BSSRDF& table, float sigmaT, float rho, float u) {
    if (sigmaT == 0) return 0.0f;
    return SampleCatmullRom2D(table.rhoNum, table.radiusNum, table.rho, table.radius, table.profile, table.profileCDF, rho, u) / sigmaT;
}

__device__ inline float FM1(float eta) {
    float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta, eta5 = eta4 * eta;
    if (eta < 1)
        return 0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945 * eta3 + 2.49277f * eta4 - 0.68441f * eta5;
    else
        return -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 - 1.27198f * eta4 + 0.12746f * eta5;
}

__device__ inline float FrD(float cosThetaI, float etaI, float etaT) {
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

__device__ inline float FrSh(float F0, float cosTheta) {
    return F0 + (1.0f - F0) * pow5(1.0f - cosTheta);
}

template <typename T>
__device__ inline T paramSoE(const T& A) {
    // search light configuration
    T p = abs(A - T(0.8f));
    return T(1.85f) - A + 7.0f * p * p * p;

    // diffuse configuration
    // T p = A - T(0.8f);
    // return T(1.9f) - A + 3.5f * p * p;

    // search light configuration with dmfp to d
    // T p = A - T(0.33f); 
    // p = p * p;
    // return T(3.5f) + 100.0f * p * p;
}

__device__ inline void sampleBSSRDFprobeRay (
    float r1, float r2, float r3,
    Vec3f& normal, Vec3f& hitpoint,
    Vec3f& sigmaT, Vec3f& rho,
    Vec3f& probeRayOrig, Vec3f& probeRayDir, float& probeRayLength, 
    BSSRDF& table, Vec3f& vx, Vec3f& vy, float& radius) {

    // sample a channel (r g b)
    int ch = r1 * 3.0f;
    r1 = r1 * 3.0f - ch;

    // sample axis
    Vec3f probex, probey;
    if (r1 < 0.5f) {
        probeRayDir = normal;  probex = vx; probey = vy; r1 *= 2;
    } else if (r1 < 0.75f) {
        probeRayDir = vx;  probex = normal; probey = vy; r1 *= (r1 - 0.5f) * 4;
    } else {
        probeRayDir = vy;  probex = vx; probey = normal; r1 *= (r1 - 0.75f) * 4;
    }

    // sample radius
    #if USE_SOE
    float s = paramSoE(rho[ch]);
    radius = - logf(1.0f - r2 * 0.99f) / sigmaT[ch] / s;
    float radiusMax =  - logf(0.01f) / sigmaT[ch] / s;
    if (r1 < 0.5f) {
        radius *= 3.0f;
        radiusMax *= 3.0f;
    }
    #else
    radius = sampleBSSRDFtable(table, sigmaT[ch], rho[ch], r2 * 0.99f);
    float radiusMax = sampleBSSRDFtable(table, sigmaT[ch], rho[ch], 0.99f);
    #endif
    float phi = TWO_PI * r3;

    // probe ray
    probeRayLength = 2.0f * sqrtf(radiusMax * radiusMax - radius * radius);
    probeRayOrig = hitpoint + radius * (probex * cosf(phi) + probey * sinf(phi)) - probeRayLength * probeRayDir * 0.5f;
}

__device__ inline void calculateBSSRDF( 
        const Vec3f& ns, const Vec3f& normalNext,
        const Vec3f& sigmaT, const Vec3f& rho,
        Vec3f& beta, const BSSRDF& table, const Vec3f& d,
        const Vec3f& ss, const Vec3f& ts)
{
    // real radius
    float radius = d.length();

    // three sampled radius
    Vec3f dLocal (dot(ss, d), dot(ts, d), dot(ns, d));
    dLocal *= dLocal;
    Vec3f radiusProjection { 
        sqrtf(dLocal.y + dLocal.z), 
        sqrtf(dLocal.z + dLocal.x), 
        sqrtf(dLocal.x + dLocal.y) 
    };

    // three direction pdf
    Vec3f axisChannelPdf {
        abs(dot(ss, normalNext)) * 0.08333333333f,  // 0.25 / 3
        abs(dot(ts, normalNext)) * 0.08333333333f,  // 0.25 / 3
        abs(dot(ns, normalNext)) * 0.16666666666f   // 0.5 / 3
    };

    // pre-calculate
    #if USE_SOE 
    Vec3f s = paramSoE(rho);
    #else
    Vec3f sigmaT2 = sigmaT * sigmaT;
    #endif

    // pdf
    float pdf = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        #if USE_SOE 
        Vec3f axisPdf = (exp3f(- s * radiusProjection[axis] * sigmaT) 
                       + exp3f(- s * radiusProjection[axis] * sigmaT / 3.0f) / 3.0f) / FOUR_PI * rho * s * sigmaT;
        if (radiusProjection[axis] > 1e-4f) {
            axisPdf /= radiusProjection[axis];
        }
        pdf += (axisPdf.x + axisPdf.y + axisPdf.z) * axisChannelPdf[axis];
        #else
        float axisPdf = 0.0f;
        for (int ch = 0; ch < 3; ++ch) {
            // channel pdf
            float channelPdf = 0.0f;
            float rOptical = radiusProjection[axis] * sigmaT[ch];
    
            // get r Sr(r, rho) and rhoEff(rho)
            int rhoOffset, radiusOffset;
            float rhoWeights[4], radiusWeights[4];
            if (!CatmullRomWeights(table.rhoNum,    table.rho,    rho[ch],  &rhoOffset,    rhoWeights    ) ||
                !CatmullRomWeights(table.radiusNum, table.radius, rOptical, &radiusOffset, radiusWeights ))
                break;
    
            float sr = 0.0f;
            float rhoEff = 0.0f;
            for (int i = 0; i < 4; ++i) {
                rhoEff += table.rhoEff[rhoOffset + i] * rhoWeights[i];
                for (int j = 0; j < 4; ++j) {
                    sr += EvalProfile(table, rhoOffset + i, radiusOffset + j) * rhoWeights[i] * radiusWeights[j];
                }
            }

            // scale back, / rhoEff to 1 Dimension
            channelPdf = sr * sigmaT2[ch] / rhoEff;

            // remove radius
            if (rOptical > 1e-4f) {
                channelPdf /= rOptical;
            }

            // add
            axisPdf += max(0.0f, channelPdf);
        }
        // add
        pdf += axisPdf * axisChannelPdf[axis];
        #endif    
    }

    // Sr(r, rho)
    #if USE_SOE
    Vec3f Sr = (exp3f(- s * radius * sigmaT) + exp3f(- s * radius * sigmaT / 3.0f)) / EIGHT_PI * rho * s * sigmaT;
    if (radius > 1e-4f) {
        Sr /= radius;
    }
    #else
    Vec3f Sr (0.0f, 0.0f, 0.0f);
    for (int ch = 0; ch < 3; ++ch) {
        // get r Sr(r, rho)
        float rOptical = radius * sigmaT[ch];
        int rhoOffset, radiusOffset;
        float rhoWeights[4], radiusWeights[4];
        if (!CatmullRomWeights(table.rhoNum,    table.rho,    rho[ch],  &rhoOffset,    rhoWeights    ) ||
            !CatmullRomWeights(table.radiusNum, table.radius, rOptical, &radiusOffset, radiusWeights ))
            continue;

        float sr = 0.0f;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                sr += EvalProfile(table, rhoOffset + i, radiusOffset + j) * rhoWeights[i] * radiusWeights[j];
            }
        }

        // remove radius
        if (rOptical > 1e-4f) {
            sr /= rOptical;
        }

        // scale back
        Sr[ch] = max(0.0f, sr * sigmaT2[ch]);
    }
    #endif
    
    // sanity output limitation
    beta = min3f(Sr / pdf, Vec3f(10.0f));
}