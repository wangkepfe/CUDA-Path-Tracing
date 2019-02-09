#pragma once

#include <cuda_runtime.h>
#include "linear_math.h"
#include "mathDefine.h"

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
    return Clamp(first - 1, 0, size - 2);
}

__device__ inline bool CatmullRomWeights(int size, float *nodes, float x, int *offset, float *weights) {
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

__device__ inline float SampleCatmullRom2D(int size1, int size2, 
    float *nodes1, float *nodes2, float *values, float *cdf, 
    float alpha, float u, float *fval, float *pdf) {
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

__device__ inline float sampleBSSRDFtable(float sigmaT, float rho, float u) {
    if (sigmaT == 0) return -1;
    return SampleCatmullRom2D(table.nRhoSamples, table.nRadiusSamples, 
        table.rhoSamples.get(), table.radiusSamples.get(), table.profile.get(), table.profileCDF.get(), 
        rho, u) / sigmaT;
}

__device__ inline void sampleBSSRDFprobeRay (
    float r1, float r2, float r3,
    Vec3f& normal, Vec3f& hitpoint,
    Vec3f sigmaT, Vec3f rho
    Vec3f& probeRayOrig, Vec3f& vz, float& probeRayLength
) {
    // probe ray direction
    vz = normal;

    // localize coord sys
    vz.normalize();
    Vec3f w;
    if (abs(vz.x) < SQRT_OF_ONE_THIRD) { 
		w = Vec3f(1, 0, 0);
	} else if (abs(vz.y) < SQRT_OF_ONE_THIRD) { 
		w = Vec3f(0, 1, 0);
	} else {
		w = Vec3f(0, 0, 1);
    }
    Vec3f vx = cross(vz, w).normalize();
    Vec3f vy = cross(vz, vx);

    // sample a channel (r g b)
    int ch = r1 * 3.0f;

    // sample radius
    float radius = sampleBSSRDFtable(sigmaT[ch], rho[ch], r2);
    float phi = 2 * Pi * r3;

    // radiusMax
    float radiusMax = sampleBSSRDFtable(sigmaT[ch], rho[ch], 0.999f);

    // probe ray
    probeRayOrig = hitpoint + radius * (vx * cos(phi) + vy * sin(phi)) - l * vz * 0.5f;
    probeRayLength = 2 * sqrt(radiusMax * radiusMax - radius * radius);
}