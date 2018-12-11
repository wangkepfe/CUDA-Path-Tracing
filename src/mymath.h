#pragma once 

#include <math.h>
#include <cuda_runtime.h>

#include "linear_math.h"

inline __host__ __device__ void Barycentric2D(Vec2f p, Vec2f a, Vec2f b, Vec2f c, float &u, float &v, float &w)
{
    Vec2f v0 = b - a, v1 = c - a, v2 = p - a;
    float den = v0.x * v1.y - v1.x * v0.y;
    v = (v2.x * v1.y - v1.x * v2.y) / den;
    w = (v0.x * v2.y - v2.x * v0.y) / den;
    u = 1.0f - v - w;
}

inline __host__ __device__ void Barycentric(const Vec3f& p, const Vec3f& a, const Vec3f& b, const Vec3f& c, float &u, float &v, float &w)
{
    Vec3f v0 = b - a;
    Vec3f v1 = c - a;
    Vec3f v2 = p - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;
}