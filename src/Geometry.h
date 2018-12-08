#pragma once

#include "linear_math.h"

struct Vertex
{
	Vec3f p;
	Vec3f n;
	Vec2f uv;
};

struct Triangle {
	// indexes in vertices array
	unsigned _idx1;
	unsigned _idx2;
	unsigned _idx3;
};

