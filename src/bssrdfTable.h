#pragma once

// bssrdf table for gpu
struct BSSRDF {
	const int rhoNum;
	const int radiusNum;
	const float *rho;
	const float *radius;
	const float *profile;
	const float *profileCDF;
	const float *rhoEff;
};