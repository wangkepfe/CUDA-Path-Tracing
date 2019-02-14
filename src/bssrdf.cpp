#include "bssrdf.h"

#define NUM_BEAM_SAMPLES 100

using namespace std;

BssrdfTable::BssrdfTable(int nRhoSamples, int nRadiusSamples)
    : nRhoSamples(nRhoSamples),
      nRadiusSamples(nRadiusSamples),
      rhoSamples(new float[nRhoSamples]),
      radiusSamples(new float[nRadiusSamples]),
      profile(new float[nRadiusSamples * nRhoSamples]),
      rhoEff(new float[nRhoSamples]),
      profileCDF(new float[nRadiusSamples * nRhoSamples]) {}

float FresnelMoment1(float eta) {
    float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta, eta5 = eta4 * eta;
    if (eta < 1)
        return 0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945 * eta3 + 2.49277f * eta4 - 0.68441f * eta5;
    else
        return -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 - 1.27198f * eta4 + 0.12746f * eta5;
}

float FresnelMoment2(float eta) {
    float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta, eta5 = eta4 * eta;
    if (eta < 1) {
        return 0.27614f - 0.87350f * eta + 1.12077f * eta2 - 0.65095f * eta3 + 0.07883f * eta4 + 0.04860f * eta5;
    } else {
        float r_eta = 1 / eta, r_eta2 = r_eta * r_eta, r_eta3 = r_eta2 * r_eta;
        return -547.033f + 45.3087f * r_eta3 - 218.725f * r_eta2 + 458.843f * r_eta + 404.557f * eta - 189.519f * eta2 + 54.9327f * eta3 - 9.00603f * eta4 + 0.63942f * eta5;
    }
}

float BeamDiffusionMS(float sigma_s, float sigma_a, float g, float eta, float r) {
    const int nSamples = NUM_BEAM_SAMPLES;

    // reduced scattering coefficients
    float sigmap_s = sigma_s * (1 - g);
    float sigmap_t = sigma_a + sigmap_s;
    float rhop = sigmap_s / sigmap_t;

    // non-classical diffusion coefficient
    float D_g = (2 * sigma_a + sigmap_s) / (3 * sigmap_t * sigmap_t);

    // effective transport coefficient
    float sigma_tr = sqrt(sigma_a / D_g);

    // linear extrapolation distance
    float fm1 = FresnelMoment1(eta);
    float fm2 = FresnelMoment2(eta);
    float ze = -2 * D_g * (1 + 3 * fm2) / (1 - 2 * fm1);

    // exitance scale factors
    float cPhi = 0.25f * (1 - 2 * fm1);
    float cE = 0.5f * (1 - 3 * fm2);

    float Ed = 0;
    for (int i = 0; i < nSamples; ++i) {
        // Sample real point source depth
        float zr = -log(1 - (i + 0.5f) / nSamples) / sigmap_t;

        float zv = -zr + 2 * ze;
        float dr = sqrt(r * r + zr * zr);
        float dv = sqrt(r * r + zv * zv);

        // Compute dipole fluence rate
        float phiD = Inv4Pi / D_g * (exp(-sigma_tr * dr) / dr - exp(-sigma_tr * dv) / dv);

        // Compute dipole vector irradiance
        float EDn = Inv4Pi * (zr * (1 + sigma_tr * dr) * exp(-sigma_tr * dr) / (dr * dr * dr) -
                              zv * (1 + sigma_tr * dv) * exp(-sigma_tr * dv) / (dv * dv * dv));

        // Add contribution
        float E1 = phiD * cPhi + EDn * cE;
        float kappa = 1 - exp(-2 * sigmap_t * (dr + zr));
        Ed += kappa * rhop * rhop * E1;
    }
    return Ed / nSamples;
}

inline float PhaseHG(float cosTheta, float g) {
    float denom = 1 + g * g + 2 * g * cosTheta;
    return Inv4Pi * (1 - g * g) / (denom * sqrt(denom));
}

inline float clamp(float n, float lower, float upper) {
    return fmax(lower, fmin(n, upper));
}

float FrDielectric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = clamp(cosThetaI, -1, 1);
    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        swap(etaI, etaT);
        cosThetaI = abs(cosThetaI);
    }

    // Compute _cosThetaT_ using Snell's law
    float sinThetaI = sqrt(fmax((float)0, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    if (sinThetaT >= 1) return 1;
    float cosThetaT = sqrt(fmax((float)0, 1 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                  ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                  ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2;
}

float BeamDiffusionSS(float sigma_s, float sigma_a, float g, float eta, float r) {
    
    // material parameters
    float sigma_t = sigma_a + sigma_s;
    float rho = sigma_s / sigma_t;

    // minimum t below the critical angle
    float tCrit = r * sqrt(eta * eta - 1);

    float Ess = 0;
    const int nSamples = NUM_BEAM_SAMPLES;

    for (int i = 0; i < nSamples; ++i) {

        // single scattering integrand
        float ti = tCrit - log(1 - (i + .5f) / nSamples) / sigma_t;

        // Determine length d of connecting segment and cosThetaO
        float d = sqrt(r * r + ti * ti);
        float cosThetaO = ti / d;

        // Add contributions
        Ess += rho * exp(-sigma_t * (d + tCrit)) / (d * d) * PhaseHG(cosThetaO, g) * (1 - FrDielectric(-cosThetaO, 1, eta)) * abs(cosThetaO);
    }

    return Ess / nSamples;
}

float IntegrateCatmullRom(int n, const float *x, const float *values, float *cdf) {
    float sum = 0;
    cdf[0] = 0;
    for (int i = 0; i < n - 1; ++i) {
        
        float x0 = x[i], x1 = x[i + 1];
        float f0 = values[i], f1 = values[i + 1];
        float width = x1 - x0;

        float d0, d1;
        if (i > 0)
            d0 = width * (f1 - values[i - 1]) / (x1 - x[i - 1]);
        else
            d0 = f1 - f0;
        if (i + 2 < n)
            d1 = width * (values[i + 2] - f0) / (x[i + 2] - x0);
        else
            d1 = f1 - f0;

        sum += ((d0 - d1) * (1.f / 12.f) + (f0 + f1) * .5f) * width;
        cdf[i + 1] = sum;
    }
    return sum;
}

void ComputeBeamDiffusionBSSRDF(float g, float eta, BssrdfTable *t) {

    // radius values
    t->radiusSamples[0] = 0;
    t->radiusSamples[1] = 2.5e-3f; // first step 0.0025
    for (int i = 2; i < t->nRadiusSamples; ++i) {
        t->radiusSamples[i] = t->radiusSamples[i - 1] * 1.2f; // each step
    }

    // albedo values
    for (int i = 0; i < t->nRhoSamples; ++i) {
        t->rhoSamples[i] = (1 - exp(-8 * i / (float)(t->nRhoSamples - 1))) / (1 - exp(-8));
    }

    // diffusion profile
    for (int i = 0; i < t->nRhoSamples; ++i) {

        // scattering profile
        for (int j = 0; j < t->nRadiusSamples; ++j) {
            float rho = t->rhoSamples[i];
            float r = t->radiusSamples[j];

            t->profile[i * t->nRadiusSamples + j] = 2 * Pi * r * (BeamDiffusionSS(rho, 1 - rho, g, eta, r) + BeamDiffusionMS(rho, 1 - rho, g, eta, r));
        }

        // Compute effective albedo and CDF for importance sampling
        t->rhoEff[i] = IntegrateCatmullRom(t->nRadiusSamples, t->radiusSamples.get(), &t->profile[i * t->nRadiusSamples], &t->profileCDF[i * t->nRadiusSamples]);
    }
}