#include "bssrdf.h"

// void SubsurfaceFromDiffuse(const BSSRDFTable &t, const Vec3f &rhoEff, const Vec3f &mfp, Vec3f *sigma_a, Vec3f *sigma_s) {
//     for (int c = 0; c < Spectrum::nSamples; ++c) {
//         Float rho = InvertCatmullRom(t.nRhoSamples, t.rhoSamples.get(),
//                     t.rhoEff.get(), rhoEff[c]);
//         (*sigma_s)[c] = rho / mfp[c];
//         (*sigma_a)[c] = (1 - rho) / mfp[c];
//     }
// }