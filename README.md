# features
### Specular reflection (Mirror)
<img src="renderingResult/specularReflection.PNG">
### Lambertian (Matte)
<img src="renderingResult/lambertian.PNG">
### Microfacet reflection (Plastic)
<img src="renderingResult/microfacetRough.PNG">
### Microfacet reflection (Metal)
<img src="renderingResult/microfacetReflection.PNG">
### Microfacet reflection (Brushed metal)
<img src="renderingResult/microfacetAnisotropic01.PNG">
### Microfacet reflection (Brushed metal)
<img src="renderingResult/microfacetAnisotropic02.PNG">
### Fresnel blend reflection (Substrate/Coated metal)
<img src="renderingResult/fresnelBlend.PNG">
### Specular transmission (Glass)
<img src="renderingResult/specularGlass.PNG">
### Microfacet transmission (Frosted glass)
<img src="renderingResult/roughGlass.PNG">
### Medium (Smoke)
<img src="renderingResult/mediumSmoke.PNG">
### Medium (Tea)
<img src="renderingResult/mediumTea.PNG">
### Medium (Milk)
<img src="renderingResult/mediumMilk.PNG">
### Medium (Jade)
<img src="renderingResult/mediumJade.PNG">

# external dependancy
- cuda
- glut
- glew

# reference

framework based on Matching Socks CUDA path tracer by Samuel Lapere, 2016 https://raytracey.blogspot.com

bvh based on the GPU ray tracing framework of Timo Aila, Samuli Laine and Tero Karras (Nvidia Research)

based on Source code for original framework: 
- https://code.google.com/archive/p/understanding-the-efficiency-of-ray-traversal-on-gpus/
- https://research.nvidia.com/publication/understanding-efficiency-ray-traversal-gpus-kepler-and-fermi-addendum
- https://mediatech.aalto.fi/~timo/HPG2009/

anti-aliasing, depth of field based on https://github.com/peterkutz/GPUPathTracer

material modeling, surface/media based on "Physically Based Rendering: From Theory To Implementation", by Matt Pharr, Wenzel Jakob, and Greg Humphreys https://github.com/mmp/pbrt-v3, http://pbrt.org



