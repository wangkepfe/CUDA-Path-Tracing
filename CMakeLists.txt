cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8)

project(raytracer)

# find cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# find glew and freeglut
include_directories(external/glew/include)
include_directories(external/freeglut/include)
find_library(glew_library     NAMES glew32   PATHS ${CMAKE_SOURCE_DIR}/external/glew/lib/Release/x64)
find_library(freeglut_library NAMES freeglut PATHS ${CMAKE_SOURCE_DIR}/external/freeglut/lib/x64)

# set compiler
SET(CMAKE_C_COMPILER nvcc)
SET(CMAKE_CXX_COMPILER nvcc)

# glob source
file(GLOB SOURCE src/*.cpp src/*.cu src/*.h src/*.cuh)

# target
add_executable(raytracer ${SOURCE})

# link lib
target_link_libraries(raytracer ${CUDA_LIBRARIES} ${glew_library} ${freeglut_library})