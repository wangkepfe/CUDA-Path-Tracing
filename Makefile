# -----------------------------------------------
GLEW_INC=external/glew/include
GLUT_INC=external/freeglut/include
INC_FLAGS=--include-path ${GLEW_INC},${GLUT_INC}
# -----------------------------------------------
GLEW_LIB=external/glew/lib/Release/x64/glew32
GLUT_LIB=external/freeglut/lib/x64/freeglut
LIB_FLAGS=--library ${GLEW_LIB},${GLUT_LIB}
# -----------------------------------------------
EXE=raytracer
OUT_FLAGS=-o ${EXE}
# -----------------------------------------------
SRC_DIR=src
SRC=$(wildcard ${SRC_DIR}/*.cpp ${SRC_DIR}/*.cc ${SRC_DIR}/*.cu)
# -----------------------------------------------
COMPILER=nvcc
FLAGS=${OUT_FLAGS} ${INC_FLAGS} ${LIB_FLAGS}
# -----------------------------------------------
all: ${EXE}

${EXE}: ${SRC}
	${COMPILER} ${SRC} ${FLAGS}

test: ${EXE}
	${EXE}