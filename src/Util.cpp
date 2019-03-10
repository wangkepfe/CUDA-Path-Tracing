#include "Util.h"

// this hash function calculates a new random number generator seed for each frame, based on framenumber
unsigned int WangHash(unsigned int a)
{
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

void writeToPPM(const char* fname, int width, int height, Vec3f* accuBuffer, unsigned int frameNum) {
    FILE *f = fopen(fname, "w");          
    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    for (int i = 0; i < width * height; i++) {
		accuBuffer[i] /= static_cast<float>(frameNum);
        fprintf(f, "%d %d %d ", pixelToInt(accuBuffer[i].x), pixelToInt(accuBuffer[i].y), pixelToInt(accuBuffer[i].z));
	}
    fclose(f);
    printf("Successfully wrote result image to %s\n", fname);
}
