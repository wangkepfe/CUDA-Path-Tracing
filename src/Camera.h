// code for depth-of-field, mouse + keyboard user interaction based on https://github.com/peterkutz/GPUPathTracer

#pragma once

#include "linear_math.h"
#include "mathDefine.h"
#include <iostream>

// Camera struct, used to store interactive camera data, copied to the GPU and used by CUDA for each frame
struct Camera {
	Vec2f resolution;
	Vec3f position;
	Vec3f view;
	Vec3f up;
	Vec2f fov;
	float apertureRadius;
	float focalDistance;
	
	float envMapRotation;
};

// class for interactive camera object, updated on the CPU for each frame and copied into Camera struct
class InteractiveCamera
{
public:

	Vec3f centerPosition;
	Vec3f viewDirection;
	float yaw;
	float pitch;
	float radius;
	float apertureRadius;
	float focalDistance;

	void fixYaw();
	void fixPitch();
	void fixRadius();
	void fixApertureRadius();
	void fixFocalDistance();

	InteractiveCamera();
	~InteractiveCamera();
	void changeYaw(float m);
	void changePitch(float m);
	void changeRadius(float m);
	void changeAltitude(float m);
	void changeFocalDistance(float m);
	void strafe(float m);
	void goForward(float m);
	void rotateRight(float m);
	void changeApertureDiameter(float m);
	void setResolution(float x, float y);
	void setFOVX(float fovx);

	void buildRenderCamera(Camera* renderCamera);

	void saveToFile(const std::string &camFileName);
	void loadFromFile(const std::string &camFileName);

	float envMapRotation;

	Vec2f resolution;
	Vec2f fov;
};
