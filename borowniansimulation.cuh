#pragma once


#include <cuda.h>
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <iostream>

struct CuVector2F {
    float x;
    float y;
};
extern "C" __declspec(dllexport) void diffusionRandom_cu(float* data, const float* dx, int number, float region, float maxregion, int* maxstep_p, int simulate_round);
extern "C" __declspec(dllexport) void diffusionRandom_cu2d(float* x, float* y, const float* dx, const float* EM, int size, const float* efield, int ex, int ey,
	int* maxstep_p, int simulate_round, float regionx, float regiony, float region0);


