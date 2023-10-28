#pragma once
#include <cuda.h>
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <iostream>
#include <cusolverDn.h> 
#include <cublas_v2.h>
#include "randomGenerate.h"

extern "C" __declspec(dllexport) void diffusionRandom_cu3dRPY(double* position, const double* EM_c, int size, 
    const float* efield, int ex, int ey,
    int maxstep, double region, double regionxyz, 
    double D0, double radius_max, double radius_diff, 
    double ke, double kv, double debye, double*, double*);

