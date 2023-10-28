#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>
#include <cmath>
#include <random>
#include <Eigen/Core> 
#include <Eigen/SVD>
#include <Eigen/Dense>

namespace  e = Eigen;
extern "C" __declspec(dllexport) void blockCurrent(const double* RM0_array,
    const double* EFM_in, const double* Dipole_in,
    const double* Mob_in, const double* SQM_in,
    const double* E_in, const double* dE_in, const double* Res_in, const double* diffusion_reduce, const double* electrophoretic_reduce,
    const int* skip, int col, int row, double* const data_ptr);

extern "C" __declspec(dllexport) void rotationMatrix(const double* RM0_array,
    const double* EFM_in, const double* Dipole_in,
    const double* Mob_in, const double* SQM_in,
    const double* E_in, const double* dE_in, const double* diffusion_reduce, const double* electrophoretic_reduce,
    const int* skip, int col, int row, double* const data_ptr);
