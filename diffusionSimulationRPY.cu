#ifndef __CUDACC__  
#define __CUDACC__
#endif
#include "pch.h"
#include "diffusionSimulationRPY.cuh" 
#include "errorHandle.cuh"

double AH = 1.33e-20;
double pi = 3.14159265359;
texture<float2, cudaTextureType2D, cudaReadModeElementType> efieldtex;

__global__ void kernel_DLVO_tensor(double* x, double* y, double* z,
	double* FDLVO, double* radius, int size, double debye, double ke, double kv) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		int i = idx * 9 * size;
		double xp = x[idx];
		double yp = y[idx];
		double zp = z[idx];
		double radiusp = radius[idx];
		double R = 0;
		double h2 = 0;
		double h = 0;
		double Rij2 = 0;
		double Rij = 0;
		double Fp = 0;
		int jdx = idx;
		double Fw = (abs(zp) > radiusp + 0.1) ? -2.0 * kv / (zp * zp) + -2.0 * ke * exp(-1.0 / debye * zp) : 0;
		while (jdx < size) {
			R = sqrt(radiusp * radius[jdx]);
			Rij2 = (xp - x[jdx]) * (xp - x[jdx]) + (yp - y[jdx]) * (yp - y[jdx]) + (zp - z[jdx]) * (zp - z[jdx]);
			Rij = sqrt(Rij2);
			h = Rij - radiusp - radius[jdx]; 
			h2 = h * h;
			i = idx * 3;
			if (idx == jdx) {
				FDLVO[i] = 0;
				FDLVO[i + 1] = 0;
				FDLVO[i + 2] = Fw;
			}
			else if (h > 0.1){
				Fp = ke * exp(-1.0 / debye * h) / Rij;
				FDLVO[i] += kv * (x[jdx] - xp) / Rij / h2 + Fp * (x[jdx] - xp);
				FDLVO[i+1] += kv * (y[jdx] - yp) / Rij / h2 + Fp * (y[jdx] - yp);
				FDLVO[i+2] += kv * (z[jdx] - zp) / Rij / h2 + Fp * (z[jdx] - zp);
			}
			jdx++;
		}
		jdx = 0;
		while (jdx < idx) {
			R = sqrt(radiusp * radius[jdx]);
			Rij2 = (xp - x[jdx]) * (xp - x[jdx]) + (yp - y[jdx]) * (yp - y[jdx]) + (zp - z[jdx]) * (zp - z[jdx]);
			Rij = sqrt(Rij2);
			h = Rij - radiusp - radius[jdx];
			h2 = h * h;
			i = idx * 3;
			if(h > 0.1) {
				Fp = ke * exp(-1.0 / debye * h) / Rij;
				FDLVO[i] += kv * (x[jdx] - xp) / Rij / h2 + Fp * (x[jdx] - xp);
				FDLVO[i + 1] += kv * (y[jdx] - yp) / Rij / h2 + Fp * (y[jdx] - yp);
				FDLVO[i + 2] += kv * (z[jdx] - zp) / Rij / h2 + Fp * (z[jdx] - zp);
			}
			jdx++;
		}
	}
}

__global__ void kernel_HI_tensor(double* x, double* y, double* z, 
	double* M, double* radius, int size, double radius_max, double radius_diff) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		int i = idx * 9 * size;
		double xp = x[idx];
		double yp = y[idx];
		double zp = z[idx];
		double radiusp = radius[idx];
		double Rij = 0;
		double Rij2 = 0;
		double Rij3 = 0;
		double aiaj = 0;
		int jdx = idx;
		int t1 = 0;
		int t2 = 0;
		while (jdx < size) {
			Rij2 = (xp - x[jdx]) * (xp - x[jdx]) + (yp - y[jdx]) * (yp - y[jdx]) + (zp - z[jdx]) * (zp - z[jdx]);
			Rij = sqrt(Rij2);
			i = jdx * 3 + idx * 9 * size;
			if (Rij > (radiusp + radius[jdx])) {
				t2 = 1.0 - (radiusp * radiusp + radius[jdx] * radius[jdx]) / Rij2;
				t1 = (1.0 - t2) / 3.0 + 1.0;
				t2 /= Rij2;
				M[i] = 3.0 / 4.0 / Rij * (t1 + t2 * (xp - x[jdx]) * (xp - x[jdx]));
				M[i + 1] = 3.0 / 4.0 / Rij * t2 * (xp - x[jdx]) * (yp - y[jdx]);
				M[i + 2] = 3.0 / 4.0 / Rij * t2 * (xp - x[jdx]) * (zp - z[jdx]);
				M[i + 3 * size] = M[i + 1];
				M[i + 3 * size + 1] = 3.0 / 4.0 / Rij * (t1 + t2 * (yp - y[jdx]) * (yp - y[jdx]));
				M[i + 3 * size + 2] = 3.0 / 4.0 / Rij * t2 * (zp - z[jdx]) * (yp - y[jdx]);
				M[i + 6 * size] = M[i + 2];
				M[i + 6 * size + 1] = M[i + 3 * size + 2];
				M[i + 6 * size + 2] = 3.0 / 4.0 / Rij * (t1 + t2 * (zp - z[jdx]) * (zp - z[jdx]));
			}
			else if (radius_diff >= Rij) {
				M[i] = 1.0 / radius_max;
				M[i + 1] = 0;
				M[i + 2] = 0;
				M[i + 3 * size] = 0;
				M[i + 3 * size + 1] = M[i];
				M[i + 3 * size + 2] = 0;
				M[i + 6 * size] = 0;
				M[i + 6 * size + 1] = 0;
				M[i + 6 * size + 2] = M[i];
			}
			else {
				Rij3 = pow(Rij2, 1.5);
				aiaj = radiusp * radius[jdx];
				t2 = 3.0 * pow(((radiusp - radius[jdx]) * (radiusp - radius[jdx]) - Rij2), 2) / 32.0 / Rij3;
				t1 = (16.0 * Rij3 * (radiusp + radius[jdx]) - pow(((radiusp - radius[jdx]) * (radiusp - radius[jdx]) + 3.0 * Rij2), 2)) / 32.0 / Rij3;
				t2 /= Rij2;
				M[i] = 1.0 / aiaj * (t1 + t2 * (xp - x[jdx]) * (xp - x[jdx]));
				M[i + 1] = 1.0 / aiaj * t2 * (xp - x[jdx]) * (yp - y[jdx]);
				M[i + 2] = 1.0 / aiaj * t2 * (xp - x[jdx]) * (zp - z[jdx]);
				M[i + size] = M[i + 1];
				M[i + size + 1] = 1.0 / aiaj * (t1 + t2 * (yp - y[jdx]) * (yp - y[jdx]));
				M[i + size + 2] = 1.0 / aiaj * t2 * (zp - z[jdx]) * (yp - y[jdx]);
				M[i + 2 * size] = M[i + 2];
				M[i + 2 * size + 1] = M[i + 3 * size + 2];
				M[i + 2 * size + 2] = 1.0 / aiaj * (t1 + t2 * (zp - z[jdx]) * (zp - z[jdx]));
			}
			jdx++;
		}
		jdx = 0;
		while (jdx < idx) {
			Rij2 = (xp - x[jdx]) * (xp - x[jdx]) + (yp - y[jdx]) * (yp - y[jdx]) + (zp - z[jdx]) * (zp - z[jdx]);
			Rij = sqrt(Rij2);
			i = jdx * 3 + idx * 9 * size;
			if (Rij > (radiusp + radius[jdx]) * (radiusp + radius[jdx])) {
				t2 = 1.0 - (radiusp * radiusp + radius[jdx] * radius[jdx]) / Rij2;
				t1 = (1.0 - t2) / 3.0 + 1.0;
				M[i] = 3.0 / 4.0 / Rij * (t1 + t2 * (xp - x[jdx]) * (xp - x[jdx]));
				M[i + 1] = 3.0 / 4.0 / Rij * t2 * (xp - x[jdx]) * (yp - y[jdx]);
				M[i + 2] = 3.0 / 4.0 / Rij * t2 * (xp - x[jdx]) * (zp - z[jdx]);
				M[i + 3 * size] = M[i + 1];
				M[i + 3 * size + 1] = 3.0 / 4.0 / Rij * (t1 + t2 * (yp - y[jdx]) * (yp - y[jdx]));
				M[i + 3 * size + 2] = 3.0 / 4.0 / Rij * t2 * (zp - z[jdx]) * (yp - y[jdx]);
				M[i + 6 * size] = M[i + 2];
				M[i + 6 * size + 1] = M[i + 3 * size + 2];
				M[i + 6 * size + 2] = 3.0 / 4.0 / Rij * (t1 + t2 * (zp - z[jdx]) * (zp - z[jdx]));
			}
			else if (radius_diff >= Rij) {
				M[i] = 1.0 / radius_max;
				M[i + 1] = 0;
				M[i + 2] = 0;
				M[i + 3 * size] = 0;
				M[i + 3 * size + 1] = M[i];
				M[i + 3 * size + 2] = 0;
				M[i + 6 * size] = 0;
				M[i + 6 * size + 1] = 0;
				M[i + 6 * size + 2] = M[i];
			}
			else {
				Rij3 = pow(Rij2, 1.5);
				aiaj = radiusp * radius[jdx];
				t2 = 3.0 * pow(((radiusp - radius[jdx]) * (radiusp - radius[jdx]) - Rij2), 2) / 32.0 / Rij3;
				t1 = (16.0 * Rij3 * (radiusp + radius[jdx]) - pow(((radiusp - radius[jdx]) * (radiusp - radius[jdx]) + 3.0 * Rij2), 2)) / 32.0 / Rij3;
				M[i] = 1.0 / aiaj * (t1 + t2 * (xp - x[jdx]) * (xp - x[jdx]));
				M[i + 1] = 1.0 / aiaj * t2 * (xp - x[jdx]) * (yp - y[jdx]);
				M[i + 2] = 1.0 / aiaj * t2 * (xp - x[jdx]) * (zp - z[jdx]);
				M[i + 3 * size] = M[i + 1];
				M[i + 3 * size + 1] = 1.0 / aiaj * (t1 + t2 * (yp - y[jdx]) * (yp - y[jdx]));
				M[i + 3 * size + 2] = 1.0 / aiaj * t2 * (zp - z[jdx]) * (yp - y[jdx]);
				M[i + 6 * size] = M[i + 2];
				M[i + 6 * size + 1] = M[i + 3 * size + 2];
				M[i + 6 * size + 2] = 1.0 / aiaj * (t1 + t2 * (zp - z[jdx]) * (zp - z[jdx]));
			}
			jdx++;
		}
		
	}
};

__global__ void kernel_update(double* x, double* y, double* z, double* EM, int size, double* dx, double* FDLVO,
	double region, double regionxyz, double D0) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		double sqrtxy = sqrt(abs(x[idx] * y[idx]));
		float2 efield = tex2D<float2>(efieldtex, sqrtxy, z[idx]);
		int i = 3 * idx;
		x[idx] += dx[i] * D0 + double(efield.x) * EM[idx] * x[idx] / sqrtxy + FDLVO[i];
		y[idx] += dx[i + 1] * D0 + double(efield.x) * EM[idx] * y[idx] / sqrtxy + FDLVO[i + 1];
		z[idx] += dx[i + 2] * D0 + double(efield.y) * EM[idx] + FDLVO[i + 2];
		if (x[idx] < -1.0 * regionxyz) 
			x[idx] = 2.0 * region + x[idx];
		if (x[idx] > regionxyz)
			x[idx] = -2.0 * regionxyz + x[idx];
		if (y[idx] < -1.0 * regionxyz)
			y[idx] = 2.0 * regionxyz + y[idx];
		if (y[idx] > regionxyz)
			y[idx] = -2.0 * regionxyz + y[idx];
		if (z[idx] > 2 * regionxyz)
			z[idx] = 4 * regionxyz - z[idx];
	}
}

void diffusionRandom_cu3dRPY(double* position, const double* EM_c, int size, const float* efield, int ex, int ey,
    int maxstep, double region, double regionxyz, double D0, double radius_max, double radius_diff, 
	double ke, double kv, double debye, double* Mcp, double* M2) {
    //initialize the memory, texture, random seed, cholesky option, multiply option 
	//init variable
	double* x;
	double* y;
	double* z;
	double* radius; 
	double* EM;
	double* M;
	double* workspace;
	double* dx;
	double* X;
	double* gpu_X;
	double* FDLVO;
	size_t matrix_size = static_cast<size_t>(size) * 9 * sizeof(double) * static_cast<size_t>(size);
	//alloate memory 10000 beads take 13GB memory in GPU
	X = new double[3 * size];
	cudaArray* cuArray;
	cudaMalloc((void**)&x, size * sizeof(double));
	cudaMalloc((void**)&y, size * sizeof(double));
	cudaMalloc((void**)&z, size * sizeof(double));
	cudaMalloc((void**)&EM, size * sizeof(double)); 
	cudaMalloc((void**)&radius, size * sizeof(double));
	cudaMalloc((void**)&M, matrix_size);
	cudaMalloc((void**)&dx, 3 * size * sizeof(double));
	cudaMalloc((void**)&gpu_X, 3 * size * sizeof(double));
	cudaMalloc((void**)&FDLVO, 3 * size * sizeof(double));
	//copy value and init the textrue 
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();
	cudaMallocArray(&cuArray, &channelDesc, ex, ey);
	cudaMemcpy((void*)x, (void*)position, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)y, (void*)(position + size), size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)z, (void*)(position + 2 * size), size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)radius, (void*)(position + 3 * size), size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)EM, (void*)EM_c, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuArray, 0, 0, efield, ex * ey* sizeof(float2), cudaMemcpyHostToDevice);
	
	efieldtex.addressMode[0] = cudaAddressModeClamp;
	efieldtex.addressMode[1] = cudaAddressModeClamp;
	efieldtex.normalized = false;
	cudaBindTextureToArray(efieldtex, cuArray);
	//chelosky decomposition, worksize int may overflow, sparse matrix try later
	int* chelo_info;
	cudaMalloc(&chelo_info, sizeof(int));
	int workSize = 0;
	cusolverDnHandle_t chole_handle;
	cusolverDnCreate(&chole_handle);
	cusolverDnDpotrf_bufferSize(chole_handle, CUBLAS_FILL_MODE_LOWER, 3 * size, M, 3 * size, &workSize);
	
	cudaMalloc(&workspace, workSize * sizeof(double));
	//cblas multiply matrix vector 
	cublasHandle_t multi_handle;
	cublasCreate(&multi_handle); 
	const double alpha = 1;
	const double beta = 0;
	//exit(EXIT_FAILURE);
	//for loop -> hi_tensor -> cholesky decomposition -> brownian update 
	int blockDims = 256;
	int gridDims = (size + blockDims - 1) / blockDims;
	for (int i = 0; i < maxstep; i++) {
		if (i % 10 == 0) {
			kernel_HI_tensor << <gridDims, blockDims >> > (x, y, z, M, radius, size, radius_max, radius_diff);
			cudaCheckError(cudaGetLastError());
			generateRandomNumbers(X, size * 3);
			cudaMemcpy((void*)gpu_X, (void*)X, 3 * size * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy((void*)Mcp, (void*)M, matrix_size, cudaMemcpyDeviceToHost);
			kernel_DLVO_tensor << <gridDims, blockDims >> > (x, y, z, FDLVO, radius, size, debye, ke, kv);
			cudaDeviceSynchronize();
			cublasDsymv(multi_handle, CUBLAS_FILL_MODE_LOWER, size * 3, &alpha, M, 3 * size, FDLVO, 1, &beta, FDLVO, 1);
			cudaDeviceSynchronize();
			cusolverDnDpotrf(chole_handle, CUBLAS_FILL_MODE_LOWER, 3 * size, M, 3 * size, workspace, workSize, chelo_info);
			cudaDeviceSynchronize();
			cudaMemcpy((void*)M2, (void*)M, matrix_size, cudaMemcpyDeviceToHost);
			cudaMemcpy((void*)M2, (void*)FDLVO, 3 * size * 8, cudaMemcpyDeviceToHost);
			cublasDsymv(multi_handle, CUBLAS_FILL_MODE_LOWER, size * 3, &alpha, M, 3 * size, gpu_X, 1, &beta, gpu_X, 1);
		}
		kernel_update << <gridDims, blockDims>> > (x, y, z, EM, size, gpu_X, FDLVO, region, regionxyz, D0);
	}

	cudaMemcpy((void*)position, (void*)x, size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)(position + size), (void*)y, size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)(position + 2 * size), (void*)z, size * sizeof(double), cudaMemcpyDeviceToHost);
	
	cusolverDnDestroy(chole_handle);
	cublasDestroy(multi_handle);
	cudaFree(x);
	cudaFree(y);
	cudaFree(z); 
	cudaFree(radius); 
	cudaFree(EM);
	cudaFree(dx); 
	cudaFree(gpu_X); 
	cudaFree(M);
	cudaFree(FDLVO);
	cudaFree(workspace); 
	cudaUnbindTexture(efieldtex);
	cudaFreeArray(cuArray);
	cudaFree(chelo_info);
	delete[] X;
}