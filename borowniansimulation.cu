#ifndef __CUDACC__  
#define __CUDACC__
#endif
#include "pch.h"
#include "borowniansimulation.cuh" 

texture<float2, cudaTextureType2D, cudaReadModeElementType> efieldtex;


__global__ void kernel_sim(float* data, float* dx, int size, float region, float maxregion, int maxstep_p, 
	int* not_terminated, unsigned long long seed, int simulate_round) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curandState state;
	curand_init(seed, idx, 0, &state);
	volatile __shared__ int wrap_found;
	if (threadIdx.x == 0)
		wrap_found = *not_terminated;
	__syncthreads();
	int i = 0;
	int trip = 0;
	while (trip < simulate_round) {
		if (threadIdx.x == 0) 
			wrap_found = 0;
		if (idx == 0) {
			atomicCAS(not_terminated + trip, 0, 0);
		}
		i = 0;
		__syncthreads();
		while (!wrap_found && i < maxstep_p) {
			if (idx < size) {
				data[idx] += dx[idx] * curand_normal(&state);
				if (data[idx] > maxregion) {
					data[idx] = maxregion;
				}
				if (data[idx] < region) {
					data[idx] = maxregion;
					atomicCAS(not_terminated + trip, 0, i);
				}
				if (threadIdx.x == 0 && *(not_terminated + trip))
					wrap_found = true;
			}
			i++;
			__syncthreads();
		}
		trip++;
		__syncthreads();
	}
	
}


__global__ void kernel_2d(float* x, float* y, float* EM, float* dx,
	int size, float region, float regionx, float regiony, int maxstep_p,
	int* not_terminated, unsigned long long seed, int simulate_round) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curandState state;
	curand_init(seed, idx, 0, &state);
	volatile __shared__ int wrap_found;
	if (threadIdx.x == 0)
		wrap_found = *not_terminated;
	__syncthreads();
	int i = 0;
	int trip = 0;
	float2 efield;
	while (trip < simulate_round) {
		if (threadIdx.x == 0)
			wrap_found = 0;
		if (idx == 0) {
			atomicCAS(not_terminated + trip, 0, 0);
		}
		i = 0;
		__syncthreads();
		//printf("Hello from block %d, thread %f\n", threadIdx.y, region);
		while (!wrap_found && i < maxstep_p) {
			if (idx < size && x[idx] > 0 && y[idx] > 0) {
				efield = tex2D<float2>(efieldtex, x[idx], y[idx]);
				x[idx] += dx[idx] * curand_normal(&state) + efield.x * EM[idx];
				y[idx] += dx[idx] * curand_normal(&state) + efield.y * EM[idx];
				if (x[idx] > regionx) 
					x[idx] = regionx;
				if (x[idx] < 0)
					x[idx] *= -1;
				if (y[idx] > regiony) 
					y[idx] = regiony;
				if (y[idx] < 0)
					y[idx] = 0;
				if (sqrtf(x[idx]*x[idx] + y[idx]*y[idx]) < region) {
					x[idx] = 0;
					y[idx] = 0;
					atomicCAS(not_terminated + trip, 0, i);
				}
				if (threadIdx.x == 0 && *(not_terminated + trip))
					wrap_found = true;
			}

			i++;
			__syncthreads();
		}
		trip++;
		__syncthreads();
	}

}

void diffusionRandom_cu(float* data, const float* dx, int size, float region, float maxregion, int* maxstep_p, int simulate_round) {
	float* gpu_data;
	float* gpu_dx;
	int* not_terminate;
	cudaMalloc((void**)&gpu_data, size * sizeof(float)); 
	cudaMalloc((void**)&gpu_dx, size * sizeof(float));
	cudaMalloc((void**)&not_terminate, simulate_round * 4);
	unsigned long long seed = time(NULL); 
	int blockDims = 256;
	int gridDims = (size + blockDims - 1) / blockDims;
	cudaMemcpy((void*)gpu_data, (void*)data, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)gpu_dx, (void*)dx, size * sizeof(float), cudaMemcpyHostToDevice);
	kernel_sim << <gridDims, blockDims >> > (gpu_data, gpu_dx, size, region, maxregion, *maxstep_p, not_terminate, seed, simulate_round);
	cudaMemcpy((void*)maxstep_p, (void*)not_terminate, simulate_round * 4, cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)data, (void*)gpu_data, size * 4, cudaMemcpyDeviceToHost); 
	cudaFree(gpu_data); 
	cudaFree(gpu_dx);
	cudaFree(not_terminate);
	return;
}


void diffusionRandom_cu2d(float* x, float* y, const float* dx, const float* EM, int size, const float* efield, int ex, int ey,
	int* maxstep_p, int simulate_round, float regionx, float regiony, float region0) {
	float* gpu_x;
	float* gpu_y;
	float* gpu_em;
	float* gpu_dx;
	int* not_terminate;
	cudaArray* cuArray;
	cudaMalloc((void**)&gpu_x, size * sizeof(float));
	cudaMalloc((void**)&gpu_y, size * sizeof(float));
	cudaMalloc((void**)&gpu_dx, size * sizeof(float));
	cudaMalloc((void**)&gpu_em, size * sizeof(float));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();
	cudaMallocArray(&cuArray, &channelDesc, ex, ey);
	cudaMalloc((void**)&not_terminate, simulate_round * 4);
	cudaMemcpy((void*)gpu_x, (void*)x, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)gpu_dx, (void*)dx, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)gpu_y, (void*)y, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)gpu_em, (void*)EM, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuArray, 0, 0, efield, ex * ey * sizeof(float2), cudaMemcpyHostToDevice);
	efieldtex.addressMode[0] = cudaAddressModeClamp;
	efieldtex.addressMode[1] = cudaAddressModeClamp;
	efieldtex.normalized = false;
	cudaBindTextureToArray(efieldtex, cuArray);

	unsigned long long seed = time(NULL);
	int blockDims = 256;
	
	int gridDims = (size + blockDims - 1) / blockDims;
	
	kernel_2d<<<gridDims, blockDims>>>(gpu_x, gpu_y, gpu_em, gpu_dx, size, region0, regionx, regiony, *maxstep_p, not_terminate, seed, simulate_round);
	cudaMemcpy((void*)maxstep_p, (void*)not_terminate, simulate_round * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)x, (void*)gpu_x, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)y, (void*)gpu_y, size * sizeof(float), cudaMemcpyDeviceToHost);


	cudaFree(gpu_x);
	cudaFree(gpu_dx);
	cudaFree(gpu_em);
	cudaUnbindTexture(efieldtex);
	cudaFreeArray(cuArray);
	cudaFree(not_terminate);
	return;
}