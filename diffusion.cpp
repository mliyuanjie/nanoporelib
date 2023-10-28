#include "pch.h"
#include "diffusion.h" 


void diffusionRandom(float* data, const float* dx, int number, float region, float maxregion, int* maxstep_p) {
	/*
	data: position of each particle
	dx: diffusion coefficient distance sqrt(2*D*dt)->nm 
	drift: electrophoretic mobility nm default 0
	number: total number of protein 
	radius: maxlength of the simulation region 
	region: stop region->effective trap region 
	maxstep: maxtep of the simulation 
	*/
	double avog = 6.02214076e23;
	double pi = 3.14159265359;
	std::random_device dev;
	std::mt19937 rng(dev());
	int maxstep = maxstep_p[0];
	std::cout << *maxstep_p;
	std::normal_distribution<> dist{ 0.0, 1.0 };
	for (int it = 0; it < maxstep; it++) {
		int sum = 0;
		#pragma omp parallel for reduction(+ : sum)
		for (int i = 1; i < number; i++) {
			data[i] = data[i] + dx[i] * dist(rng);
			if (data[i] < region) {
				sum += 1;
			}
			if (data[i] > maxregion) 
				data[i] = maxregion;
		}
		if (sum > 0) {
			maxstep_p[0] = it;
			break;
		}
	}
	return;
}