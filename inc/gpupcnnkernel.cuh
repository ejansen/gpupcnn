//////////////////////////////////////////////
// GPU-BASED PULSE-COUPLED NEURAL NETWORK
// developer : ERIC JANSEN
// e-mail : janseneric[at]gmail[dot]com
// ONLY WORKING UNDER LINUX
//////////////////////////////////////////////

#ifndef _GPUPCNNKERNEL_CUH_
#define _GPUPCNNKERNEL_CUH_

#include <opencv2/gpu/devmem2d.hpp>
#include <cmath>

void callGPUProcessS2(cv::gpu::DevMem2D_<float> S,
	cv::gpu::DevMem2D_<float> S2);

void callGPUProcessE(cv::gpu::DevMem2D_<float> E);

void callGPUKernel(const cv::gpu::DevMem2D_<float>& F,
	const cv::gpu::DevMem2D_<float>& L,
	const cv::gpu::DevMem2D_<float>& E,
	const cv::gpu::DevMem2D_<float>& S2,
	const cv::gpu::DevMem2D_<unsigned char>& Sum1,
	cv::gpu::PtrStep Y,
	const float& dAF,const float& dAL,const float& dAE,
	const float& dVF,const float& dVE,const float& dB,
  const int& R,const int& C);

#endif
