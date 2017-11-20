//////////////////////////////////////////////
// GPU-BASED PULSE-COUPLED NEURAL NETWORK
// developer : ERIC JANSEN
// e-mail : janseneric[at]gmail[dot]com
// ONLY WORKING UNDER LINUX
//////////////////////////////////////////////

#include "gpupcnnkernel.cuh"

__global__ void gpuProcessS2(cv::gpu::DevMem2D_<float> S,//const cv::gpu::DevMem2D_<float>& S,
		cv::gpu::DevMem2D_<float> S2)//cv::gpu::DevMem2D_<float>& S2)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > S2.rows || j > S2.cols) return;

  S2.ptr(i)[j] = S.ptr(i)[j]/255.0;
}

__global__ void gpuProcessE(cv::gpu::DevMem2D_<float> E)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > E.rows || j > E.cols) return;

  E.ptr(i)[j] = 2.0;
}

__global__ void gpuPCNN(cv::gpu::DevMem2Df F,
		cv::gpu::DevMem2Df L,
		cv::gpu::DevMem2Df E,
		cv::gpu::DevMem2Df S2,
		cv::gpu::DevMem2D Sum1,
		cv::gpu::PtrStep Y,
		float dAF,float dAL,float dAE,
		float dVF,float dVE,float dB,
    int R,int C)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= R || j >= C) return;

    float dF,dL,dU,dE;

    dF = F.ptr(i)[j];
    dF = exp(-dAF) * F.ptr(i)[j] + S2.ptr(i)[j] + dVF * Sum1.ptr(i)[j];
    F.ptr(i)[j] = dF;

    dL = L.ptr(i)[j];
    dL = exp(-dAL) * dL + L.ptr(i)[j] * Sum1.ptr(i)[j];
    L.ptr(i)[j] = dL;

    dU = F.ptr(i)[j] * (1 + dB * L.ptr(i)[j]);
    dE = E.ptr(i)[j];
    dE = exp(-dAE) * E.ptr(i)[j] + dVE * Y.ptr(i)[j];
    E.ptr(i)[j] = dE;

    if (dU - dE > 0) Y.ptr(i)[j] = 255;
    else Y.ptr(i)[j] = 0;
}

void callGPUProcessS2(cv::gpu::DevMem2D_<float> S,
    cv::gpu::DevMem2D_<float> S2)
{
  dim3 block(16,16);
  dim3 grid((S2.rows+15)/16,(S2.cols+15)/16);
  gpuProcessS2<<<grid,block>>>(S,S2);
}

void callGPUProcessE(cv::gpu::DevMem2D_<float> E)
{
  dim3 block(16,16);
  dim3 grid((E.rows+15)/16,(E.cols+15)/16);
  gpuProcessE<<<grid,block>>>(E);
}

void callGPUKernel(const cv::gpu::DevMem2Df& F,
	const cv::gpu::DevMem2Df& L,
	const cv::gpu::DevMem2Df& E,
	const cv::gpu::DevMem2Df& S2,
  const cv::gpu::DevMem2D& Sum1,
  cv::gpu::PtrStep Y,
//	const cv::gpu::DevMem2D& Y,
	const float& dAF,const float& dAL,const float& dAE,
	const float& dVF,const float& dVE,const float& dB,
  const int& R,const int& C)
{
  cv::gpu::DevMem2D_<float> gF(F),gL(L),gE(E),gS2(S2);
  cv::gpu::DevMem2D_<unsigned char> gSum1(Sum1);//gY(Y);

	dim3 block(16,16);
	dim3 grid((F.cols+block.x-1)/block.x,(F.rows+block.y-1)/block.y);
	gpuPCNN<<<grid,block>>>(gF,gL,gE,gS2,gSum1,Y,
			dAF,dAL,dAE,dVF,dVE,dB,R,C);
}
