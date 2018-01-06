#pragma omp parallel
#pragma once

#define __cuda_cuda_h__

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include <stdio.h>
#include<cuda.h>
#include<cublas.h>
#include<cuda_device_runtime_api.h>
#include<device_launch_parameters.h>
#include<device_functions.h>
#include<arrayfire.h>
#include<af\cuda.h>
#include<iostream>
#include<math.h>
#include<omp.h>

#define PI 3.142578956
#define  MAX(a,b)= (((a)>(b))? (a):(b))
#define  MIN(a,b)= (((a)<(b))? (a):(b))

using namespace af;

__device__ float GaussianProcessDistribution(float el, float menA, float stdA) {
	float a, b, c;
	a = 1 / (sqrt(2 * PI*stdA*stdA));
	b = ((el - menA)*(el - menA)) / (2 * stdA*stdA);
	c = a*exp(-b);
	return (c);
}

__global__ void GaussianProcessKernel(float *out, float *in, const int nx, const int ny, float menA, float stdA)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy*nx + ix;

	if (ix < nx && iy < ny) {
		out[idx] = GaussianProcessDistribution(in[idx], menA, stdA);
		__syncthreads();
	}
}

static af::array cudaGaussianDistribution(const af::array &nsrc, float menA, float stdA) {
	af::array temp;
	int  height = nsrc.dims(0);
	int  width = nsrc.dims(1);
	size_t nbyte = height*width*sizeof(f32);

	// Get Arrayfire's internal CUDA stream
	int af_id = af::getDevice();
	cudaStream_t af_stream = afcu::getStream(af_id);

	// allocate host memory
	float *h_x = nsrc.host<float>();
	float *h_y = (float*)malloc(nbyte);

	// allocate device memory
	float *d_A, *d_B;
	cudaMalloc((float**)&d_A, nbyte);
	cudaMalloc((float**)&d_B, nbyte);

	// copy data from host to device
	cudaMemcpy(d_A, h_x, nbyte, cudaMemcpyHostToDevice);
	dim3 block(32, 32);
	dim3 grid((height + block.x - 1) / block.x, (width + block.y - 1) / block.y);
	GaussianProcessKernel << < grid, block, 0, af_stream >> >(d_B, d_A, height, width, menA, stdA);

	// copy data from device to host
	cudaMemcpy(h_y, d_B, nbyte, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	cudaDeviceSynchronize();
	temp = af::array(height, width, h_y, afHost);

	//memory free
	cudaFree(d_A);
	cudaFree(d_B);
	free(h_y);
	return(temp);
}

__global__ void GaussianProcessSegKernel(float *out, float *ina, float *inb, int nx, int ny, float kld)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy*nx + ix;
	float a = 0.25f;
	float x = a*abs((1 - kld));
	float y;
	if (ix < nx && iy < ny) {
		if (ina[idx] > inb[idx]) {
			y = (inb[idx]);
			inb[idx] = a*(y - x);
			__syncthreads();
		}
	}

	if (ix < nx && iy < ny) {
		if (ina[idx] > inb[idx])
			out[idx] = 1.0f;
		else
			out[idx] = 0.0f;
		__syncthreads();
	}
}

static af::array GaussianProcessSegMethod(af::array& inA, af::array& gA, float kld) {
	af::array temp;
	int  height = inA.dims(0);
	int  width = inA.dims(1);
	size_t nbyte = height*width*sizeof(f32);

	// Get Arrayfire's internal CUDA stream
	int af_id = af::getDevice();
	cudaStream_t af_stream = afcu::getStream(af_id);

	// allocate host memory
	float *h_xa = inA.host<float>();
	float *h_xb = gA.host<float>();
	float *h_ref = (float*)malloc(nbyte);

	// allocate device memory
	float *d_A, *d_B, *d_C;
	cudaMalloc((float**)&d_A, nbyte);
	cudaMalloc((float**)&d_B, nbyte);
	cudaMalloc((float**)&d_C, nbyte);

	// copy data from host to device
	cudaMemcpy(d_A, h_xa, nbyte, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_xb, nbyte, cudaMemcpyHostToDevice);

	dim3 block(32, 32);
	dim3 grid((height + block.x - 1) / block.x, (width + block.y - 1) / block.y);
	GaussianProcessSegKernel << < grid, block, 0, af_stream >> >(d_C, d_A, d_B, height, width, kld);

	// copy data from device to host
	cudaMemcpy(h_ref, d_C, nbyte, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	cudaDeviceSynchronize();
	temp = af::array(height, width, h_ref, afHost);

	//memory free
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_ref);
	return(temp);
}

void prewitt(af::array &mag, af::array &dir, const af::array &in)
{
	static float h1[] = { 1, 1, 1 };
	static float h2[] = { -1, 0, 1 };
	static af::array colf(3, 1, h1);
	static af::array rowf(3, 1, h2);
	// Find the gradients
	af::array Gy = convolve(rowf, colf, in);
	af::array Gx = convolve(colf, rowf, in);
	// Find magnitude and direction
	mag = hypot(Gx, Gy);
	dir = atan2(Gy, Gx);
}
void sobelFilter(af::array &mag, af::array &dir, const af::array &in)
{
	af::array Gx, Gy;
	sobel(Gx, Gy, in, 3);
	// Find magnitude and direction
	mag = hypot(Gx, Gy);
	dir = atan2(Gy, Gx);
}

af::array normalize(const af::array &in)
{
	float mx = max<float>(in);
	float mn = min<float>(in);
	return (in - mn) / (mx - mn);
}

af::array edge(const af::array &in, int method = 0)
{
	int w = 5;
	if (in.dims(0) < 512) w = 3;
	if (in.dims(0) > 2048) w = 7;
	int h = 5;
	if (in.dims(0) < 512) h = 3;
	if (in.dims(0) > 2048) h = 7;
	af::array ker = gaussianKernel(w, h);
	af::array smooth = convolve(in, ker);
	af::array mag, dir;
	switch (method) {
	case  1: prewitt(mag, dir, smooth); break;
	case  2: sobelFilter(mag, dir, smooth);   break;
	default: throw af::exception("Unsupported type");
	}
	return normalize(mag);
}

static void imageUSSegDemo(bool console) {
	// Load color image
	af::array imsrcA;
	imsrcA = loadImage("src.jpg", AF_RGB);
	//dimension of image
	int rows = imsrcA.dims(0);
	int cols = imsrcA.dims(1);
	int chns = imsrcA.dims(2);
	//creat new image
	af::array nsrcA = moddims(imsrcA, rows, cols*chns);

	//Difference of Gaussians of Energy response
	af::array dogA, dogB;
	int rada1 = 1;
	int rada2 = 2;
	dogA = af::dog(nsrcA, rada1, rada2);
	dogA = af::abs(dogA);

	// pixel density region by Calculate image gradients
	af::array ixA, iyA;
	grad(ixA, iyA, dogA);

	// Compute second-order derivatives
	af::array ixxA = ixA * ixA;
	af::array ixyA = ixA * iyA;
	af::array iyyA = iyA * iyA;

	// Calculate trace
	af::array itrA = ixxA + iyyA;

	// Calculate determinant
	af::array idetA = ixxA * iyyA + ixyA * ixyA;
	float cr = af::corrcoef<float>(ixxA, iyyA);

	// Calculate Energy response
	af::array resA = idetA + cr*(itrA * itrA);

	// mean and std values
	nsrcA = nsrcA;
	float menA = af::mean<float>(nsrcA);
	float stdA = af::stdev<float>(nsrcA);
	float menA1 = af::mean<float>(resA);
	float stdA1 = af::stdev<float>(resA);

	af::array pdfG = cudaGaussianDistribution(nsrcA, menA, stdA);
	af::array pdfH = cudaGaussianDistribution(resA, menA1, stdA1);

	// CUDA gaussian KLD estimation manipulation
	float kldAB = log(stdA1 / stdA) + ((stdA*stdA) + (menA - menA1)* (menA - menA1)) / (2 * stdA1*stdA1) - .05f;
	kldAB = 1 / kldAB;
	float bias = 1.90f;
	cr = (bias* cr);
	af::array hgpdf = ((cr + pdfH*kldAB) + pdfG);

	// CUDA image fusion
	nsrcA /= 255.f;
	af::array segm = GaussianProcessSegMethod(nsrcA, hgpdf, kldAB);
	//edges
	af::array prewitt = edge(segm, 1);
	af::array sobelFilter = edge(nsrcA, 2);
	prewitt = moddims(prewitt, rows, cols, chns);
	sobelFilter = moddims(sobelFilter, rows, cols, chns);

	af::array out = regions(segm.as(b8), AF_CONNECTIVITY_8);
	af::array fimg = moddims(out, rows, cols, chns);
	// display input images
	af::array color_imgA = imsrcA / 255.f;

	if (!console) {
		af::Window wnd("Segmentation Demo");
		wnd.setPos(50, 50);
		while (!wnd.close()) {
			wnd.grid(1, 4);
			wnd(0, 0).image(color_imgA, "Color source Image");
			wnd(0, 1).image(sobelFilter, "befor edge");
			wnd(0, 2).image(prewitt, "after edge");
			wnd(0, 3).image(fimg.as(u8), "Color Segment Image");
			wnd.show();
		}
	}
}

int main(int argc, char** argv) {
	int device = argc > 1 ? atoi(argv[1]) : 0;
	bool console = argc > 2 ? argv[2][0] == '_' : false;
	int dev = 0;
	cudaSetDevice(dev);
	cudaDeviceProp devPro;
	cudaGetDeviceProperties(&devPro, dev);
	try {
		af::setDevice(device);
		af::info();
		printf("Using Device %d: %s\n", dev, devPro.name);
		// check if support mapped memory
		if (!devPro.canMapHostMemory) {
			printf("Device %d does not support mapping CPU host memory!\n", dev);
			cudaDeviceReset();
			exit(0);
		}
		/*Function demo*/
		imageUSSegDemo(console);
		int i = 5;
#pragma omp parallel
		{
			printf("E is equal to %d\n", i);
		}

		int j = 256; // a shared variable
#pragma omp parallel
		{
			int x; // a variable local or private to each thread
			x = omp_get_thread_num();
			printf("x = %d, i = %d\n", x, j);
		}
	}
	catch (af::exception exp) {
		fprintf(stderr, "%s\n", exp.what());
		throw;
	}
	system("PAUSE");
	return EXIT_SUCCESS;
}