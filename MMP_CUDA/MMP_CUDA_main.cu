//ММП на CUDA
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <iomanip>

#include <windows.h>

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

using namespace std;

// Constants

const int S = 134862;
const int m = 247;
const int K = S / m;

const int grid_size_m = 13;
const int block_size_m = 19;

const int per_thread_m = m / (grid_size_m * block_size_m);

// Submatrices in main algorithm

double a[K][m*m];
double b[K][m*m];
double c[K][m*m];
double d[K][m*m];
double e[K][m*m];
double f[K][m];

// Temporal variables in main algorithm

double tmpv[m];
double tmpv2[m];
double tmpv3[m];
double tmpm[m*m];
double tmpm2[m*m];
double tmpm3[m*m];
double delta[m*m];
double alpha[K - 1][m*m];
double beta[K - 2][m*m];
double gamma[K][m];

// Solution of SLAE
double y[K][m];

// Device variables

double *dev_a = new double[m*m];
double *dev_b = new double[m*m];
double *dev_c = new double[m*m];
double *dev_d = new double[m*m];
double *dev_e = new double[m*m];
double *dev_f = new double[m];

double *dev_tmpv = new double[m];
double *dev_tmpv2 = new double[m];
double *dev_tmpv3 = new double[m];
double *dev_tmpm = new double[m*m];
double *dev_tmpm2 = new double[m*m];
double *dev_tmpm3 = new double[m*m];
double *dev_delta = new double[m*m];
double *dev_alpha = new double[m*m];
double *dev_beta = new double[m*m];
double *dev_gamma = new double[m];

double *dev_y = new double[m];

void CUDA_prep() {
	CUDA_CALL(cudaMalloc((void**)&dev_a, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_b, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_c, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_d, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_e, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_f, m * sizeof(double)));

	CUDA_CALL(cudaMalloc((void**)&dev_tmpv, m * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_tmpv2, m * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_tmpv3, m * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_tmpm, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_tmpm2, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_tmpm3, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_delta, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_alpha, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_beta, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_gamma, m * sizeof(double)));

	CUDA_CALL(cudaMalloc((void**)&dev_y, m * sizeof(double)));
}

void CUDA_end() {
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_d);
	cudaFree(dev_e);
	cudaFree(dev_f);

	cudaFree(dev_tmpv);
	cudaFree(dev_tmpv2);
	cudaFree(dev_tmpv3);
	cudaFree(dev_tmpm);
	cudaFree(dev_tmpm2);
	cudaFree(dev_tmpm3);
	cudaFree(dev_delta);
	cudaFree(dev_alpha);
	cudaFree(dev_beta);
	cudaFree(dev_gamma);

	cudaFree(dev_y);
}


// Functions
void showv(double * ptr, int start, int size) {
	for (int i = start; i < start + size; i++) {
		cout << ptr[i] << endl;
	}
}

void write_in_file(double *X) {
	fstream result_file;
	result_file.open("X_cuda.dat");

	for (int i = 0; i < S; ++i) {
		result_file << X[i] << endl;
	}

	result_file.close();
	cout << endl;
	cout << "Answer was written in file" << endl;
}

//----------------------------------------------------

void addm(double * A, double * B, double * C) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			C[i*m + j] = A[i*m + j] + B[i*m + j];
		}
	}
}

__global__ void GPU_addm(double * A, double * B, double * C) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread_m; i < (thread + 1) * per_thread_m; i++) {
		for (int j = 0; j < m; j++) {
			C[i*m + j] = A[i*m + j] + B[i*m + j];
		}
	}
}

//----------------------------------------------------


void subm(double * A, double * B, double * C) {
#pragma omp parallel for
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			C[i*m + j] = A[i*m + j] - B[i*m + j];
		}
	}
}

__global__ void GPU_subm(double * A, double * B, double * C) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread_m; i < (thread + 1) * per_thread_m; i++) {
		for (int j = 0; j < m; j++) {
			C[i*m + j] = A[i*m + j] - B[i*m + j];
		}
	}
}

//----------------------------------------------------

void mulm(double * A, double * B, double * C) {
#pragma omp parallel for
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			C[i*m + j] = 0;
			for (int k = 0; k < m; k++) {
				C[i*m + j] += A[i*m + k] * B[k*m + j];
			}
		}
	}
}

__global__ void GPU_mulm(double * A, double * B, double * C) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread_m; i < (thread + 1) * per_thread_m; i++) {
		for (int j = 0; j < m; j++) {
			C[i*m + j] = 0;
			for (int k = 0; k < m; k++) {
				C[i*m + j] += A[i*m + k] * B[k*m + j];
			}
		}
	}
}

//----------------------------------------------------

void addv(double * A, double * B, double * C) {
#pragma omp parallel for
	for (int i = 0; i < m; i++) {
		C[i] = A[i] + B[i];
	}
}

__global__ void GPU_addv(double * A, double * B, double * C) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread_m; i < (thread + 1) * per_thread_m; i++) {
		C[i] = A[i] + B[i];
	}
}

//----------------------------------------------------

void subv(double * A, double * B, double * C) {
#pragma omp parallel for
	for (int i = 0; i < m; i++) {
		C[i] = A[i] - B[i];
	}
}

__global__ void GPU_subv(double * A, double * B, double * C) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread_m; i < (thread + 1) * per_thread_m; i++) {
		C[i] = A[i] - B[i];
	}
}

//----------------------------------------------------

void mulmv(double * A, double * B, double * C) {
#pragma omp parallel for
	for (int i = 0; i < m; i++) {
		C[i] = 0;
		for (int j = 0; j < m; j++) {
			C[i] += A[i*m + j] * B[j];
		}
	}
}


__global__ void GPU_mulmv(double * A, double * B, double * C) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread_m; i < (thread + 1) * per_thread_m; i++) {
		C[i] = 0;
		for (int j = 0; j < m; j++) {
			C[i] += A[i*m + j] * B[j];
		}
	}
}

//----------------------------------------------------

void copyv(double * A, double * B) {
#pragma omp parallel for
	for (int i = 0; i < m; i++) {
		B[i] = A[i];
	}
}

__global__ void GPU_copyv(double * A, double * B) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread_m; i < (thread + 1) * per_thread_m; i++) {
		B[i] = A[i];
	}
}

//----------------------------------------------------

void copym(double * A, double * B) {
#pragma omp parallel for
	for (int i = 0; i < m*m; i++) {
		B[i] = A[i];
	}
}

//<<<13, 19>>>
__global__ void GPU_copym(double * A, double * B) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * m; i < (thread + 1) * m; i++) {
		B[i] = A[i];
	}
}

//----------------------------------------------------

//для спуска = size - thread;

void solvev(double * A, double * B, double * X) {
	// Gaussian elimination 
	double tmpA[m*m];
	copym(A, tmpA);
	copyv(B, X);

	// Direct
	for (int d = 0; d < m; d++) {
		// dividing on diagonal element
		if (tmpA[d*m + d] != 1) {
			X[d] /= tmpA[d*m + d];
			for (int j = m - 1; j > d; j--) {
				tmpA[d*m + j] /= tmpA[d*m + d];
			}
			tmpA[d*m + d] = 1;
		}

		// nulling elements below diagonal in column
		for (int i = d + 1; i < m; i++) {
			if (tmpA[i*m + d] != 0) {
				double mult = -tmpA[i*m + d] / tmpA[d*m + d];
				X[i] += mult*X[d];
				for (int j = d; j < m; j++) {
					tmpA[i*m + j] += mult*tmpA[d*m + j];
				}
			}
		}
	}

	// Inverse
	for (int d = m - 1; d >= 0; d--) {
		// nulling elements above diagonal in column
		for (int i = d - 1; i >= 0; i--) {
			if (tmpA[i*m + d] != 0) {
				double mult = -tmpA[i*m + d] / tmpA[d*m + d];
				X[i] += mult*X[d];
				for (int j = d; j < m; j++) {
					tmpA[i*m + j] += mult*tmpA[d*m + j];
				}
			}
		}
	}
}

//__global__ void GPU_solvev(double * A, double * B, double * X) {
//	// Gaussian elimination 
//	int thread = blockIdx.x * blockDim.x + threadIdx.x;
//	
//	double tmpA[m*m];
//	double mult = 0;
//	copym(A, tmpA);
//	copyv(B, X);
//
//	// Direct
//	for (int d = thread * per_thread_m; d < (thread + 1) * per_thread_m; d++) {
//		// dividing on diagonal element
//		if (tmpA[d*m + d] != 1) {
//			X[d] /= tmpA[d*m + d];
//			for (int j = m - 1; j > d; j--) {
//				tmpA[d*m + j] /= tmpA[d*m + d];
//			}
//			tmpA[d*m + d] = 1;
//		}
//
//		// nulling elements below diagonal in column
//		for (int i = d + 1; i < m; i++) {
//			if (tmpA[i*m + d] != 0) {
//				mult = -tmpA[i*m + d] / tmpA[d*m + d];
//				X[i] += mult*X[d];
//				for (int j = d; j < m; j++) {
//					tmpA[i*m + j] += mult*tmpA[d*m + j];
//				}
//			}
//		}
//	}
//
//	// Inverse
//	for (int d = ((thread + 1) * per_thread_m) - 1; d >= thread * per_thread_m; --d) {
//	//for (int d = m - 1; d >= 0; d--) {
//		// nulling elements above diagonal in column
//		for (int i = d - 1; i >= 0; i--) {
//			if (tmpA[i*m + d] != 0) {
//				mult = -tmpA[i*m + d] / tmpA[d*m + d];
//				X[i] += mult*X[d];
//				for (int j = d; j < m; j++) {
//					tmpA[i*m + j] += mult*tmpA[d*m + j];
//				}
//			}
//		}
//	}
//}

//----------------------------------------------------

void solvem(double * A, double * B, double * X) {
	// Gaussian elimination 
	double tmpA[m*m];
	copym(B, X);
	copym(A, tmpA);

	// Direct

	for (int d = 0; d < m; d++) {
		if (tmpA[d*m + d] != 1) {
			for (int j = 0; j < m; j++) {
				X[d*m + j] /= tmpA[d*m + d];
			}
			for (int j = m - 1; j >= 0; j--) {
				tmpA[d*m + j] /= tmpA[d*m + d];
			}
		}

		for (int i = d + 1; i < m; i++) {
			if (tmpA[i*m + d] != 0) {
				double mult = -tmpA[i*m + d] / tmpA[d*m + d];
				for (int j = 0; j < m; j++) {
					X[i*m + j] += mult*X[d*m + j];
				}
				for (int j = 0; j < m; j++) {
					tmpA[i*m + j] += mult*tmpA[d*m + j];
				}
			}
		}
	}

	// Inverse

	for (int d = m - 1; d >= 0; d--) {
		for (int i = d - 1; i >= 0; i--) {
			if (tmpA[i*m + d] != 0) {
				double mult = -tmpA[i*m + d] / tmpA[d*m + d];
				for (int j = 0; j < m; j++) {
					X[i*m + j] += mult*X[d*m + j];
				}
				for (int j = d; j < m; j++) {
					tmpA[i*m + j] += mult*tmpA[d*m + j];
				}
			}
		}
	}
}

//__global__ void GPU_solvem(double * A, double * B, double * X) {
//	// Gaussian elimination 
//	int thread = blockIdx.x * blockDim.x + threadIdx.x;
//	double mult = 0;
//	double tmpA[m*m];
//	copym(B, X);
//	copym(A, tmpA);
//
//	// Direct
//
//	for (int d = thread * per_thread_m; d < (thread + 1) * per_thread_m; d++) {
//		if (tmpA[d*m + d] != 1) {
//			for (int j = 0; j < m; j++) {
//				X[d*m + j] /= tmpA[d*m + d];
//			}
//			for (int j = m - 1; j >= 0; j--) {
//				tmpA[d*m + j] /= tmpA[d*m + d];
//			}
//		}
//
//		for (int i = d + 1; i < m; i++) {
//			if (tmpA[i*m + d] != 0) {
//				mult = -tmpA[i*m + d] / tmpA[d*m + d];
//				for (int j = 0; j < m; j++) {
//					X[i*m + j] += mult*X[d*m + j];
//				}
//				for (int j = 0; j < m; j++) {
//					tmpA[i*m + j] += mult*tmpA[d*m + j];
//				}
//			}
//		}
//	}
//
//	// Inverse
//
//	for (int d = ((thread + 1) * per_thread_m) - 1; d >= thread * per_thread_m; --d) {
//		for (int i = d - 1; i >= 0; i--) {
//			if (tmpA[i*m + d] != 0) {
//				mult = -tmpA[i*m + d] / tmpA[d*m + d];
//				for (int j = 0; j < m; j++) {
//					X[i*m + j] += mult*X[d*m + j];
//				}
//				for (int j = d; j < m; j++) {
//					tmpA[i*m + j] += mult*tmpA[d*m + j];
//				}
//			}
//		}
//	}
//}

//----------------------------------------------------

// Preparations
int prep() {
	string tmps;
	double tmpd;
	ifstream A1f;
	ifstream A2f;
	ifstream A3f;
	ifstream A4f;
	ifstream A5f;
	ifstream Ff;

	A1f.open("A1.dat");
	A2f.open("A2.dat");
	A3f.open("A3.dat");
	A4f.open("A4.dat");
	A5f.open("A5.dat");
	Ff.open("F");

	if (A1f.is_open() && A2f.is_open() && A3f.is_open()
		&& A4f.is_open() && A5f.is_open() && Ff.is_open()) {
		std::getline(A1f, tmps);
		std::getline(A2f, tmps);
		std::getline(A3f, tmps);
		std::getline(A4f, tmps);
		std::getline(A5f, tmps);
		std::getline(Ff, tmps);

		A2f >> tmpd;
		for (int i = 0; i < m; i++) {
			A1f >> tmpd;
		}

		for (int i = 0; i < K; i++) {
			for (int j = 0; j < m; j++) {
				A3f >> c[i][j*m + j];
				if (j - 1 >= 0) {
					A2f >> c[i][j*m + j - 1];
				}
				if (j + 1 < m) {
					A4f >> c[i][j*m + j + 1];
				}
				if (i >= 1) {
					A1f >> b[i][j*m + j];
					b[i][j*m + j] *= -1;
				}
				if (i < K - 1) {
					A5f >> d[i][j*m + j];
					d[i][j*m + j] *= -1;
				}
				if (i >= 2) {
					a[i][j*m + j] = 0.001;
				}
				if (i < K - 2) {
					e[i][j*m + j] = 0.001;
				}
				Ff >> f[i][j];
			}
			A2f >> tmpd;
			A4f >> tmpd;
		}

		A1f.close();
		A2f.close();
		A3f.close();
		A4f.close();
		A5f.close();
		Ff.close();
	}
	else {
		std::cout << "File error, program aborted" << std::endl;
		return 1;
	}

	return 0;
}

// Main algorithm
int algo() {

	CUDA_CALL(cudaMalloc(&dev_a, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_b, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_c, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_d, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_e, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_f, m * sizeof(double)));

	CUDA_CALL(cudaMalloc(&dev_tmpv, m * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_tmpv2, m * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_tmpv3, m * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_tmpm, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_tmpm2, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_tmpm3, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_delta, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_alpha, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_beta, m * m * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_gamma, m * sizeof(double)));

	CUDA_CALL(cudaMalloc(&dev_y, m * sizeof(double)));

	//----------------------------------------------------

	cout << "first start" << endl;
	solvem(c[0], d[0], alpha[0]);
	solvem(c[0], e[0], beta[0]);
	solvev(c[0], f[0], gamma[0]);

	//----------------------------------------------------

	cout << "second start" << endl;
	CUDA_CALL(cudaMemcpy(dev_b, b[1], m * m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_alpha, alpha[0], m * m * sizeof(double), cudaMemcpyHostToDevice));
	//CUDA_CALL(cudaMemcpy(dev_tmpm, tmpm, m * m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_c, c[1], m * m * sizeof(double), cudaMemcpyHostToDevice));

	//mulm(b[1], alpha[0], tmpm);
	GPU_mulm<<<grid_size_m, block_size_m>>>(dev_b, dev_alpha, dev_tmpm);
	//subm(c[1], tmpm, delta);
	GPU_subm<<<grid_size_m, block_size_m>>>(dev_c, dev_tmpm, dev_delta);

	//----------------------------------------------------

	cout << "third start" << endl;
	CUDA_CALL(cudaMemcpy(dev_beta, beta[0], m * m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_d, d[1], m * m * sizeof(double), cudaMemcpyHostToDevice));

	//mulm(b[1], beta[0], tmpm);
	GPU_mulm<<<grid_size_m, block_size_m>>>(dev_b, dev_beta, dev_tmpm);
	//subm(d[1], tmpm, tmpm2);
	GPU_subm<<<grid_size_m, block_size_m>>>(dev_d, dev_tmpm, dev_tmpm2);

	CUDA_CALL(cudaMemcpy(delta, dev_delta, m * m * sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(tmpm2, dev_tmpm2, m * m * sizeof(double), cudaMemcpyDeviceToHost));

	solvem(delta, tmpm2, alpha[1]);
	solvem(delta, e[1], beta[1]);

	CUDA_CALL(cudaMemcpy(dev_b, b[1], m * m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_gamma, gamma[0], m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_f, f[1], m * sizeof(double), cudaMemcpyHostToDevice));

	//mulmv(b[1], gamma[0], tmpv);
	GPU_mulmv<<<grid_size_m, block_size_m>>>(dev_b, dev_gamma, dev_tmpv);
	//addv(f[1], tmpv, tmpv2);
	GPU_addv<<<grid_size_m, block_size_m>>>(dev_f, dev_tmpv, dev_tmpv2);

	CUDA_CALL(cudaMemcpy(tmpv2, dev_tmpv2, m * sizeof(double), cudaMemcpyDeviceToHost));

	solvev(delta, tmpv2, gamma[1]);

	//----------------------------------------------------

	cout << "cycle start" << endl;
	for (int i = 2; i < K - 2; i++) {
		cout << "i = " << i << " / " << K - 2 << endl;
		
		CUDA_CALL(cudaMemcpy(dev_a, a[i], m * m * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_alpha, alpha[i - 2], m * m * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_b, b[i], m * m * sizeof(double), cudaMemcpyHostToDevice));

		//mulm(a[i], alpha[i - 2], tmpm);
		GPU_mulm<<<grid_size_m, block_size_m>>>(dev_a, dev_alpha, dev_tmpm);
		//subm(tmpm, b[i], tmpm3);
		GPU_subm<<<grid_size_m, block_size_m>>>(dev_tmpm, dev_b, dev_tmpm3);

		//----------------------------------------------------

		CUDA_CALL(cudaMemcpy(dev_alpha, alpha[i - 1], m * m * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_c, c[i], m * m * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_beta, beta[i - 2], m * m * sizeof(double), cudaMemcpyHostToDevice));

		//mulm(tmpm3, alpha[i - 1], tmpm);
		GPU_mulm<<<grid_size_m, block_size_m>>>(dev_tmpm3, dev_alpha, dev_tmpm);

		//addm(tmpm, c[i], tmpm2);
		GPU_addm<<<grid_size_m, block_size_m>>>(dev_tmpm, dev_c, dev_tmpm2);

		//mulm(a[i], beta[i - 2], tmpm);
		GPU_mulm<<<grid_size_m, block_size_m>>>(dev_a, dev_beta, dev_tmpm);

		//subm(tmpm2, tmpm, delta);
		GPU_subm<<<grid_size_m, block_size_m>>>(dev_tmpm2, dev_tmpm, dev_delta);

		//----------------------------------------------------

		CUDA_CALL(cudaMemcpy(dev_beta, beta[i - 1], m * m * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_d, d[i], m * m * sizeof(double), cudaMemcpyHostToDevice));

		//mulm(tmpm3, beta[i - 1], tmpm);
		GPU_mulm<<<grid_size_m, block_size_m>>>(dev_tmpm3, dev_beta, dev_tmpm);

		//addm(tmpm, d[i], tmpm2);
		GPU_addm<<<grid_size_m, block_size_m>>>(dev_tmpm, dev_d, dev_tmpm2);

		CUDA_CALL(cudaMemcpy(delta, dev_delta, m * m * sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(tmpm2, dev_tmpm2, m * m * sizeof(double), cudaMemcpyDeviceToHost));

		solvem(delta, tmpm2, alpha[i]);

		//----------------------------------------------------
		
		solvem(delta, e[i], beta[i]);

		//----------------------------------------------------

		CUDA_CALL(cudaMemcpy(dev_gamma, gamma[i - 1], m * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_f, f[i], m * sizeof(double), cudaMemcpyHostToDevice));

		//mulmv(tmpm3, gamma[i - 1], tmpm);
		GPU_mulmv<<<grid_size_m, block_size_m>>>(dev_tmpm3, dev_gamma, dev_tmpm);

		//subv(f[i], tmpm, tmpm2);
		GPU_subv<<<grid_size_m, block_size_m>>>(dev_f, dev_tmpm, dev_tmpm2);

		CUDA_CALL(cudaMemcpy(dev_gamma, gamma[i - 2], m * sizeof(double), cudaMemcpyHostToDevice));

		//mulmv(a[i], gamma[i - 2], tmpm);
		GPU_mulmv<<<grid_size_m, block_size_m>>>(dev_a, dev_gamma, dev_tmpm);

		//subv(tmpm2, tmpm, tmpm3);
		GPU_subv<<<grid_size_m, block_size_m>>>(dev_tmpm2, dev_tmpm, dev_tmpm3);

		CUDA_CALL(cudaMemcpy(tmpm3, dev_tmpm3, m * m * sizeof(double), cudaMemcpyDeviceToHost));

		solvev(delta, tmpm3, gamma[i]);
	}

	//----------------------------------------------------

	cout << "fourth start" << endl;

	CUDA_CALL(cudaMemcpy(dev_a, a[K - 2], m * m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_alpha, alpha[K - 4], m * m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_b, b[K - 2], m * m * sizeof(double), cudaMemcpyHostToDevice));
	
	//mulm(a[K - 2], alpha[K - 4], tmpm);
	GPU_mulm<<<grid_size_m, block_size_m>>>(dev_a, dev_alpha, dev_tmpm);

	//subm(tmpm, b[K - 2], tmpm3);
	GPU_subm<<<grid_size_m, block_size_m>>>(dev_tmpm, dev_b, dev_tmpm3);

	CUDA_CALL(cudaMemcpy(dev_a, a[K - 3], m * m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_c, c[K - 2], m * m * sizeof(double), cudaMemcpyHostToDevice));
	
	//mulm(tmpm3, alpha[K - 3], tmpm);
	GPU_mulm<<<grid_size_m, block_size_m>>>(dev_tmpm3, dev_alpha, dev_tmpm);
	
	//addm(tmpm, c[K - 2], tmpm2);
	GPU_addm<<<grid_size_m, block_size_m>>>(dev_tmpm, dev_c, dev_tmpm2);

	CUDA_CALL(cudaMemcpy(dev_a, a[K - 2], m * m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_beta, beta[K - 4], m * m * sizeof(double), cudaMemcpyHostToDevice));

	//mulm(a[K - 2], beta[K - 4], tmpm);
	GPU_mulm<<<grid_size_m, block_size_m>>>(dev_a, dev_beta, dev_tmpm);

	//subm(tmpm2, tmpm, delta);
	GPU_subm<<<grid_size_m, block_size_m>>>(dev_tmpm2, dev_tmpm, dev_delta);
	
	//----------------------------------------------------

	cout << "fifth start" << endl;

	CUDA_CALL(cudaMemcpy(dev_beta, beta[K - 3], m * m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_d, d[K - 2], m * m * sizeof(double), cudaMemcpyHostToDevice));

	//mulm(tmpm3, beta[K - 3], tmpm);
	GPU_mulm<<<grid_size_m, block_size_m>>>(dev_tmpm3, dev_beta, dev_tmpm);

	//addm(tmpm, d[K - 2], tmpm2);
	GPU_addm<<<grid_size_m, block_size_m>>>(dev_tmpm, dev_d, dev_tmpm2);

	CUDA_CALL(cudaMemcpy(delta, dev_delta, m * m * sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(tmpm2, dev_tmpm2, m * m * sizeof(double), cudaMemcpyDeviceToHost));

	solvem(delta, tmpm2, alpha[K - 2]);

	//----------------------------------------------------

	cout << "sixth start" << endl;

	CUDA_CALL(cudaMemcpy(dev_gamma, gamma[K - 3], m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_f, f[K - 2], m * sizeof(double), cudaMemcpyHostToDevice));

	//mulmv(tmpm3, gamma[K - 3], tmpm);
	GPU_mulmv<<<grid_size_m, block_size_m>>>(dev_tmpm3, dev_gamma, dev_tmpm);

	//subv(f[K - 2], tmpm, tmpm2);
	GPU_subv<<<grid_size_m, block_size_m>>>(dev_f, dev_tmpm, dev_tmpm2);

	CUDA_CALL(cudaMemcpy(dev_a, a[K - 2], m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_gamma, gamma[K - 4], m * sizeof(double), cudaMemcpyHostToDevice));


	//mulmv(a[K - 2], gamma[K - 4], tmpm);
	GPU_mulmv<<<grid_size_m, block_size_m>>>(dev_a, dev_gamma, dev_tmpm);

	//subv(tmpm2, tmpm, tmpm3);
	GPU_subv<<<grid_size_m, block_size_m>>>(dev_tmpm2, dev_tmpm, dev_tmpm3);

	CUDA_CALL(cudaMemcpy(tmpm3, dev_tmpm3, m * m * sizeof(double), cudaMemcpyDeviceToHost));

	solvev(delta, tmpm3, gamma[K - 2]);

	//----------------------------------------------------

	cout << "seventh start" << endl;

	CUDA_CALL(cudaMemcpy(dev_a, a[K - 1], m * m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_alpha, alpha[K - 3], m * m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_b, b[K - 1], m * m * sizeof(double), cudaMemcpyHostToDevice));

	//mulm(a[K - 1], alpha[K - 3], tmpm);
	GPU_mulm<<<grid_size_m, block_size_m>>>(dev_a, dev_alpha, dev_tmpm);

	//subm(tmpm, b[K - 1], tmpm3);
	GPU_subm<<<grid_size_m, block_size_m>>>(dev_tmpm, dev_b, dev_tmpm3);

	CUDA_CALL(cudaMemcpy(dev_alpha, alpha[K - 2], m * m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_c, c[K - 1], m * m * sizeof(double), cudaMemcpyHostToDevice));

	//mulm(tmpm3, alpha[K - 2], tmpm);
	GPU_mulm<<<grid_size_m, block_size_m>>>(dev_tmpm3, dev_alpha, dev_tmpm);

	//addm(tmpm, c[K - 1], tmpm2);
	GPU_addm<<<grid_size_m, block_size_m>>>(dev_tmpm, dev_c, dev_tmpm2);

	CUDA_CALL(cudaMemcpy(dev_beta, beta[K - 3], m * m * sizeof(double), cudaMemcpyHostToDevice));

	//mulm(a[K - 1], beta[K - 3], tmpm);
	GPU_mulm<<<grid_size_m, block_size_m>>>(dev_a, dev_beta, dev_tmpm);

	//subm(tmpm2, tmpm, delta);
	GPU_subm<<<grid_size_m, block_size_m>>>(dev_tmpm2, dev_tmpm, dev_delta);

	//----------------------------------------------------

	cout << "eight start" << endl;

	CUDA_CALL(cudaMemcpy(dev_gamma, gamma[K - 2], m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_f, f[K - 1], m * sizeof(double), cudaMemcpyHostToDevice));

	//mulmv(tmpm3, gamma[K - 2], tmpm);
	GPU_mulmv<<<grid_size_m, block_size_m>>>(dev_tmpm3, dev_gamma, dev_tmpm);

	//subv(f[K - 1], tmpm, tmpm2);
	GPU_subv<<<grid_size_m, block_size_m>>>(dev_f, dev_tmpm, dev_tmpm2);

	CUDA_CALL(cudaMemcpy(dev_gamma, gamma[K - 3], m * sizeof(double), cudaMemcpyHostToDevice));

	//mulmv(a[K - 1], gamma[K - 3], tmpm);
	GPU_mulmv<<<grid_size_m, block_size_m>>>(dev_a, dev_gamma, dev_tmpm);

	//subv(tmpm2, tmpm, tmpm3);
	GPU_subv<<<grid_size_m, block_size_m>>>(dev_tmpm2, dev_tmpm, dev_tmpm3);

	CUDA_CALL(cudaMemcpy(delta, dev_delta, m * m * sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(tmpm3, dev_tmpm3, m * m * sizeof(double), cudaMemcpyDeviceToHost));

	solvev(delta, tmpm3, gamma[K - 1]);

	//----------------------------------------------------

	cout << "nine start" << endl;

	CUDA_CALL(cudaMemcpy(dev_gamma, gamma[K - 1], m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_y, y[K - 1], m * sizeof(double), cudaMemcpyHostToDevice));

	//copyv(gamma[K - 1], y[K - 1]);
	CUDA_CALL(cudaMemcpy(dev_y, dev_gamma, m * sizeof(double), cudaMemcpyDeviceToDevice));
	CUDA_CALL(cudaMemcpy(y[K - 1], dev_y, m * sizeof(double), cudaMemcpyDeviceToHost));

	//mulmv(alpha[K - 2], y[K - 1], tmpv);
	GPU_mulmv<<<grid_size_m, block_size_m>>>(dev_alpha, dev_y, dev_tmpv);

	CUDA_CALL(cudaMemcpy(dev_gamma, gamma[K - 2], m * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_y, y[K - 2], m * sizeof(double), cudaMemcpyHostToDevice));
	
	//addv(tmpv, gamma[K - 2], y[K - 2]);
	GPU_addv<<<grid_size_m, block_size_m>>>(dev_tmpv, dev_gamma, dev_y);
	CUDA_CALL(cudaMemcpy(y[K - 2], dev_y, m * sizeof(double), cudaMemcpyDeviceToHost));

	//----------------------------------------------------

	//Распараллелить только это?

	cout << "second cycle start" << endl;
	for (int i = K - 3; i >= 0; i--) {

		cout << "i = " << i << " / " << K - 3 << endl;

		CUDA_CALL(cudaMemcpy(dev_alpha, alpha[i], m * m * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_y, y[i + 1], m * sizeof(double), cudaMemcpyHostToDevice));

		//mulmv(alpha[i], y[i + 1], tmpv);
		GPU_mulmv<<<grid_size_m, block_size_m>>>(dev_alpha, dev_y, dev_tmpv);

		CUDA_CALL(cudaMemcpy(dev_beta, beta[i], m * m * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_y, y[i + 2], m * sizeof(double), cudaMemcpyHostToDevice));

		//mulmv(beta[i], y[i + 2], tmpv2);
		GPU_mulmv<<<grid_size_m, block_size_m>>>(dev_alpha, dev_y, dev_tmpv2);

		//subv(tmpv, tmpv2, tmpv3);
		GPU_subv<<<grid_size_m, block_size_m>>>(dev_tmpv, dev_tmpv2, dev_tmpv3);

		CUDA_CALL(cudaMemcpy(dev_gamma, gamma[i], m * sizeof(double), cudaMemcpyHostToDevice));

		//addv(tmpv3, gamma[i], y[i]);
		GPU_addv<<<grid_size_m, block_size_m>>>(dev_tmpv3, dev_gamma, dev_y);

		CUDA_CALL(cudaMemcpy(y[i], dev_y, m * sizeof(double), cudaMemcpyDeviceToHost));

	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_d);
	cudaFree(dev_e);
	cudaFree(dev_f);

	cudaFree(dev_tmpv);
	cudaFree(dev_tmpv2);
	cudaFree(dev_tmpv3);
	cudaFree(dev_tmpm);
	cudaFree(dev_tmpm2);
	cudaFree(dev_tmpm3);
	cudaFree(dev_delta);
	cudaFree(dev_alpha);
	cudaFree(dev_beta);
	cudaFree(dev_gamma);

	cudaFree(dev_y);

	return 0;
}

// Entry point
int main() {
	// prepare

	clock_t begin = clock();
	if (prep() != 0) {
		cout << "1 No" << endl;
		return 1;
	}
	clock_t end = clock();
	std::cout << "Preparations = " << double(end - begin)
		/ CLOCKS_PER_SEC << " seconds" << std::endl;

	// CUDA_prep
	//CUDA_prep();

	// algorithm

	CUDA_CALL(cudaDeviceSynchronize());

	begin = clock();
	if (algo() != 0) {
		cout << "2 No" << endl;
		return 1;
	}
	end = clock();
	std::cout << "Algorithm = " << double(end - begin)
		/ CLOCKS_PER_SEC << " seconds" << std::endl;

	// CUDA_end
	//CUDA_end();

	// show result and exit

	//showv(y[0], 0, S);
	write_in_file(y[0]);

	system("pause");
	return 0;


	//Runtime: 728.236 sec || 13 x 19
}