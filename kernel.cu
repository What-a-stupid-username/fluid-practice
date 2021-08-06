#include "kernel.cuh"



__device__ float r[GRID_COUNT][GRID_COUNT];
__device__ float p[GRID_COUNT][GRID_COUNT];
__device__ float Ap[GRID_COUNT][GRID_COUNT];
__device__ float r_r;
__device__ float p_A_p;
__device__ float next_r_r;

__device__ int valids[GRID_COUNT * GRID_COUNT];
__device__ float A[GRID_COUNT * GRID_COUNT][4], B[GRID_COUNT * GRID_COUNT], x[GRID_COUNT][GRID_COUNT];
__device__ float padding_A;

__device__ float& RA(int i, int j) {
	if (j < 0) return padding_A;
	int row, col;
	if (j >= i) {
		col = i;
		row = (j - i) / GRID_COUNT + 2 * (j - i) % GRID_COUNT;
	}
	else {
		col = j;
		row = (i - j) / GRID_COUNT + 2 * (i - j) % GRID_COUNT;
	}
	return A[col][row];
}

__device__ float& SR(float q[GRID_COUNT][GRID_COUNT], int x, int y) {
	return q[min(GRID_COUNT - 1, max(0, x))][min(GRID_COUNT - 1, max(0, y))];
}


__global__ void Run(int count) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= count) return;
	int col = valids[id];
	
	int vi = col % GRID_COUNT, vj = col / GRID_COUNT;
	float k =
		RA(col, (vi + 1) + vj * GRID_COUNT) * SR(x, vi + 1, vj) +
		RA(col, (vi - 1) + vj * GRID_COUNT) * SR(x, vi - 1, vj) +
		RA(col, vi + (vj + 1) * GRID_COUNT) * SR(x, vi, vj + 1) +
		RA(col, vi + (vj - 1) * GRID_COUNT) * SR(x, vi, vj - 1);
	x[vi][vj] = (B[col] - k) / RA(col, col);
}

__global__ void Run0(int count) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= count) return;
	int col = valids[id];

	int vi = col % GRID_COUNT, vj = col / GRID_COUNT;
	p[vi][vj] = r[vi][vj] = B[col] - (
		RA(col, col) * SR(x, vi, vj) +
		RA(col, (vi + 1) + vj * GRID_COUNT) * SR(x, vi + 1, vj) +
		RA(col, (vi - 1) + vj * GRID_COUNT) * SR(x, vi - 1, vj) +
		RA(col, vi + (vj + 1) * GRID_COUNT) * SR(x, vi, vj + 1) +
		RA(col, vi + (vj - 1) * GRID_COUNT) * SR(x, vi, vj - 1)
		);
}

__global__ void Run1(int count) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= count) return;
	int col = valids[id];

	int vi = col % GRID_COUNT, vj = col / GRID_COUNT;
	atomicAdd(&r_r, r[vi][vj] * r[vi][vj]);
}

__global__ void Run2(int count) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= count) return;
	int col = valids[id];

	int vi = col % GRID_COUNT, vj = col / GRID_COUNT;
	Ap[vi][vj] =
		RA(col, col) * SR(p, vi, vj) +
		RA(col, (vi + 1) + vj * GRID_COUNT) * SR(p, vi + 1, vj) +
		RA(col, (vi - 1) + vj * GRID_COUNT) * SR(p, vi - 1, vj) +
		RA(col, vi + (vj + 1) * GRID_COUNT) * SR(p, vi, vj + 1) +
		RA(col, vi + (vj - 1) * GRID_COUNT) * SR(p, vi, vj - 1);
}

__global__ void Run3(int count) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= count) return;
	int col = valids[id];

	int vi = col % GRID_COUNT, vj = col / GRID_COUNT;
	atomicAdd(&p_A_p, p[vi][vj] * Ap[vi][vj]);
}

__global__ void Run4(int count) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= count) return;
	int col = valids[id];

	int vi = col % GRID_COUNT, vj = col / GRID_COUNT;

	float a = p_A_p == 0 ? 0 : r_r / p_A_p;
	x[vi][vj] += a * p[vi][vj];
	r[vi][vj] += -a * Ap[vi][vj];
}

__global__ void Run5(int count) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= count) return;
	int col = valids[id];

	int vi = col % GRID_COUNT, vj = col / GRID_COUNT;

	atomicAdd(&next_r_r, r[vi][vj] * r[vi][vj]);
}

__global__ void Run6(int count) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= count) return;
	int col = valids[id];

	int vi = col % GRID_COUNT, vj = col / GRID_COUNT;

	float beta = r_r == 0 ? 0 : next_r_r / r_r;
	p[vi][vj] = r[vi][vj] + beta * p[vi][vj];
}

void Run(int LoopNum, void* validList, int validCount, void* A_, void* b_, void* res_) {

	void* ptr;
	cudaGetSymbolAddress(&ptr, r);
	cudaMemset(ptr, 0, sizeof(float) * GRID_COUNT * GRID_COUNT);
	cudaGetSymbolAddress(&ptr, p);
	cudaMemset(ptr, 0, sizeof(float) * GRID_COUNT * GRID_COUNT);
	cudaGetSymbolAddress(&ptr, Ap);
	cudaMemset(ptr, 0, sizeof(float) * GRID_COUNT * GRID_COUNT);


	cudaMemcpyToSymbol(A, A_, sizeof(float) * GRID_COUNT * GRID_COUNT * 4, 0, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(B, b_, sizeof(float) * GRID_COUNT * GRID_COUNT, 0, cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(valids, validList, sizeof(int) * GRID_COUNT * GRID_COUNT, 0, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(x, res_, sizeof(float) * GRID_COUNT * GRID_COUNT, 0, cudaMemcpyKind::cudaMemcpyHostToDevice);

	dim3 dimBlock(32);
	dim3 dimGrid((validCount + 31) / 32);

	Run0<<<dimGrid, dimBlock>>>(validCount);
	Run1<<<dimGrid, dimBlock>>>(validCount);

	float next_r_r_host;
	float zero = 0;
	int total_iter = 10000;
	for (int loop = 0; loop < total_iter; loop++) {
		Run2<<<dimGrid, dimBlock>>>(validCount);
		cudaMemcpyToSymbol(p_A_p, &zero, sizeof(float), 0, cudaMemcpyKind::cudaMemcpyHostToDevice);
		Run3<<<dimGrid, dimBlock>>>(validCount);
		Run4<<<dimGrid, dimBlock>>>(validCount);
		cudaMemcpyToSymbol(next_r_r, &zero, sizeof(float), 0, cudaMemcpyKind::cudaMemcpyHostToDevice);
		Run5<<<dimGrid, dimBlock>>>(validCount);

		cudaMemcpyFromSymbol(&next_r_r_host, next_r_r, sizeof(float), 0, cudaMemcpyKind::cudaMemcpyDeviceToHost);
		Run6<<<dimGrid, dimBlock>>>(validCount);

		cudaMemcpyToSymbol(r_r, &next_r_r_host, sizeof(float), 0, cudaMemcpyKind::cudaMemcpyHostToDevice);
		if (next_r_r_host < 1) {
			printf("early out with %d iter  ", loop);
			break;
		}
	}
	printf("error: %f\n", next_r_r_host);

	for (int i = 0; i < LoopNum; i++)
	{
		Run<<<dimGrid, dimBlock>>>(validCount);
	}  
	  
	cudaMemcpyFromSymbol(res_, x, sizeof(float) * GRID_COUNT * GRID_COUNT, 0, cudaMemcpyKind::cudaMemcpyDeviceToHost); 
}