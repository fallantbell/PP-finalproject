#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/CycleTimer.h"

#define MIN_MACRO(a,b) ( (a) < (b) ? (a) : (b) )
#define block_dim 32

void input(int *node_ptr,int *edge_ptr,float **matrix){
    scanf("%d",node_ptr);
    scanf("%d",edge_ptr);
    int node=*node_ptr,edge=*edge_ptr;
    
    if(node % block_dim !=0){
        node = node - (node % block_dim) + block_dim;
    }
    *matrix = (float*)malloc(sizeof(float)*node*node);


    for(int i=0;i<node;i++){
        for(int j=0;j<node;j++){
            if(i==j){
                *(*matrix+i*node+j)=0.0;
            }
            else{
                *(*matrix+i*node+j)=100000.0;
            }
            
        }
    }

    for(int i=0;i<edge;i++){
        int n1,n2;
        float v;
        scanf("%d %d %f",&n1,&n2,&v);
        *(*matrix+n1*node+n2)=v;
    }
}
void output(int n,float *matrix){

    int node = n;
    if(node % block_dim !=0){
        node = node - (node % block_dim) + block_dim;
    }

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            printf("%.3f\n",matrix[i*node+j]);
        }
        // printf("\n");
    }
}

// kernel for pass 1
__global__ void apsp_parallel_2_kernel_1(float* dev_dist, int N, int stage) {

	// get indices for the current cell
	int i = stage*block_dim + threadIdx.y;
	int j = stage*block_dim + threadIdx.x;

	int tid = i*N + j;

	// allocate shared memory
	__shared__ float sd[block_dim][block_dim];
    
    // if(i<N && j<N){
        // copy data from main memory to shared memory
        sd[threadIdx.y][threadIdx.x] = dev_dist[tid];
        __syncthreads();

        // iterate for the values of k
        for (int k = 0; k < block_dim; k++) {

            float vertex   = sd[threadIdx.y][threadIdx.x];
            float alt_path = sd[k][threadIdx.x] + sd[threadIdx.y][k];

            sd[threadIdx.y][threadIdx.x] = MIN_MACRO( vertex, alt_path );
            __syncthreads();

        }

        // write result back to main memory
        dev_dist[tid] = sd[threadIdx.y][threadIdx.x];
    // }

	
}

// kernel for pass 2
__global__ void apsp_parallel_2_kernel_2(float* dev_dist, int N, int stage) {
	
	// get indices of the current block
	int skip_center_block = MIN_MACRO( (blockIdx.x+1)/(stage+1), 1 );

	int box_y = 0;
	int box_x = 0;

	// block in the same row with the primary block
	if (blockIdx.y == 0) {
		box_y = stage;
		box_x = blockIdx.x + skip_center_block;
	
	}
	// block in the same column with the primary block
	else {
		box_y = blockIdx.x + skip_center_block;
		box_x = stage;
	}

	// get indices for the current cell
	int i = box_y * block_dim + threadIdx.y;
	int j = box_x * block_dim + threadIdx.x;

	// get indices for the cell of the primary block
	int pi = stage*block_dim + threadIdx.y;
	int pj = stage*block_dim + threadIdx.x;

	// get indices of the cells from the device main memory
	int tid = i*N + j;
	int ptid = pi*N + pj;

	// allocate shared memory
	__shared__ float sd[block_dim][2*block_dim];

    // if(i < N && j < N && pi < N && pj < N){
        // copy current block and primary block to shared memory
        sd[threadIdx.y][threadIdx.x]             = dev_dist[tid]; // 當前的 block
        sd[threadIdx.y][block_dim + threadIdx.x] = dev_dist[ptid]; // phase1 算出來的對角線 block
        __syncthreads();

        // block in the same row with the primary block
        if (blockIdx.y == 0) {
            for (int k = 0; k < block_dim; k++) {
                // if(j+k >= N || pj+k >= N || i+k >= N || pi+k >=N){ 
                //     break;
                // }

                float vertex   = sd[threadIdx.y][threadIdx.x];
                float alt_path = sd[k][threadIdx.x] 
                                                + sd[threadIdx.y][block_dim + k];

                sd[threadIdx.y][threadIdx.x] = MIN_MACRO( vertex, alt_path );
                __syncthreads();

            }
        }
        // block in the same column with the primary block
        else {
            for (int k = 0; k < block_dim; k++) {
                // if(j+k >= N || pj+k >= N || i+k >= N || pi+k >=N){ 
                //     break;
                // }

                float vertex   = sd[threadIdx.y][threadIdx.x];
                float alt_path = sd[threadIdx.y][k] 
                                            + sd[k][block_dim + threadIdx.x];

                sd[threadIdx.y][threadIdx.x] = MIN_MACRO( vertex, alt_path );
                __syncthreads();
            }
        }

        // write result back to main memory
        dev_dist[tid] = sd[threadIdx.y][threadIdx.x];
    // }

	
}

// kernel for pass 3
__global__ void apsp_parallel_2_kernel_3(float* dev_dist, int N, int stage) {

	// get indices of the current block
	int skip_center_block_y = MIN_MACRO( (blockIdx.y+1)/(stage+1), 1 );
	int skip_center_block_x = MIN_MACRO( (blockIdx.x+1)/(stage+1), 1 );

	int box_y = blockIdx.y + skip_center_block_y;
	int box_x = blockIdx.x + skip_center_block_x;

	// get indices for the current cell
	int i = box_y * block_dim + threadIdx.y;
	int j = box_x * block_dim + threadIdx.x;

	// get indices from the cell in the same row with the current box
	int ri = i;
	int rj = stage*block_dim + threadIdx.x;
	
	// get indices from the cell in the same column with the current box
	int ci = stage*block_dim + threadIdx.y;
	int cj = j;

	// get indices of the cells from the device main memory
	int  tid =  i*N +  j;
	int rtid = ri*N + rj;
	int ctid = ci*N + cj;

	// allocate shared memory
	__shared__ float sd[block_dim][3*block_dim];

    // if(i < N && j < N && ri < N && rj < N && ci < N && cj < N){
        // copy current block and depending blocks to shared memory
        sd[threadIdx.y][threadIdx.x]               = dev_dist[tid];
        sd[threadIdx.y][  block_dim + threadIdx.x] = dev_dist[rtid]; 
        sd[threadIdx.y][2*block_dim + threadIdx.x] = dev_dist[ctid]; 
        __syncthreads();

        for (int k = 0; k < block_dim; k++) {
            // if(j+k >= N || rj+k >= N || i+k >= N || ri+k >=N || ci+k >= N || cj+k >=N ){ 
            //     break;
            // }

            float vertex   = sd[threadIdx.y][threadIdx.x];
            float alt_path = sd[threadIdx.y][block_dim + k]
                                    + sd[k][2*block_dim + threadIdx.x];

            sd[threadIdx.y][threadIdx.x] = MIN_MACRO( vertex, alt_path );
            __syncthreads();
        }

        // write result back to main memory
        dev_dist[tid] = sd[threadIdx.y][threadIdx.x];
    // }

	
}
 
int main(int argc, char *argv[]){
    int node,edge;
    float *matrix;
    float *matrix_gpu;
    int threadnum = block_dim;
    double T_S,T_E;
    float elapsedTime;

    input(&node,&edge,&matrix);
    int o_node = node;
    if(node % block_dim !=0){
        node = node - (node % block_dim) + block_dim;
    }

    cudaSetDevice(4);

    cudaEvent_t e_start, e_stop;
	cudaEventCreate(&e_start);
	cudaEventCreate(&e_stop);

    T_S = currentSeconds();

    cudaEventRecord(e_start, 0);

    cudaMalloc((void**)&matrix_gpu, sizeof(float) *node*node);

    cudaEventRecord(e_stop, 0);
	cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    
    // printf("\n\033[31m [malloc time]: \033[0m\t\t[%.3f] ms\n\n", elapsedTime);
    printf("%.3f\n", elapsedTime);
    cudaEventRecord(e_start, 0);
        
    cudaMemcpy(matrix_gpu, matrix, sizeof(float) *node*node, cudaMemcpyHostToDevice);

    cudaEventRecord(e_stop, 0);
	cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    // printf("\n\033[31m [mem2cuda time]: \033[0m\t\t[%.3f] ms\n\n", elapsedTime);
    printf("%.3f\n", elapsedTime);
    cudaEventRecord(e_start, 0);

    int nBlocks = node/block_dim;
    dim3 blocks1(1);
    dim3 blocks2(nBlocks-1,2);
    dim3 blocks3(nBlocks-1, nBlocks-1);
    dim3 threads(threadnum,threadnum);
    for (int stage = 0; stage < nBlocks; stage++) {

        // pass 1 - launch kernel 1
        apsp_parallel_2_kernel_1<<<blocks1,threads>>>(matrix_gpu,node,stage);

        // pass 2 - launch kernel 2
        apsp_parallel_2_kernel_2<<<blocks2,threads>>>(matrix_gpu,node,stage);

        // pass 3 - launch kernel 3
        apsp_parallel_2_kernel_3<<<blocks3,threads>>>(matrix_gpu,node,stage);
    }
    cudaEventRecord(e_stop, 0);
	cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    
    // printf("\n\033[31m [execution time]: \033[0m\t\t[%.3f] ms\n\n", elapsedTime);
    printf("%.3f\n", elapsedTime);
    cudaEventRecord(e_start, 0);
    cudaMemcpy(matrix, matrix_gpu , sizeof(float) *node*node, cudaMemcpyDeviceToHost);

    cudaEventRecord(e_stop, 0);
	cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    // printf("\n\033[31m [cuda2mem time]: \033[0m\t\t[%.3f] ms\n\n", elapsedTime);
    printf("%.3f\n", elapsedTime);


    T_E = currentSeconds();
    // printf("\n\033[31m [Total time]: \033[0m\t\t[%.3f] ms\n\n", (T_E-T_S) * 1000);
    printf("%.3f\n", (T_E-T_S) * 1000);
    // output(o_node,matrix);


    cudaFree(matrix_gpu);
    free(matrix);

}
