#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/CycleTimer.h"

#define block_dim 16

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
__global__ void kernel_1(int N, int stage,float* matrix) {

    int i,j,tid;

	i = stage*block_dim + threadIdx.y;
	j = stage*block_dim + threadIdx.x;
	tid = i*N + j;

	// allocate shared memory
	__shared__ float shared_block[block_dim][block_dim];
    
    // if(i<N && j<N){
        // to shared memory
        shared_block[threadIdx.y][threadIdx.x] = matrix[tid];
        __syncthreads();
        float Dik;
        float Dkj;
        float Dij;

        for (int k = 0; k < block_dim; k++) {

            Dik = shared_block[threadIdx.y][k];
            Dkj = shared_block[k][threadIdx.x];
            Dij = shared_block[threadIdx.y][threadIdx.x];
            if(Dik+Dkj<Dij){
                shared_block[threadIdx.y][threadIdx.x] = Dik+Dkj;
            }

            __syncthreads();

        }

        // back to main memory
        matrix[tid] = shared_block[threadIdx.y][threadIdx.x];
    // }

	
}

// kernel for pass 2
__global__ void kernel_2(int N, int stage,float* matrix) {
    
    int box_x,box_y;
    int i,j,tid,diagonal_i,diagonal_j,diagonal_tid;

	int skip_num;
    if((blockIdx.x+1)/(stage+1) < 1){
        skip_num = (blockIdx.x+1)/(stage+1);
    }
    else{
        skip_num = 1;
    }
    
    box_y = 0;
	box_x = 0;

	if (blockIdx.y == 0) {
		box_y = stage;
		box_x = blockIdx.x + skip_num;
	
	}
	else {
		box_y = blockIdx.x + skip_num;
		box_x = stage;
	}

	// current block
	i = box_y * block_dim + threadIdx.y;
	j = box_x * block_dim + threadIdx.x;
    tid = i*N + j;

	// diagonal block
	diagonal_i = stage*block_dim + threadIdx.y;
	diagonal_j = stage*block_dim + threadIdx.x;
    diagonal_tid = diagonal_i*N + diagonal_j;

	// allocate shared memory
	__shared__ float shared_block[block_dim][2*block_dim];

    // if(i < N && j < N && diagonal_i < N && diagonal_j < N){
        // to shared memory
        shared_block[threadIdx.y][threadIdx.x]             = matrix[tid]; // current block
        shared_block[threadIdx.y][block_dim + threadIdx.x] = matrix[diagonal_tid]; // diagonal block
        __syncthreads();
        float Dik;
        float Dkj;
        float Dij;

        // same row diagonal block
        if (blockIdx.y == 0) {
            for (int k = 0; k < block_dim; k++) {
                // if(j+k >= N || diagonal_j+k >= N || i+k >= N || diagonal_i+k >=N){ 
                //     break;
                // }

                Dik = shared_block[threadIdx.y][block_dim + k];
                Dkj = shared_block[k][threadIdx.x];
                Dij = shared_block[threadIdx.y][threadIdx.x];
                if(Dik+Dkj<Dij){
                    shared_block[threadIdx.y][threadIdx.x] = Dik+Dkj;
                }

                __syncthreads();

            }
        }
        // same column diagonal block
        else {
            for (int k = 0; k < block_dim; k++) {
                // if(j+k >= N || diagonal_j+k >= N || i+k >= N || diagonal_i+k >=N){ 
                //     break;
                // }

                Dik = shared_block[threadIdx.y][k];
                Dkj = shared_block[k][block_dim + threadIdx.x];
                Dij = shared_block[threadIdx.y][threadIdx.x];
                if(Dik+Dkj<Dij){
                    shared_block[threadIdx.y][threadIdx.x] = Dik+Dkj;
                }

                __syncthreads();
            }
        }

        // back to main memory
        matrix[tid] = shared_block[threadIdx.y][threadIdx.x];
    // }

	
}

// kernel for pass 3
__global__ void kernel_3(int N, int stage,float* matrix) {

    int skip_num_x,skip_num_y;
    int box_x,box_y;
    int i,j,row_i,row_j,col_i,col_j,tid,row_tid,col_tid;


    if((blockIdx.y+1)/(stage+1) < 1){
        skip_num_y=(blockIdx.y+1)/(stage+1);
    }
    else{
        skip_num_y=1;
    }

    if((blockIdx.x+1)/(stage+1) < 1){
        skip_num_x = (blockIdx.x+1)/(stage+1);
    }
    else{
        skip_num_x=1;
    }

	box_y = blockIdx.y + skip_num_y;
	box_x = blockIdx.x + skip_num_x;

	// current block
	i = box_y * block_dim + threadIdx.y;
	j = box_x * block_dim + threadIdx.x;
    tid =  i*N + j;

	// same row block
	row_i = i;
	row_j = stage*block_dim + threadIdx.x;
    row_tid = row_i*N + row_j;
	
	// same column block
	col_i = stage*block_dim + threadIdx.y;
	col_j = j;
    col_tid = col_i*N + col_j;

	// allocate shared memory
	__shared__ float shared_block[block_dim][3*block_dim];

    // if(i < N && j < N && row_i < N && row_j < N && col_i < N && col_j < N){
        // to shared memory
        shared_block[threadIdx.y][threadIdx.x]               = matrix[tid];
        shared_block[threadIdx.y][  block_dim + threadIdx.x] = matrix[row_tid]; 
        shared_block[threadIdx.y][2*block_dim + threadIdx.x] = matrix[col_tid]; 
        __syncthreads();

        float Dik;
        float Dkj;
        float Dij;

        for (int k = 0; k < block_dim; k++) {
            Dik = shared_block[threadIdx.y][block_dim + k];
            Dkj = shared_block[k][2*block_dim + threadIdx.x];
            Dij = shared_block[threadIdx.y][threadIdx.x];
            if(Dik+Dkj<Dij){
                shared_block[threadIdx.y][threadIdx.x] = Dik+Dkj;
            }
            __syncthreads();
        }

        // back to main memory
        matrix[tid] = shared_block[threadIdx.y][threadIdx.x];
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
    
    printf("\n\033[31m [malloc time]: \033[0m\t\t[%.3f] ms\n\n", elapsedTime);
    // printf("%.3f\n", elapsedTime);
    cudaEventRecord(e_start, 0);
        
    cudaMemcpy(matrix_gpu, matrix, sizeof(float) *node*node, cudaMemcpyHostToDevice);

    cudaEventRecord(e_stop, 0);
	cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    printf("\n\033[31m [mem2cuda time]: \033[0m\t\t[%.3f] ms\n\n", elapsedTime);
    // printf("%.3f\n", elapsedTime);
    cudaEventRecord(e_start, 0);

    int nBlocks = node/block_dim;
    dim3 blocks1(1);
    dim3 blocks2(nBlocks-1,2);
    dim3 blocks3(nBlocks-1, nBlocks-1);
    dim3 threads(threadnum,threadnum);
    for (int stage = 0; stage < nBlocks; stage++) {

        // pass 1 - launch kernel 1
        kernel_1<<<blocks1,threads>>>(node,stage,matrix_gpu);

        // pass 2 - launch kernel 2
        kernel_2<<<blocks2,threads>>>(node,stage,matrix_gpu);

        // pass 3 - launch kernel 3
        kernel_3<<<blocks3,threads>>>(node,stage,matrix_gpu);
    }
    cudaEventRecord(e_stop, 0);
	cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    
    printf("\n\033[31m [execution time]: \033[0m\t\t[%.3f] ms\n\n", elapsedTime);
    // printf("%.3f\n", elapsedTime);
    cudaEventRecord(e_start, 0);
    cudaMemcpy(matrix, matrix_gpu , sizeof(float) *node*node, cudaMemcpyDeviceToHost);

    cudaEventRecord(e_stop, 0);
	cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    printf("\n\033[31m [cuda2mem time]: \033[0m\t\t[%.3f] ms\n\n", elapsedTime);
    // printf("%.3f\n", elapsedTime);


    T_E = currentSeconds();
    printf("\n\033[31m [Total time]: \033[0m\t\t[%.3f] ms\n\n", (T_E-T_S) * 1000);
    // printf("%.3f\n", (T_E-T_S) * 1000);
    output(o_node,matrix);


    cudaFree(matrix_gpu);
    free(matrix);

}
