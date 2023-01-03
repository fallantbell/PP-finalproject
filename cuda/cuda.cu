#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/CycleTimer.h"

void input(int *node_ptr,int *edge_ptr,float **matrix){
    scanf("%d",node_ptr);
    scanf("%d",edge_ptr);
    int node=*node_ptr,edge=*edge_ptr;
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
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            printf("%.3f\n",matrix[i*n+j]);
        }
        // printf("\n");
    }
}

__global__ void APSP(int n,int k,float *matrix){

    int j = blockIdx.x*blockDim.x+threadIdx.x;
    int i = blockIdx.y*blockDim.y+threadIdx.y;

    if(i<n && j<n){ // 因為node 不一定是threadnum 的倍數, 所以有可能會超過node大小, 要做額外判斷
        int in=i*n;
        int kn=k*n;
        float Dik=matrix[in+k];
        float Dkj=matrix[kn+j];
        float Dij=matrix[in+j];
        if(Dik+Dkj<Dij){
            matrix[in+j]=Dik+Dkj;
        }
    }
}

int main(int argc, char *argv[]){
    int node,edge;
    float *matrix;
    float *matrix_gpu;
    int threadnum = strtol(argv[1],NULL,10);;
    double T_S,T_E;
    float elapsedTime;

    input(&node,&edge,&matrix);

    cudaSetDevice(5); 

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

    dim3 block((node+threadnum-1)/threadnum,(node+threadnum-1)/threadnum);
    dim3 threads(threadnum,threadnum);

    for(int k=0;k<node;k++){
        APSP<<<block,threads>>>(node,k,matrix_gpu);
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
    // output(node,matrix);

    cudaFree(matrix_gpu);
    free(matrix);

}
