#include <stdio.h>
#include <stdlib.h>
#include "../common/CycleTimer.h"
#include <time.h>
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

void APSP(int n,float *matrix){

    // printf("\033[31m APSP --------------------- \n\n\033[0m");
    for(int k=0;k<n;k++){
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                float dis = matrix[i*n+k]+matrix[k*n+j];
                if(dis < matrix[i*n+j]){
                    matrix[i*n+j] = dis;
                }
               
            }
        }
    }

    // for(int i=0;i<n;i++){
    //     for(int j=0;j<n;j++){
    //         printf("%.2f \t",matrix[i*n+j]);
    //     }
    //     printf("\n");
    // }
}
void output(int n,float *matrix){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            printf("%.3f\n",matrix[i*n+j]);
        }
        // printf("\n");
    }
}

int main(void){
    int node,edge;
    float *matrix;
    double start_time, end_time;
    double S,E;

    input(&node,&edge,&matrix);
        
    start_time = currentSeconds();
    S = clock();
    APSP(node,matrix);
    end_time = currentSeconds();
    E = clock();
    float zero=0.0;
    // printf("\n\033[31m [execution time]: \033[0m\t\t[%.3f] ms\n\n",(end_time - start_time)* 1000);
    printf("\n\033[31m [malloc time]: \033[0m\t\t[%.3f] ms\n\n", (zero) * 1000);
    printf("\n\033[31m [mem2cuda time]: \033[0m\t\t[%.3f] ms\n\n", (zero) * 1000);
    printf("\n\033[31m [execution time]: \033[0m\t\t[%.3f] ms\n\n",(end_time - start_time)* 1000);
    printf("\n\033[31m [cuda2mem time]: \033[0m\t\t[%.3f] ms\n\n", (zero) * 1000);
    printf("\n\033[31m [Total time]: \033[0m\t\t[%.3f] ms\n\n",(E-S)/CLOCKS_PER_SEC);
    // output(node,matrix);
    free(matrix);
}
