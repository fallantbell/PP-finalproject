#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "../common/CycleTimer.h"

#define min(X,Y) ((X) < (Y) ? (X):(Y))

float *matrix;
int n,edge;

typedef struct{
    int thread_id;
    int n_num;
    int k;
} Arg;

void input(){
    scanf("%d",&n);
    scanf("%d",&edge);

    matrix = (float*)malloc(sizeof(float)*n*n);

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(i==j){
                *(matrix+i*n+j)=0.0;
            }
            else{
                *(matrix+i*n+j)=100000.0;
            }
            
        }
    }

    for(int i=0;i<edge;i++){
        int n1,n2;
        float v;
        scanf("%d %d %f",&n1,&n2,&v);
        *(matrix+n1*n+n2)=v;
    }
}

void *APSP(void *arg){

    Arg *data = (Arg *)arg;
    int thread_id = data->thread_id;
    int n_num = data->n_num;
    int k = data->k;

    int start = thread_id*n_num;
    int end = (thread_id+1)*n_num;
    if(n-end < n_num){
        end = n;
    }
    // printf("threadid = %d \tstart = %d \tend=%d\n",thread_id,start,end);

    for(int i=start;i<end;i++){
        for(int j=0;j<n;j++){
            float dis = matrix[i*n+k]+matrix[k*n+j];
            if(dis < matrix[i*n+j]){
                matrix[i*n+j] = dis;
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

void output(){
    // printf("yes bro \n");
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            printf("%.2f \t\t",matrix[i*n+j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]){

    double start_time, end_time;
    int thread_num = strtol(argv[1],NULL,10);
    
    pthread_t* p;
    Arg arg[thread_num];
    p = (pthread_t*) malloc (thread_num*sizeof(pthread_t)); 

    input();
    // printf("\033[31m APSP --------------------- \n\n\033[0m");

    start_time = currentSeconds();

    for(int k=0;k<n;k++){
        // create thread
        for(int i=0;i<thread_num;i++){
            arg[i].thread_id = i;
            arg[i].n_num = n/thread_num;
            arg[i].k = k;

            pthread_create(&p[i],NULL,APSP,(void*)&arg[i]);
        }
        for(int i=0;i<thread_num;i++){
            pthread_join(p[i],NULL);
        }
    }

    end_time = currentSeconds();
    printf("\n\033[31m [execution time]: \033[0m\t\t[%.3f] ms\n\n", (end_time-start_time) * 1000);

    output();

    free(matrix);
    free(p);

}