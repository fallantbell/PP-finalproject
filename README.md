# PP-finalproject

### Introduction
This is nycu 2022 parallel programming final project.  
In this project, we use cuda to accelerate the Floyd-warshall algorithm and compare the execution time to the serial one. Moreover, we also take advantage of the tiling technique to reduce global memory access time, which improve the performance furthermore.  

### How to use

1. make file

```
make
```

2. run cuda/cuda_tiling code

```
./cuda < "yourdata" > output_cuda.txt
```

3. check the correctness

```
./serial < "yourdata" > output_serial.txt
diff output_cuda.txt output_serial.txt
```
If there is no different, it is correct.

### Dataset
You can download dataset here  
https://algs4.cs.princeton.edu/44sp/

### Platform

Ubuntu 18.04  
CPU: Intel(R) Xeon(R) Silver 4210  
GPU: RTX 2080ti  
Core: 4352  
CUDA version 11.3  


### Result
- execution time  
![image](https://github.com/fallantbell/PP-finalproject/blob/main/result/execution_time.png)
- speed up  
![image](https://github.com/fallantbell/PP-finalproject/blob/main/result/speedup.png)
- execution time with different tiling size  
![image](https://github.com/fallantbell/PP-finalproject/blob/main/result/different_tiling_size.png)

### Reference
https://www.cise.ufl.edu/~sahni/papers/shortj.pdf  
https://repository.upenn.edu/cgi/viewcontent.cgi?article=1213&context=hms  


