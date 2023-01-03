# PP-finalproject

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

### Result


### Reference
https://www.cise.ufl.edu/~sahni/papers/shortj.pdf  
https://repository.upenn.edu/cgi/viewcontent.cgi?article=1213&context=hms  


