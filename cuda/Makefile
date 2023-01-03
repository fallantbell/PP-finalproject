NVCC = nvcc
CC = gcc
CFLAG = -O3 -g -I../common -I./
LFLAG = -lm

# EXE = cuda
EXE = cuda_with_tiling

OBJ = cuda.o

all: ${EXE}

${EXE}: ${OBJ}
	$(NVCC) -o $@ $(OBJ)


# cuda.o: cuda.cu
# 	${NVCC} -c cuda.cu -o $@

cuda.o: cuda_with_tiling.cu
	${NVCC} -c cuda_with_tiling.cu -o $@

.PHONY: clean

clean:
	rm -rf ${EXE} ${OBJ}
