test: array_1.o common.o
	nvcc -o test array_1.o common.o

array_1.o  : array_1.cu common.h
			nvcc -c array_1.cu

common.o : common.cpp common.h
			nvcc -c common.cpp
clean:
	rm *.o
