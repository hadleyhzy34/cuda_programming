test: array_2.o common.o
	nvcc -o test array_2.o common.o

array_2.o  : array_2.cu common.h
			nvcc -c array_2.cu

common.o : common.cpp common.h
			nvcc -c common.cpp
clean:
	rm *.o
