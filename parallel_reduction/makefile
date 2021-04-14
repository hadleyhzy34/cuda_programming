test: array_syn.o common.o
	nvcc -o test array_syn.o common.o

array_syn.o  : array_syn.cu common.h
			nvcc -c array_syn.cu

common.o : common.cpp common.h
			nvcc -c common.cpp
clean:
	rm *.o
