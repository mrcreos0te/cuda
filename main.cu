#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n, int count)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
    count++;
}

int main( int argc, char* argv[] )
{
    cudaError_t error;

    // Size of vectors
    int n = 100;

    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;

    // Device input vectors
    double *d_a;
    double *d_b;
    //Device output vector
    double *d_c;

    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        //h_a[i] = sin(i)*sin(i);
        //h_b[i] = cos(i)*cos(i);
        h_a[i] = 1;
        h_b[i] = 1;
    }

    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n/blockSize);

    std::clock_t start;
    double duration;
    start = std::clock();
    int count = 0;
    printf("executing kernel...\n");
    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n, count);

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC/ 1000;
    std::cout<<"time: "<< duration <<'\n';
    std::cout<<"count: "<< count <<'\n';
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
    printf("3 %s\n",cudaGetErrorString(error));
    }

    // Copy array back to host
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );

    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++){
        sum += h_c[i];
        printf("result: %f\n", h_c[i]);
    }

    printf("final result: %f\n", sum/n);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    system("pause");
}

/*// Kernel definition
__global__ void VecAdd(int* A, int* B)
{
    int* C;
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(int argc, char *argv[])
{
    cudaError_t error;
    const int sz = 100000;
    int randArrayA[sz];
    int randArrayB[sz];
    for(int i=0;i<sz;i++)
        randArrayA[i]=rand()%100;  //Generate number between 0 to 99
    for(int i=0;i<sz;i++)
        randArrayB[i]=rand()%100;  //Generate number between 0 to 99
    int *a;
    int *b;
    a = randArrayA;
    b = randArrayB;
    std::clock_t start;
    double duration;
    start = std::clock();
    VecAdd<<<sz/10000, 1024>>>(a, b);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<"printf: "<< duration <<'\n';
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
    printf("3 %s\n",cudaGetErrorString(error));
    }
    system("pause");
}*/
