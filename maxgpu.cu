#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

unsigned int getmaxcu(unsigned int *, unsigned int *, unsigned int, unsigned int);
__global__
void foo(unsigned int *, unsigned int *, unsigned int, int);

int main(int argc, char *argv[])
{
    unsigned int size = 0;  // The size of the array
    unsigned int i;  // loop index
    unsigned int * numbers; //pointer to the array
    unsigned int * max;
    unsigned int numThreads;
    unsigned int numsPerThread;
    
    if(argc !=2)
    {
       printf("usage: maxseq num\n");
       printf("num = size of the array\n");
       exit(1);
    }
   
    size = atol(argv[1]);

    /* Give each thread 1000 numbers */
    numsPerThread = 1000;

    /* I'm assuming that size is divisible by 1000 since that's the case for all the tests for the writeup */
    numThreads = size / numsPerThread;

    numbers = (unsigned int *)malloc(size * sizeof(unsigned int));
    max = (unsigned int *)malloc(numThreads * sizeof(unsigned int));
    if( !numbers )
    {
       printf("Unable to allocate mem for an array of size %u\n", size);
       exit(1);
    }    

    srand(time(NULL)); // setting a seed for the random number generator
    // Fill-up the array with random numbers from 0 to size-1 
    for( i = 0; i < size; i++)
       numbers[i] = rand()  % size;    
   
    printf(" The maximum number in the array is: %u\n", 
        getmaxcu(numbers, max, size, numThreads));

    free(numbers);
    exit(0);
}


/*
   input: pointer to an array of long int
          number of elements in the array
   output: the maximum number of the array
*/
unsigned int getmaxcu(unsigned int * num, unsigned int * max, unsigned int size, unsigned int numThreads)
{
    unsigned int i;
    unsigned int * max_d;
    unsigned int * num_d;
    unsigned int memSize = size * sizeof(unsigned int);

    /* Copy the array from the host to the device */
    cudaMalloc((void **) &num_d, memSize);
    cudaMemcpy(num_d, num, memSize, cudaMemcpyHostToDevice);

    /* Allocate space for the max number */
    cudaMalloc((void **) &max_d, numThreads * sizeof(unsigned int));

    /** 
     * Kernel invocation code
     * 
     * Max MP = 15
     * Max threads = 2048 * 15 = 30720
     * Max threads/block = 1024
     */
    //int threads = (30720 > size) ? size : 30720;
    
    /* Find number of threads, rounding up */
    //int minNumThreads = (size + numsPerThread - 1) / numsPerThread;

    int threadsPerBlock = (numThreads > 1000) ? 1000 : numThreads;
    int blocks = numThreads / threadsPerBlock;
    int chunk = size / numThreads; // How many numbers each thread will process 

    /* Invoke the kernel */
    foo<<<blocks, threadsPerBlock>>>(num_d, max_d, size, chunk);

    /* Copy the max array from device to host and find the max of those */
    cudaMemcpy(max, max_d, numThreads * sizeof(unsigned int), cudaMemcpyDeviceToHost);


    unsigned int maxNum = 0;
    unsigned int cur;
    for(i = 0; i < numThreads; i++) {
        cur = max[i];
        if(cur > maxNum)
            maxNum = cur;
    }

    cudaFree(num_d);
    cudaFree(max_d);

    return maxNum;

}

/* Each thread will check it's portion of the array and put the max number in the first index */
__global__
void foo(unsigned int * num_d, unsigned int * max_d, unsigned int size, int chunk) {
    int i = threadIdx.x;
    int j;
    int start = i * chunk;
    int end = (i+1) * chunk;

    /* Local variables to reduce global memory reads */
    unsigned int cur;
    unsigned int maxNum = 0;


    for (j = start; j < end; j++) {
        cur = num_d[j];
        if (cur > maxNum)
            maxNum = cur;
    }

    max_d[i] = maxNum;
}


