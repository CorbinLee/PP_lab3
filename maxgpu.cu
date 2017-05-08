#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

unsigned int getmaxcu(unsigned int *, unsigned int *, unsigned int);
void foo(unsigned int *, unsigned int *, unsigned int);

int main(int argc, char *argv[])
{
    unsigned int size = 0;  // The size of the array
    unsigned int i;  // loop index
    unsigned int * numbers; //pointer to the array
    unsigned int * max;
    
    if(argc !=2)
    {
       printf("usage: maxseq num\n");
       printf("num = size of the array\n");
       exit(1);
    }
   
    size = atol(argv[1]);

    numbers = (unsigned int *)malloc(size * sizeof(unsigned int));
    max = (unsigned int *)malloc(10 * sizeof(unsigned int));
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
        getmaxcu(numbers, max, size));

    free(numbers);
    exit(0);
}


/* Each thread will check it's portion of the array and put the max number in the first index */
__global__
void foo(unsigned int * num_d, unsigned int * max, unsigned int size) {
    int i = threadIdx.x;
    int chunk = size / 10;
    int j;
    max[i] = 0;

    for (j = i*chunk; j < (i+1) * chunk; j++) {
        if (num_d[j] > max[i])
            max[i] = num_d[j];
    }
}


/*
   input: pointer to an array of long int
          number of elements in the array
   output: the maximum number of the array
*/
unsigned int getmaxcu(unsigned int * num, unsigned int * max, unsigned int size)
{
    unsigned int i;
    unsigned int * max_d;
    unsigned int * num_d;
    unsigned int memSize = size * sizeof(unsigned int);

    /* Copy the array from the host to the device */
    cudaMalloc((void **) &num_d, memSize);
    cudaMemcpy(num_d, num, memSize, cudaMemcpyHostToDevice);

    /* Allocate space for the max number */
    cudaMalloc((void **) &max_d, 10 * sizeof(unsigned int));

    /** 
     * Kernel invocation code
     * 
     * Max MP = 15
     * Max threads = 2048 * 15 = 30720
     * Max threads/block = 1024
     */
    int threads = (30720 > size) ? size : 30720;
    
    int numsPerThread = 0;
    // Find number of threads, rounding up
    //int minNumThreads = (size + numsPerThread - 1) / numsPerThread;
    int blocks = 1;
    int threadsPerBlock = 10;
    foo<<<blocks, threadsPerBlock>>>(num_d, max, size);

    /* Copy the max array from device to host and find the max of those */
    cudaMemcpy(max, max_d, 10 * sizeof(unsigned int), cudaMemcpyDeviceToHost);


    unsigned int maxNum = 0;
    for(i = 1; i < 10; i++)
        if(max[i] > maxNum)
            maxNum = max[i];

    cudaFree(num_d);
    cudaFree(max_d);

    return maxNum;

}
