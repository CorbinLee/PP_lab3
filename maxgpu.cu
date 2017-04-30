#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

unsigned int getmaxcu(unsigned int *, unsigned int);

int main(int argc, char *argv[])
{
    unsigned int size = 0;  // The size of the array
    unsigned int i;  // loop index
    unsigned int * numbers; //pointer to the array
    
    if(argc !=2)
    {
       printf("usage: maxseq num\n");
       printf("num = size of the array\n");
       exit(1);
    }
   
    size = atol(argv[1]);

    numbers = (unsigned int *)malloc(size * sizeof(unsigned int));
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
        getmaxcu(numbers, size));

    free(numbers);
    exit(0);
}


/*
   input: pointer to an array of long int
          number of elements in the array
   output: the maximum number of the array
*/
unsigned int getmaxcu(unsigned int num[], unsigned int size)
{
    unsigned int i;
    //unsigned int * max;
    unsigned int * num_d;
    unsigned int memSize = size * sizeof(unsigned int);

    /* Copy the array from the host to the device */
    cudaMalloc((void **) &num_d, memSize);
    cudaMemcpy(num_d, num, memSize, cudaMemcpyHostToDevice);

    /* Allocate space for the max number */
    //cudaMalloc((void **) &max, sizeof(unsigned int));

    /* Kernel invocation code */
    //foo<<<numBlocks, numThreadsPerBlock>>>(num_d, size);


 //  for(i = 1; i < size; i++)
	// if(num[i] > max)
	//    max = num[i];

 //  return( max );
    cudaFree(num_d);
    
    return 0;

}

/* Each thread will check it's portion of the array and put the max number in the first index */
__global__
void foo(unsigned int * num_d) {

}
