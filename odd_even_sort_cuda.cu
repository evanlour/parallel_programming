#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#include "functions_cuda.h"

const long int ARRAY_SIZE = 1000000;
int main(){
    int* arr = (int*)malloc(ARRAY_SIZE * sizeof(int));
    fill_array(arr, ARRAY_SIZE);
    int min_blocks, ideal_threads;
    cudaOccupancyMaxPotentialBlockSize(&min_blocks, &ideal_threads, odd_even_sort_iteration, 0, 0);
    printf("The minimal number of blocks required is %d and the optimal amount of threads is %d.\n", min_blocks, ideal_threads);
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    odd_even_sort_cuda(arr, ARRAY_SIZE, ideal_threads);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_elapsed = 0;
    cudaEventElapsedTime(&time_elapsed, start, end);
    check_sorted(arr, ARRAY_SIZE);
    printf("The sorting took %.4f ms.\n", time_elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    free(arr);
    return 0;
}
