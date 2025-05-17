#include <stdio.h>
#include <cuda.h>
#include "functions_cuda.h"

const long int ARRAY_SIZE = 14335;
int main(int argc, char* argv[]){
    //Obtain useful data
    int arr_size;
    if(argc > 1){
        arr_size = atoi(argv[1]);
    }else{
        arr_size = ARRAY_SIZE;
    }

    get_device_cuda_info();
    int* arr = (int*)malloc(arr_size * sizeof(int));
    fill_array(arr, arr_size);
    int max_blocks, max_threads;
    receive_max_capabilities_odd_even_sort(&max_blocks, &max_threads);
    printf("The min number of blocks is %d and the max number of threads is %d. The max array size is %d for the fast algorithm\n", max_blocks, max_threads, max_blocks * max_threads);
    printf("Currently running with an array size of %d\n", arr_size);
    //Start timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    if(int(floor(arr_size / max_threads)) < max_blocks){
        odd_even_sort_cuda_fast(arr, arr_size, max_threads, max_blocks);
    }else{
        odd_even_sort_cuda_slow(arr, arr_size, max_threads);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_elapsed = 0;
    cudaEventElapsedTime(&time_elapsed, start, end);
    //Stop timing

    check_sorted(arr, arr_size);
    printf("The sorting took %.4f ms\n", time_elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    free(arr);
    return 0;
}
