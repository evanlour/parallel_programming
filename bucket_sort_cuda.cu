#include <stdio.h>
#include <cuda.h>
#include "functions_cuda.h"

const long int ARRAY_SIZE = 15000;
const int ARRAY_BUCKETS = 500;
int main(int argc, char* argv[]){
    //Obtain useful data
    int arr_size;
    int buckets;
    if(argc > 1){
        arr_size = atoi(argv[1]);
    }else{
        arr_size = ARRAY_SIZE;
    }

    if(argc > 2){
        buckets = atoi(argv[2]);
    }else{
        buckets = ARRAY_BUCKETS;
    }

    get_device_cuda_info();
    int* arr = (int*)malloc(arr_size * sizeof(int));
    int* arr_thrust = (int*)malloc(arr_size * sizeof(int));
    fill_array(arr, arr_size);
    memcpy(arr_thrust, arr, arr_size * sizeof(arr[0]));
    int max_blocks, max_threads;
    receive_max_capabilities_odd_even_sort(&max_blocks, &max_threads);
    printf("The min number of blocks is %d and the max number of threads is %d. The max array size is %d for the fast algorithm\n", max_blocks, max_threads, max_blocks * max_threads);
    printf("Currently running with an array size of %d\n", arr_size);
    max_blocks = (arr_size + max_threads - 1) / max_threads;
    //Start timing
    cudaEvent_t start, end, start_thrust, end_thrust;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&start_thrust);
    cudaEventCreate(&end_thrust);
    cudaEventRecord(start);
    bucket_sort_cuda(arr, arr_size, buckets, max_blocks, max_threads);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_elapsed = 0;
    cudaEventElapsedTime(&time_elapsed, start, end);
    //Stop timing

    check_sorted(arr, arr_size);
    printf("The sorting with the bucket sort algorithm took %.4f ms\n", time_elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaEventRecord(start_thrust);
    int* dev_arr;
    cudaMalloc((void**)&dev_arr, arr_size * sizeof(int));
    cudaMemcpy(dev_arr, arr_thrust, arr_size * sizeof(int), cudaMemcpyHostToDevice);
    thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(dev_arr);
    thrust::sort(dev_ptr, dev_ptr + arr_size);
    cudaEventRecord(end_thrust);
    cudaEventSynchronize(end_thrust);
    cudaEventElapsedTime(&time_elapsed, start_thrust, end_thrust);
    printf("The sorting with the thrust sort algorithm took %.4f ms\n", time_elapsed);
    cudaEventDestroy(start_thrust);
    cudaEventDestroy(end_thrust);

    cudaFree(dev_arr);
    free(arr);
    return 0;
}
