#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

void fill_array(int* arr, int size){//Fill the array will values 1-1000
    for (int i = 0; i < size; i++){
        arr[i] = 1 + rand() % (1000);
    }
}

void print_array(int* arr, int size){//For debugging
    for(int i = 0; i < size; i++){
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void check_sorted(int* arr, int size){
    for(int i = 0; i < size - 1; i++){
        if(arr[i] > arr[i+1]){
            printf("The array is not sorted correctly!\n");
            return;
        }
    }
    printf("The array is sorted correctly!\n");
}

__global__ void odd_even_sort_iteration(int* arr, int size, int phase){
    int index = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + phase;
    if(index + 1 < size){
        if(arr[index] > arr[index + 1]){
            int temp = arr[index];
            arr[index] = arr[index + 1];
            arr[index + 1] = temp;
        }
    }
}

void odd_even_sort_cuda(int* arr, long size, int threads){
    int* dev_arr;
    int blocks = size / threads + 1;
    long int arr_size = size * sizeof(int);
    cudaMalloc((void**)&dev_arr, arr_size);
    cudaMemcpy(dev_arr, arr, arr_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    for(int i = 0; i < size; i++){
        odd_even_sort_iteration<<<blocks,threads>>>(dev_arr, size, 0);
        cudaDeviceSynchronize();
        odd_even_sort_iteration<<<blocks,threads>>>(dev_arr, size, 1);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(arr, dev_arr, arr_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
    cudaDeviceSynchronize();
}
