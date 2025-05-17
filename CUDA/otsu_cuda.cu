#include <stdio.h>
#include <cuda.h>
#include "functions_cuda.h"

int main(int argc, char* argv[]){
    //Obtain useful data

    get_device_cuda_info();
    int max_blocks, max_threads;
    receive_max_capabilities_odd_even_sort(&max_blocks, &max_threads);
    printf("The min number of blocks is %d and the max number of threads is %d. The max array size is %d for the fast algorithm\n", max_blocks, max_threads, max_blocks * max_threads);
    // max_blocks = (arr_size + max_threads - 1) / max_threads;
    Image image;
    read_image("img.jpg", &image, 0);
    //get_image_info(&image);
    convert_to_grayscale(&image, 16);
    int img_size = image.width * image.height;
    int threads = 256;
    max_blocks  = (img_size + threads - 1) / threads;
    //Start timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    int found_thresh = run_otsu_cuda(&image, max_threads, max_blocks);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_elapsed = 0;
    cudaEventElapsedTime(&time_elapsed, start, end);
    printf("The threshold found is %d and the elapsed time is %f ms\n", found_thresh, time_elapsed);
    //Stop timing
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    free_image(&image);
    return 0;
}
