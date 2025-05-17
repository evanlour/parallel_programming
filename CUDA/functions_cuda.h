#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include "stb_image.h"
#include "stb_image_write.h"
#include <string.h>

typedef struct {
    unsigned char* data;
    char extension[8];
    int width;
    int height;
    int channels;
} Image;

void get_device_cuda_info(){
    int devices;
    cudaGetDeviceCount(&devices);
    for(int i = 0; i < devices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device name: %s\n", prop.name);
        printf("Device driver version: %d.%d\n", prop.major, prop.minor);
        printf("Total GPU memory (GB): %d\n", int(round(prop.totalGlobalMem / (1024 * 1024 * 1024))));
        printf("Total GPU L2 cache (KB): %d\n", int(prop.l2CacheSize / 1024));
        printf("Shared memory per block (KB): %d\n", int(prop.sharedMemPerBlock / 1024));
        printf("Number of multiprocessors available: %d\n", prop.multiProcessorCount);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max number of blocks: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("Max number of threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Memory Clock Rate (GHz): %.2f\n", float(prop.memoryClockRate / 1000000));
        printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("Peak Memory Bandwidth (GB/s): %.2f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}

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

void odd_even_sort_cuda_slow(int* arr, long size, int threads){
    int* dev_arr;
    int blocks = size / threads + 1;
    long int arr_size = size * sizeof(int);
    cudaMalloc((void**)&dev_arr, arr_size);
    cudaMemcpy(dev_arr, arr, arr_size, cudaMemcpyHostToDevice);
    for(int i = 0; i < size; i++){
        odd_even_sort_iteration<<<blocks,threads>>>(dev_arr, size, 0);
        odd_even_sort_iteration<<<blocks,threads>>>(dev_arr, size, 1);
    }
    cudaMemcpy(arr, dev_arr, arr_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
}

namespace cg = cooperative_groups;
__global__ void odd_even_sort_cuda_fast_kernel(int *arr, int size){
    cg::grid_group grid = cg::this_grid();
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < size; i++){
        if(index + 1 < size){
            if(arr[index] > arr[index + 1]){
                int temp = arr[index];
                arr[index] = arr[index + 1];
                arr[index + 1] = temp;
            }
        }
        grid.sync();
        if(index + 2 < size){
            if(arr[index + 1] > arr[index + 2]){
                int temp = arr[index + 1];
                arr[index + 1] = arr[index + 2];
                arr[index + 2] = temp;
            }
        }
        grid.sync();
    }
}

void odd_even_sort_cuda_fast(int *arr, int size, int threads, int blocks){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if(!prop.cooperativeLaunch){
        fprintf(stderr, "Cooperative kernels not supported on this device.\n");
        exit(1);
    }
    int* dev_arr;
    long int arr_size = size * sizeof(int);
    cudaMalloc((void**)&dev_arr, arr_size);
    cudaMemcpy(dev_arr, arr, arr_size, cudaMemcpyHostToDevice);
    void *args[] = {&dev_arr, &size};
    cudaError_t err = cudaLaunchCooperativeKernel((void*)odd_even_sort_cuda_fast_kernel, blocks, threads, args, 0, nullptr);
    if(err != cudaSuccess){
        fprintf(stderr, "Kernel failed to launch: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaMemcpy(arr, dev_arr, arr_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
    cudaDeviceSynchronize();
}

void receive_max_capabilities_odd_even_sort(int *max_b, int *max_t){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    *max_t = prop.maxThreadsPerBlock;
    int blocks_per_mp;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_mp, odd_even_sort_cuda_fast, *max_t, 0);
    *max_b = blocks_per_mp * prop.multiProcessorCount;
}

__global__ void find_array_max(int* arr, int arr_size, int* max_val){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < arr_size){
        atomicMax(max_val, arr[index]);
    }
}

void radix_sort(int* device_arr, int size){
    int* temp_arr;
    cudaMalloc((void**)&temp_arr, size * sizeof(int));

    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        temp_storage, temp_storage_bytes,
        device_arr, temp_arr, size
    );
    cudaMalloc(&temp_storage, temp_storage_bytes);
        cub::DeviceRadixSort::SortKeys(
        temp_storage, temp_storage_bytes,
        device_arr, temp_arr, size
    );
    cudaMemcpy(device_arr, temp_arr, size * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaFree(temp_arr);
    cudaFree(temp_storage);
}

__global__ void bucket_sort_identify_buckets(int* arr, int* bucket_sizes, int size, int max_val, int buckets){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= size){return;};
    int divider = max((max_val / buckets), 1);
    atomicAdd(&bucket_sizes[arr[index] / divider], 1);
}

__global__ void bucket_sort_create_buckets(int* arr, int* out_arr, int size, int* bucket_indexes, int bucket_limiter, int buckets){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= size){return;};
    int value = arr[index];
    int bucket = min(value / bucket_limiter, buckets - 1);
    int new_pos = atomicAdd(&bucket_indexes[bucket], 1);
    out_arr[new_pos] = value;
}

void bucket_sort_cuda(int* arr, int size, int buckets, int blocks, int threads){
    //Stage 0: Initializing
    int counter_arr[buckets];
    int counter_arr_helper[buckets];
    for(int i = 0; i < buckets; i++){
        counter_arr[i] = 0;
        counter_arr_helper[i] = 0;
    }

    int *dev_arr;
    int *dev_binned_arr;
    int *dev_counter_arr;
    int *dev_bucket_indexes;
    long int arr_size = size * sizeof(int);
    int counter_arr_size = buckets * sizeof(int);
    cudaMalloc((void**)&dev_arr, arr_size);
    cudaMalloc((void**)&dev_binned_arr, arr_size);
    cudaMalloc((void**)&dev_counter_arr, counter_arr_size);
    cudaMalloc((void**)&dev_bucket_indexes, counter_arr_size);
    cudaMemcpy(dev_arr, arr, arr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_binned_arr, arr, arr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_counter_arr, counter_arr, counter_arr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bucket_indexes, counter_arr_helper, counter_arr_size, cudaMemcpyHostToDevice);
    //Stage 1: Calculating bucket sizes;
    int max_val = INT_MIN;
    int *dev_max_val;
    cudaMalloc(&dev_max_val, sizeof(int));
    cudaMemcpy(dev_max_val, &max_val, sizeof(int), cudaMemcpyHostToDevice);
    find_array_max<<<blocks, threads>>>(dev_arr, size, dev_max_val);
    cudaMemcpy(&max_val, dev_max_val, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_max_val);
    bucket_sort_identify_buckets<<<blocks, threads>>>(dev_arr, dev_counter_arr, size, max_val, buckets);
    cudaMemcpy(counter_arr, dev_counter_arr, counter_arr_size, cudaMemcpyDeviceToHost);
    //Stage 2: Creating the buckets inside an array and creating the index points
    int bucket_indexes[buckets];
    bucket_indexes[0] = 0;
    for(int i = 1; i < buckets; i++){
        bucket_indexes[i] = bucket_indexes[i - 1] + counter_arr[i - 1];
    }
    cudaMemcpy(dev_bucket_indexes, bucket_indexes, counter_arr_size, cudaMemcpyHostToDevice);
    //Stage 3: Create the buckets
    int bucket_limiter = max(max_val / buckets, 1);
    bucket_sort_create_buckets<<<blocks, threads>>>(dev_arr, dev_binned_arr, size, dev_bucket_indexes, bucket_limiter, buckets);
    cudaMemcpy(bucket_indexes, dev_bucket_indexes, counter_arr_size, cudaMemcpyDeviceToHost); //<=====
    //thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(dev_binned_arr);
    //thrust::sort(dev_ptr, dev_ptr + bucket_indexes[0]);
    radix_sort(dev_binned_arr, bucket_indexes[0]);
    for(int i = 0; i < buckets-1; i++){
        radix_sort(dev_binned_arr + bucket_indexes[i], bucket_indexes[i + 1] - bucket_indexes[i]);
        //thrust::sort(dev_ptr + bucket_indexes[i - 1], dev_ptr + bucket_indexes[i]);
    }
    //Stage 4: Cleanup
    cudaMemcpy(arr, dev_binned_arr, arr_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
    cudaFree(dev_binned_arr);
    cudaFree(dev_counter_arr);
    cudaFree(dev_bucket_indexes);
}

void read_image(const char* path, Image* image, int grayscale){
    const char* dot = strrchr(path, '.');
    if(!dot || dot == path){
        printf("No extension found\n");
    }
    char extension[8];
    strncpy(extension, dot + 1, sizeof(extension) - 1);
    extension[sizeof(extension) - 1] = '\0';
    for(int i = 0; extension[i]; i++){
        extension[i] = tolower((unsigned char)extension[i]);
    }
    if(grayscale){
        image->data = stbi_load(path, &image->width, &image->height, NULL, 1);
        image->channels = 1;
    }else{
        image->data = stbi_load(path, &image->width, &image->height, NULL, 3);
        image->channels = 3;
    }
    if(image->data == NULL){
        printf("Failed to load image: %s\n", stbi_failure_reason());
        return;
    }
    strncpy(image->extension, extension, sizeof(image->extension) - 1);
}

void get_image_info(Image* image){
    printf("Image dimensions are %dx%d, has %d channel(s) and is of type %s\n", image->width, image->height, image->channels, image->extension);
}

void convert_to_grayscale(Image* image, int threads){
    unsigned char* grayscale_data = (unsigned char*)malloc(image->width * image->height);
    for (int i = 0; i < image->width * image->height; i++){
        int r = image->data[i * 3 + 0];
        int g = image->data[i * 3 + 1];
        int b = image->data[i * 3 + 2];
        grayscale_data[i] = (unsigned char)(0.3 * r + 0.59 * g + 0.11 * b);
    }
    free(image->data);
    image->data = grayscale_data;
    image->channels = 1;
}

void save_image(Image* image, const char* path){
    stbi_write_png(path, image->width, image->height, image->channels, image->data, image->width * image->channels);
}

void free_image(Image* image){
    stbi_image_free(image->data);
    image->data = NULL;
}

__global__ void calculate_histogram(unsigned char* data, int* histogram, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size){
        atomicAdd(&histogram[int(data[index])], 1);
    }
}

__global__ void calculate_means(int* precalculated_sums, long long* precalculated_intensities, float* meanB, float* meanF){
    int index = threadIdx.x;
    if(index < 256){
        meanB[index] = (precalculated_sums[index] != 0) ? (double)precalculated_intensities[index] / precalculated_sums[index] : 0.0;
        int foreground_sum = precalculated_sums[255] - precalculated_sums[index];
        meanF[index] = (foreground_sum != 0) ? ((double)precalculated_intensities[255] - precalculated_intensities[index]) / foreground_sum : 0.0;
    }
}

__global__ void find_threshold(int* precalculated_sums, float* meanB, float* meanF, float* variances, int total_pixels){
    int index = threadIdx.x;
    if(index < 256){
        double diff = meanB[index] - meanF[index];
        float variance = (precalculated_sums[index] / (float)total_pixels) * ((precalculated_sums[255] - precalculated_sums[index]) / (float) total_pixels) * diff * diff;
        variances[index] = variance;
    }
}

int run_otsu_cuda(Image *image, int threads, int blocks){
    //Stage 1: Creating the variables and passing them to gpu
    int array_sizes = 256;
    unsigned char* dev_data;
    int histogram[array_sizes] = {0};
    int precalculated_sums[array_sizes] = {0};
    long long precalculated_intensities[array_sizes] = {0};
    float meanB[256] = {0.0};
    float meanF[256] = {0.0};
    int* dev_histogram;
    int* dev_precalculated_sums;
    long long* dev_precalculated_intensities;
    float* dev_meanB;
    float* dev_meanF;

    cudaMalloc((void**)&dev_data, image->height * image->width * sizeof(unsigned char));
    cudaMalloc((void**)&dev_histogram, array_sizes * sizeof(int));
    cudaMalloc((void**)&dev_precalculated_sums, array_sizes * sizeof(int));
    cudaMalloc((void**)&dev_precalculated_intensities, array_sizes * sizeof(long long));
    cudaMalloc((void**)&dev_meanB, array_sizes * sizeof(float));
    cudaMalloc((void**)&dev_meanF, array_sizes * sizeof(float));
    cudaMemcpy(dev_data, image->data, image->width * image->height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_histogram, histogram, array_sizes * sizeof(int), cudaMemcpyHostToDevice);

    //Stage 2: The method begins
    calculate_histogram<<<blocks, threads>>>(dev_data, dev_histogram, image->height * image->width);
    cudaMemcpy(histogram, dev_histogram, array_sizes * sizeof(int), cudaMemcpyDeviceToHost);
    precalculated_sums[0] = histogram[0];
    precalculated_intensities[0] = 0;
    for(int i = 1; i < 256; i++){
        precalculated_sums[i] = precalculated_sums[i - 1] + histogram[i];
        precalculated_intensities[i] = precalculated_intensities[i - 1] + (long long)i * histogram[i];
    }

    cudaMemcpy(dev_precalculated_sums, precalculated_sums, array_sizes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_precalculated_intensities, precalculated_intensities, array_sizes * sizeof(long long), cudaMemcpyHostToDevice);
    calculate_means<<<1, 256>>>(dev_precalculated_sums, dev_precalculated_intensities, dev_meanB, dev_meanF);

    float variances[256] = {0.0};
    float* dev_variances;
    cudaMalloc((void**)&dev_variances, 256 * sizeof(float));
    cudaMemcpy(dev_variances, variances, 256 * sizeof(float), cudaMemcpyHostToDevice);
    find_threshold<<<1, 256>>>(dev_precalculated_sums, dev_meanB, dev_meanF, dev_variances, image->height * image ->width);
    cudaMemcpy(variances, dev_variances, 256 * sizeof(float), cudaMemcpyDeviceToHost);
    int best_threshold = 0;
    float best_variance = variances[0];
    for(int t = 1; t < 256; ++t){
        if(variances[t] > best_variance){
            best_variance = variances[t];
            best_threshold = t;
        }
    }
    //Stage 3: Cleanup
    cudaFree(dev_variances);
    cudaFree(dev_data);
    cudaFree(dev_histogram);
    cudaFree(dev_precalculated_sums);
    cudaFree(dev_precalculated_intensities);
    cudaFree(dev_meanB);
    cudaFree(dev_meanF);
    return best_threshold;
}