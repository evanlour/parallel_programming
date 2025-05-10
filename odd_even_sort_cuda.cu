#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#include "functions_cuda.h"

const int THREADS_PER_BLOCK = 512;
const long int ARRAY_SIZE = 30000;
int main(){
    int* arr = (int*)malloc(ARRAY_SIZE * sizeof(int));
    fill_array(arr, ARRAY_SIZE);
    time_t start = time(0);
    odd_even_sort_cuda(arr, ARRAY_SIZE, THREADS_PER_BLOCK);
    time_t end = time(0);
    check_sorted(arr, ARRAY_SIZE);
    printf("%ld\n", end - start);
    free(arr);
    return 0;
}
