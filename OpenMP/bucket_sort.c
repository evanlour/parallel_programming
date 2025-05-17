#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "functions.h"

const long int ARRAY_SIZE = 1000;
const int CONTAINER_NUM = 10;
double start;
double end;

int main(int argc, char* argv[]){
    srand(time(NULL));

    int container_num = 0;
    int arr_size;
    arr_size = (argc > 1) ? atoi(argv[1]) : ARRAY_SIZE;
    container_num = (argc > 2) ? atoi(argv[2]) : CONTAINER_NUM;

    FILE *file = fopen("bucket_sort_log.csv", "a");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    if (ftell(file) == 0) { // If file is empty, write headers
        fprintf(file, "Threads,Array_Size,Execution_Time(s),Method,Containers\n");
    }

    int *array_thread = (int *)malloc(arr_size * sizeof(int));
    int *array_thread1 = (int *)malloc(arr_size * sizeof(int));
    int *array_thread2 = (int *)malloc(arr_size * sizeof(int));
    int *array_thread4 = (int *)malloc(arr_size * sizeof(int));
    int *array_thread8 = (int *)malloc(arr_size * sizeof(int));
    int *array_thread16 = (int *)malloc(arr_size * sizeof(int));
    int *array_thread32 = (int *)malloc(arr_size * sizeof(int));
    int *array_thread64 = (int *)malloc(arr_size * sizeof(int));

    start = omp_get_wtime();
    fill_array(array_thread, arr_size);
    memcpy(array_thread1, array_thread, arr_size * sizeof(array_thread[0]));
    memcpy(array_thread2, array_thread, arr_size * sizeof(array_thread[0]));
    memcpy(array_thread4, array_thread, arr_size * sizeof(array_thread[0]));
    memcpy(array_thread8, array_thread, arr_size * sizeof(array_thread[0]));
    memcpy(array_thread16, array_thread, arr_size * sizeof(array_thread[0]));
    memcpy(array_thread32, array_thread, arr_size * sizeof(array_thread[0]));
    memcpy(array_thread64, array_thread, arr_size * sizeof(array_thread[0]));
    end = omp_get_wtime();
    printf("Time taken to fill and duplicate the arrays: %f.\n", end - start);

    start = omp_get_wtime();
    merge_sort_array(array_thread, arr_size);
    end = omp_get_wtime();
    double final_time = end - start;
    printf("The array was sorted serially using merge_sort in %f. This array will be used for validation.\n", final_time);
    fprintf(file, "%d,%d,%f,%s\n", 1, arr_size, final_time, "Merge_sort");

    run_and_log_bucket(array_thread1, array_thread, arr_size, container_num, 1, file);
    run_and_log_bucket(array_thread2, array_thread, arr_size, container_num, 2, file);
    run_and_log_bucket(array_thread4, array_thread, arr_size, container_num, 4, file);
    run_and_log_bucket(array_thread8, array_thread, arr_size, container_num, 8, file);
    run_and_log_bucket(array_thread16, array_thread, arr_size, container_num, 16, file);
    run_and_log_bucket(array_thread32, array_thread, arr_size, container_num, 32, file);
    run_and_log_bucket(array_thread64, array_thread, arr_size, container_num, 64, file);

    fclose(file);

    free(array_thread);
    free(array_thread1);
    free(array_thread2);
    free(array_thread4);
    free(array_thread8);
    free(array_thread16);
    free(array_thread32);
    free(array_thread64);
    return 0;
}
