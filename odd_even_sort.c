#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include "functions.h"

const long int ARRAY_SIZE = 10000;
double start;
double end;

int main(int argc, char* argv[]){
    srand(time(NULL));

    int arr_size = 0;
    if(argv[1] != NULL){
        arr_size = atoi(argv[1]);
    }else{
        arr_size = ARRAY_SIZE;
    }

    FILE *file = fopen("odd_even_sort_log.csv", "a");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    if (ftell(file) == 0) { // If file is empty, write headers
        fprintf(file, "Threads,Array_Size,Execution_Time(s),Method\n");
    }

    int *array_thread = (int *)malloc(arr_size * sizeof(int));
    int *array_thread1 = (int *)malloc(arr_size * sizeof(int));
    int *array_thread2 = (int *)malloc(arr_size * sizeof(int));
    int *array_thread4 = (int *)malloc(arr_size * sizeof(int));
    int *array_thread8 = (int *)malloc(arr_size * sizeof(int));
    int *array_thread16 = (int *)malloc(arr_size * sizeof(int));

    start = omp_get_wtime();
    fill_array(array_thread, arr_size);
    memcpy(array_thread1, array_thread, arr_size * sizeof(array_thread[0]));
    memcpy(array_thread2, array_thread, arr_size * sizeof(array_thread[0]));
    memcpy(array_thread4, array_thread, arr_size * sizeof(array_thread[0]));
    memcpy(array_thread8, array_thread, arr_size * sizeof(array_thread[0]));
    memcpy(array_thread16, array_thread, arr_size * sizeof(array_thread[0]));
    end = omp_get_wtime();
    printf("Time taken to fill and duplicate the arrays: %f.\n", end - start);

    start = omp_get_wtime();
    serial_bubble_sort(array_thread, arr_size);
    end = omp_get_wtime();
    double final_time = end - start;
    printf("The array was sorted using bubblesort with 1 thread in %f. This array will be used for validation.\n", final_time);
    fprintf(file, "%d,%d,%f,%s\n", 1, arr_size, final_time, "Bubblesort");

    run_and_log_data_odd_sort(array_thread1, array_thread, arr_size, 1, file);
    run_and_log_data_odd_sort(array_thread2, array_thread, arr_size, 2, file);
    run_and_log_data_odd_sort(array_thread4, array_thread, arr_size, 4, file);
    run_and_log_data_odd_sort(array_thread8, array_thread, arr_size, 8, file);
    run_and_log_data_odd_sort(array_thread16, array_thread, arr_size, 16, file);

    fclose(file);

    free(array_thread);
    free(array_thread1);
    free(array_thread2);
    free(array_thread4);
    free(array_thread8);
    free(array_thread16);
    return 0;
}