#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <limits.h>
#include <math.h>

typedef struct{//This is a dynamic array that works like a vector: It expands when needed
    int size;
    int capacity;
    int* values;
} dynamic_array;

typedef struct{//This struct simulates a stack component
    int** arrays;
    int* array_sizes;
    int size;
    int capacity;
} stack;

void init_stack(stack* st, int capacity){
    st->arrays =  (int**)malloc(sizeof(int*) * (size_t)capacity);
    st->array_sizes = (int *)malloc(sizeof(int) * (size_t)capacity);
    st->size = 0;
    st->capacity = capacity;
}

void push_stack(stack* st, int* arr, int size){
    if(st->size >= st->capacity){
        printf("Stack memory overflow!\n");
        exit(1);
    }
    st->arrays[st->size] = arr;
    st->array_sizes[st->size] = size;
    st->size++;
}

int* pop_stack(stack* st) {
    if (st->size == 0) {
        printf("Stack underflow!\n");
        exit(1);
    }
    st->size--;
    return st->arrays[st->size];
}

void delete_stack(stack* st) {
    for (int i = 0; i < st->size; i++) {
        if(st->arrays != NULL){
            free(st->arrays[i]);
        }
    }

    if(st->arrays != NULL){
        free(st->arrays);
    }
    if(st->array_sizes != NULL){
        free(st->array_sizes);
    }
}

void init_dynamic_array(dynamic_array* arr){
    arr->size = 0;
    arr->capacity = 10;
    arr->values = (int *)malloc(arr->capacity * sizeof(int));
}

void insert_to_dynamic_array(dynamic_array* arr, int value){
    if(arr->size == arr->capacity){
        arr->capacity = arr->capacity * 2;
        arr->values = (int *)realloc(arr->values, arr->capacity * sizeof(int));
        if(arr->values == NULL){
           printf("ERROR HERE\n"); 
            exit(1);
        }
    }
    arr->values[arr->size] = value;
    arr->size++;
}

void delete_dynamic_array(dynamic_array* arr){
    if(arr->values != NULL){
        free(arr->values);
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

int find_array_max(int* arr, int size){//Find the max of array to initialize the container ranges
    int max = INT_MIN;
    for(int i = 0; i < size; i++){
        if(arr[i] > max){
            max = arr[i];
        }
    }
    return max;
}

char* compare_arrays(int *array_thread1, int *array_thread2, int size){//Compare arrays to see if they are the same
    for (int i = 0; i < size; i++){
        if(array_thread1[i] != array_thread2[i]){
            print_array(array_thread1, size);
            print_array(array_thread2, size);
            return "Arrays are not the same!";
        }
    }
    return "Arrays are the same!";
}

void serial_bubble_sort(int *arr, int size){//Serial implementation of bubble_sort
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size - 1; j++){
            if(arr[j] > arr[j + 1]){
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void odd_even_sort(int *arr, int size, int threads){//Parallel implementation of odd-even-sort
    int flag;
    do{
        flag = 0;
        #pragma omp parallel shared(arr) num_threads(threads)
        {
            int local_flag = 0;
            #pragma omp for
            for(int i = 1; i < size - 1; i += 2){
                if(arr[i] > arr[i + 1]){
                    local_flag = 1;
                    int temp = arr[i];
                    arr[i] = arr[i + 1];
                    arr[i + 1] = temp;
                }
            }

            // #pragma omp atomic
            // flag |= local_flag;

            #pragma omp for
            for(int i = 0; i < size - 1; i += 2){
                if(arr[i] > arr[i + 1]){
                    local_flag = 1;
                    int temp = arr[i];
                    arr[i] = arr[i + 1];
                    arr[i + 1] = temp;
                }
            }
            #pragma omp atomic
            flag |= local_flag;
        }
    } while(flag);
}

int* merge_sort(int* arr1, int* arr2, int size1, int size2){//Serial implementation of merge_sort
    if(size1 == 0) return arr2; // or return a copy of arr2 if needed
    if(size2 == 0) return arr1; // similarly, if size2 is 0
    int pointer1 = 0;
    int pointer2 = 0;
    int* res = (int *)malloc((size1 + size2) * sizeof(int));
    if (!res) {
        // Handle memory allocation failure
        printf("Memory allocation error\n");
        exit(1);
    }
    do{
        if(arr1[pointer1] < arr2[pointer2]){
            res[pointer1 + pointer2] = arr1[pointer1];
            pointer1++;
        }else{
            res[pointer1 + pointer2] = arr2[pointer2];
            pointer2++;
        }
    }while(pointer1 < size1 && pointer2 < size2);

    for(int i = pointer1; i < size1; i++){
        res[i + pointer2] = arr1[i];
    }
    for(int i = pointer2; i < size2; i++){
        res[i + pointer1] = arr2[i];
    }
    return res;
}

void merge_sort_array(int* arr, int size){
    if(size <= 1){return;};
    int half = size / 2;
    int* left = (int *)malloc(sizeof(int) * half);
    int* right = (int *)malloc(sizeof(int) * (size - half));
    for(int i = 0; i < half; i++){
        left[i] = arr[i];
    }
    for(int i = 0; i < size - half; i++){
        right[i] = arr[half + i];
    }
    merge_sort_array(left, half);
    merge_sort_array(right, size - half);
    int* results = merge_sort(left, right, half, size - half);
    for(int i = 0; i < size; i++){
        arr[i] = results[i];
    }
    free(left);
    free(right);
    free(results);
}

void sort_container(dynamic_array *arr){
    merge_sort_array(arr->values, arr->size);
}

void bucket_sort(int *arr, int size, int container_num, int threads){//Parallel implementation of bucket_sort
    //Step 1: Create the arrays
    dynamic_array* containers = (dynamic_array *)malloc(sizeof(dynamic_array) * container_num);
    for(int i = 0; i < container_num; i++){
        init_dynamic_array(&containers[i]);
    }
    int max = find_array_max(arr, size);
    int bucket_range = (max / container_num) + 1;

    //Step 2: Distribute to containers
    #pragma omp parallel num_threads(threads)
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        // Each thread has a local set of containers
        dynamic_array* local_containers = (dynamic_array*)malloc(sizeof(dynamic_array) * container_num);
        for(int i = 0; i < container_num; i++) {
            init_dynamic_array(&local_containers[i]);
        }

        // Distribute elements into local containers
        for(int i = thread_id; i < size; i += num_threads) {
            insert_to_dynamic_array(&local_containers[arr[i] / bucket_range], arr[i]);
        }

        // Merge local containers into global containers
        #pragma omp critical
        {
            for(int i = 0; i < container_num; i++) {
                // Merge each local container into the corresponding global container
                for(int j = 0; j < local_containers[i].size; j++) {
                    insert_to_dynamic_array(&containers[i], local_containers[i].values[j]);
                }
            }
        }

        // Clean up local containers
        for(int i = 0; i < container_num; i++) {
            delete_dynamic_array(&local_containers[i]);
        }
        free(local_containers);
    }



    //Step 3: Sort the containers and save arrays to a stack
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for
        for(int i = 0; i < container_num; i++){
            sort_container(&containers[i]);
        }
    }
    
    int offset = 0;
    for(int i = 0; i < container_num; i++){
        if (containers[i].values != NULL && containers[i].size > 0) { // Avoid NULL dereference
            memcpy(&arr[offset], containers[i].values, containers[i].size * sizeof(int));
            offset += containers[i].size; // Corrected indexing
        }
    }

    //Step 5: clear the containers
    for(int i = 0; i < container_num; i++){
        if (containers[i].values != NULL) {
            delete_dynamic_array(&containers[i]);
        }
    }
    free(containers);
}

void run_and_log_bucket(int* array, int* val_array, int size, int container_num, int threads, FILE *file){
    double start, end;
    start = omp_get_wtime();
    bucket_sort(array, size, container_num, threads);
    end = omp_get_wtime();
    double final_time = end - start;
    char* validate = compare_arrays(array, val_array, size);
    printf("The array was sorted using bucket sort with a number of thread(s) of %d  %f. %s\n", threads, final_time, validate);
    fprintf(file, "%d,%d,%f,%s,%d\n", threads, size, final_time, "Bucket_sort", container_num);
}

void run_and_log_data_odd_sort(int* array, int* val_array, int size, int threads, FILE *file){
    double start, end;
    start = omp_get_wtime();
    odd_even_sort(array, size, threads);
    end = omp_get_wtime();
    double final_time = end - start;
    char* validate = compare_arrays(array, val_array, size);
    printf("The array was sorted using odd-even sort with a number of thread(s) of %d  %f. %s\n", threads, final_time, validate);
    fprintf(file, "%d,%d,%f,%s\n", threads, size, final_time, "Odd_even_sort");
}