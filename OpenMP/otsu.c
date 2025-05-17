#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>
#include "functions.h"
//https://www.pexels.com/photo/reflection-of-mountain-on-lake-braies-1525041/ The source of the image

int main(int argc, char* argv[]){
    // int threads = (argc > 1) ? atoi(argv[1]) : 1;

    FILE *file = fopen("otsu.csv", "a");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    if (ftell(file) == 0) { // If file is empty, write headers
        fprintf(file, "Threads,Execution_Time(s)\n");
    }

    Image image;
    read_image("img.jpg", &image, 0);
    //get_image_info(&image);
    convert_to_grayscale(&image, 16);
    run_and_log_otsu(&image, 1, file);
    run_and_log_otsu(&image, 2, file);
    run_and_log_otsu(&image, 4, file);
    run_and_log_otsu(&image, 8, file);
    run_and_log_otsu(&image, 16, file);
    run_and_log_otsu(&image, 32, file);
    run_and_log_otsu(&image, 64, file);
    free_image(&image);
    fclose(file);
    return 0;
}
