#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>
//#include "cuda.h"
#include "ppm/pnmio.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "ppm/stb_image_resize.h"

#define DEFAULT_PATH "/Users/azazel/Documents/Projects/MIMUW/CUDA/Training/"
#define DEFAULT_EPSILON 1.0
#define DEFAULT_RATE 1.0
#define DEFAULT_EPOCH 1

#define LAYER_COUNT 6
#define INPUT_LAYER_SIZE 4096
#define FIRST_LAYER_SIZE 8192
#define SECOND_LAYER_SIZE 6144
#define THIRD_LAYER_SIZE 3072
#define FOURTH_LAYER_SIZE 1024
#define OUTPUT_LAYER_SIZE 62
#define IMAGE_DIMENTIONS 64*64
#define RANDOM_FLOAT (float)rand() / (float)RAND_MAX

// CUDA defines
#define DIM 1024
#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f
#define SPHERES 100
#define THREADS 256


int layers[LAYER_COUNT] = { INPUT_LAYER_SIZE,
                            FIRST_LAYER_SIZE,
                            SECOND_LAYER_SIZE,
                            THIRD_LAYER_SIZE,
                            FOURTH_LAYER_SIZE,
                            OUTPUT_LAYER_SIZE };

//struct Functions {
    int reLU(int x, int derivative) {
        return derivative ? 1 : max(0, x);
    }

    int softmax(float *input, size_t input_len) {
        assert(input);

        float m = -INFINITY;
        for (size_t i = 0; i < input_len; i++) {
            if (input[i] > m) {
                m = input[i];
            }
        }

        float sum = 0.0;
        for (size_t i = 0; i < input_len; i++) {
            sum += expf(input[i] - m);
        }

        float offset = m + logf(sum);
        for (size_t i = 0; i < input_len; i++) {
            input[i] = expf(input[i] - offset);
        }
	return 0;
    }

//    int max(int a, int b) {
//        return a < b ? b : a;
//    }

    int *toGrayScale(const unsigned char *input, int x_dim, int y_dim) {
        int j = 0;
        int *grayscale = (int*)malloc(IMAGE_DIMENTIONS * sizeof(int));

        for (int i = 0; i < 3 * IMAGE_DIMENTIONS; i += 3) {
            grayscale[j++] = (0.3 * input[i]) + (0.59 * input[i + 1]) + (0.11 * input[i + 2]);
        }

        return grayscale;
    }

double random_double()
{
    return (double)rand() / (double)RAND_MAX;
}

// use cuda
/*
void createInitialWeights(int ***w, int random) {

    w = malloc(LAYER_COUNT * sizeof(int*));

    for (int k = 0; k < LAYER_COUNT; k++) {

        // no weights for input layer
        if (k == 0) {
            continue;
        }

        int size = getLayerSize(k);
        int previous_layer_size = getLayerSize(k - 1);
        int **weigths = malloc(size * sizeof(int*));

        printf("for layer = %d, previous layer size = %d, current layer size = %d/n", k, previous_layer_size, size);
        fflush( stdout );

        for (int i = 0; i < size; i++) {

            weigths[i] = malloc(previous_layer_size * sizeof(int));

            for (int j = 0; j < previous_layer_size; j++)
                // weight for node i in layer k from node j in layer k - 1
                weigths[i][j] = random ? random_double() : 1;
        }

        w[k - 1] = weigths;
    }
}
*/

int getLayerSize(int layer) { return layers[layer]; }

int **createInitialBiases() {
    int **biases = (int**)malloc(LAYER_COUNT * sizeof(int*));

    for (int i = 0; i < LAYER_COUNT; i++) {
        int size = getLayerSize(i);
        biases[i] = (int*)malloc(size * sizeof(int));

        for (int j = 0; j < size; j++)
            // bias for nod j in layer i
            biases[i][j] =  1;
    }

    return biases;
}

/* void train(int **input, int output, int ***weights, int **biases, int layer, int round) {
    int *activation_outputs = (int*)malloc(INPUT_LAYER_SIZE * sizeof(int));
    int size = getLayerSize(layer);

    // For input layer
    if (layer == 0) {
        for (int j = 0; j < size; j++) {
            activation_outputs[j] = input[layer][j];
        }
        return;
    }

    int sum = 0;
    int previous_layer_size = getLayerSize(layer - 1);
    int *results = (int*)malloc(size * sizeof(int));

    printf("previous layer size = %d, current layer size = %d", previous_layer_size, size);
    fflush( stdout );
    // For output layer
    if (layer == LAYER_COUNT - 1) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < previous_layer_size; j++) {
                sum = weights[layer - 1][i][j];// * input[layer - 1][j];
            }
            sum += biases[layer - 1][i];
            input[layer][i] = sum;
        }

        softmax(input, size);
        return;
    }

//    weights[layer - 1];

    // For hidden layers
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < previous_layer_size; j++) {
            sum = weights[layer - 1][i][j];// * input[layer - 1][j];
        }
        sum += biases[layer - 1][i];
        input[layer][i] = sum;

        results[i] = reLU(sum, 0);
    }
}*/

//__global__ random_width_generator_kernel(float **widths) {
//    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
//    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//    widths[i][j] = RANDOM_FLOAT;
//}

__global__ void train_cuda(int* input, float* weights) {

}

void readInputFrom(char *path) {
    DIR* FD;
    struct dirent* in_file;
    FILE    *common_file;
    FILE    *entry_file;
    char    buffer[BUFSIZ];

    FILE *file = fopen("/Users/azazel/Documents/Projects/MIMUW/CUDA/Training/00000/01153_00000.ppm", "rb");
    int x_dim;
    int y_dim;
    int img_colors = 0;
    int is_asci = 0;
    int pnm_type = 0;
    int **biases;

    int isRandom = 1;

    // Read contents of ppm file header
    pnm_type = get_pnm_type(file);
    rewind(file);
    read_ppm_header(file, &x_dim, &y_dim, &img_colors, &is_asci);

    unsigned char *original_data = (unsigned char*)malloc(3 * x_dim * y_dim * sizeof(char));
    unsigned char *resized_data = (unsigned char*)malloc(3 * 64 * 64 * sizeof(char));
    int *grayscale[LAYER_COUNT];
    int output = 0;

	int *image_data = (int*)malloc(3 * x_dim * y_dim * sizeof(int));
    // Read image data and resize it to 64x64px
    read_ppm_data(file, image_data, is_asci);
	for (int i = 0; i < 3 * x_dim * y_dim; i++) {
		original_data[i] = image_data[i];	
}
    stbir_resize_uint8(original_data, x_dim, y_dim, 0, resized_data, 64, 64, 0, 3);

    // Convert to grayscale
    grayscale[0] = toGrayScale(resized_data, x_dim, y_dim);
    for (int i = 1; i < LAYER_COUNT; i++) {
        grayscale[i] = (int*)malloc(getLayerSize(i) * sizeof(int));
    }

    // Initialize weights
//    createInitialWeights(weights, isRandom);

    float *weights[LAYER_COUNT];

    for (int k = 0; k < LAYER_COUNT; k++) {

        // no weights for input layer
        if (k == 0) {
            continue;
        }

        int size = getLayerSize(k);
        int previous_layer_size = getLayerSize(k - 1);
        float *weigth = (float*)malloc(size * sizeof(float));

        printf("for layer = %d, previous layer size = %d, current layer size = %d/n", k, previous_layer_size, size);
        fflush( stdout );

        for (int i = 0; i < size; i++) {

            //weigths[i] = (float*)malloc(previous_layer_size * sizeof(float));

            for (int j = 0; j < previous_layer_size; j++)
//                 weight for node i in layer k from node j in layer k - 1
                weigth[i * getLayerSize(k - 1) + j] = random ? random_double() : 1;
        }

        weights[k - 1] = weigth;
        int block_count = getLayerSize(k) / THREADS;
    }
	int cuda_weigths[LAYER_COUNT];





    biases = createInitialBiases();

    fclose(file);


    for (int i = 0; i < LAYER_COUNT; i++) {
	float *weight;
	cudaMalloc((void**)&weight, getLayerSize(i) * sizeof(float));
	cudaMemcpy(weight, weights[i], getLayerSize(i) * sizeof(float), cudaMemcpyHostToDevice);
	int *cuda_input;
	cudaMalloc((void**)&cuda_input, 4096 * sizeof(int));
	cudaMemcpy(cuda_input, grayscale[0], 4096 * sizeof(int), cudaMemcpyHostToDevice);
	train_cuda<<<getLayerSize(i) / 256, 256>>>(cuda_input, weight);
	cudaMemcpy(weights[i], weight, getLayerSize(i) * sizeof(float), cudaMemcpyDeviceToHost);
    }

    /* Don't forget to close common file before leaving */
    fclose(common_file);


//    free(resized_data);
//    free(original_data);
    free(grayscale);

    // TODO: free for every malloc
    free(weights);
    free(biases);


    /* Openiing common file for writing */
//    common_file = fopen(path, "w");
//    if (common_file == NULL)
//    {
//        fprintf(stderr, "Error : Failed to open common_file - %s\n", strerror(errno));
//        return;
//    }

//    PPM ppm = easyppm_create(141, 142, IMAGETYPE_PPM);
//    easyppm_read(&ppm, "/Users/azazel/Documents/Projects/MIMUW/CUDA/Training/00000/01153_00000.ppm");
//    if (NULL == (FD = opendir(path)))
//    {
//        fprintf(stderr, "Error : Failed to open input directory - %s\n", strerror(errno));
//        fclose(common_file);
//
//        return;
//    }
//    while ((in_file = readdir(FD)))
//    {
//
//        if (!strcmp (in_file->d_name, "."))
//            continue;
//        if (!strcmp (in_file->d_name, ".."))
//            continue;
//        /* Open directory entry file for common operation */

//        PPM ppm = easyppm_create(120, 120, IMAGETYPE_PPM);
//        easyppm_read(&ppm, "/Users/azazel/Documents/Projects/MIMUW/CUDA/Training/00000/01153_00000.ppm");
//
//


//        printf(ppm.image);

//        entry_file = fopen(in_file->d_name, "rw");
//        if (entry_file == NULL)
//        {
//            fprintf(stderr, "Error : Failed to open entry file - %s\n", strerror(errno));
//            fclose(common_file);
//
//            return;
//        }
//
//        /* Doing some struf with entry_file : */
//        /* For example use fgets */
//        while (fgets(buffer, BUFSIZ, entry_file) != NULL)
//        {
//            /* Use fprintf or fwrite to write some stuff into common_file */
//            printf(entry_file);
//        }
//
//        /* When you finish with the file, close it */
//        fclose(entry_file);
//    }
}

int main(int argc, char **argv) {

    char *data_path = DEFAULT_PATH;
    double epsilon = DEFAULT_EPSILON;
    double rate = DEFAULT_RATE;
    int epoch = DEFAULT_EPOCH;

    if (argc % 2 != 0) {
        printf("Not enough arguments passed");
    }

    for (int i = 0; i < argc; i += 2) {
        if (strcmp(argv[i], "--training_data")) {
            data_path = argv[i + 1];
        } else if (strcmp(argv[i], "--epsilon")) {
            epsilon = atof(argv[i + 1]);
        } else if (strcmp(argv[i], "--learning_rate")) {
            rate = atof(argv[i + 1]);
        } else if (strcmp(argv[i], "--epochs")) {
            epoch = atoi(argv[i + 1]);
        }
    }

    readInputFrom(data_path);

    return 0;
}
