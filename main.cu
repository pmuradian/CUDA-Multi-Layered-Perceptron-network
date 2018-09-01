#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include "cuda.h"
#include "ppm/pnmio.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "ppm/stb_image_resize.h"

#define DEFAULT_PATH "./Training/"
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
#define RANDOM_FLOAT (double)rand() / (double)RAND_MAX

// CUDA defines
#define DIM 1024
#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f
#define SPHERES 100
#define THREADS 256
#define X_DIM 64
#define Y_DIM 64


int layers[LAYER_COUNT] = { INPUT_LAYER_SIZE,
                            FIRST_LAYER_SIZE,
                            SECOND_LAYER_SIZE,
                            THIRD_LAYER_SIZE,
                            FOURTH_LAYER_SIZE,
                            OUTPUT_LAYER_SIZE };


__device__ double reLU(double x) {
    if (x > 0) {
    	return x;
    }
    return 0.0;
}

__device__ double reLU_der(double x) {
    return x < 0 ? 0 : 1;
}

__device__ int softmax(double *input, size_t input_len) {
    assert(input);

    double m = -INFINITY;
    for (size_t i = 0; i < input_len; i++) {
        if (input[i] > m) {
            m = input[i];
        }
    }
	printf("max = %lf ", m);

    long double sum = 0.0;
    for (size_t i = 0; i < input_len; i++) {
	printf("asdf = %f\n", input[i] - m);
        sum += exp(input[i] - m);
    }
	//printf("sum = %lf ", sum);

    double offset = m + logf(sum);
    for (size_t i = 0; i < input_len; i++) {
        input[i] = expf(input[i] - offset);
	printf("output = %lf \n", input[i]);
    }
    return 0;
}

__device__ double softmax_der(double calculated, double expected) {
    return calculated - expected;
}

__device__ double error_function(double expected_value, double calculated_value) {
    return -1 * (expected_value * log(calculated_value));
}

double *toGrayScale(const unsigned char *input, int x_dim, int y_dim) {
    int j = 0;
    double *grayscale = (double*)malloc(IMAGE_DIMENTIONS * sizeof(double));

    for (int i = 0; i < 3 * IMAGE_DIMENTIONS; i += 3) {
        grayscale[j++] = ((0.3 * input[i]) + (0.59 * input[i + 1]) + (0.11 * input[i + 2])) / 255.0;
    }
    return grayscale;
}

double random_double() {
    return (double)rand() / (double)RAND_MAX;
}

int getLayerSize(int layer) {
    return layer >= LAYER_COUNT ? 0 : layers[layer];
}

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

// TODO: use cuda
void createWeights(int isRandom, int** weights) {
    for (int k = 1; k < LAYER_COUNT; k++) {
        int size = getLayerSize(k);
        int previous_layer_size = getLayerSize(k - 1);
        double *weight = (double*)malloc(size * previous_layer_size * sizeof(double));

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < previous_layer_size; j++) {
                // weight for node i in layer k from node j in layer k - 1
                weight[i * previous_layer_size + j] = random ? random_double() : 1;
            }
        }
        weights[k - 1] = weight;
        int block_count = getLayerSize(k) / THREADS;
    }
}

__global__ void backward_phase_output_layer(double *expected_values, double *calculated_values, double *out_delta, int current_layer_size, int prev_layer_size) {
    int index = threadIdx.x;
    out_delta[index] = softmax_der(calculated_values[index], expected_values[index]);
}

__global__ void backward_phase(double *activated_values, double *product_sum, double *weights, double *deltas, double *out_deltas, double *out_values, int current_layer_size, int prev_layer_size, int next_layer_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;

    for (int i = 0; i < next_layer_size; i++) {
        sum += weights[index * current_layer_size + i] * deltas[i];
    }
    out_deltas[index] = reLU_der(product_sum[index]) * sum;

    for (int i = 0; i < prev_layer_size; i++) {
        out_values[index * prev_layer_size + i] = activated_values[i] * out_deltas[index];
    }
}


__global__ void forward_phase_output_layer(double *input, double *weights, double *output, double *out_product_sum, int current_layer_size, int prev_layer_size) {

    int index = threadIdx.x;
    long double sum = 0;

    for (int j = 0; j < prev_layer_size; j++) {
        sum += weights[index * prev_layer_size + j] * input[j];
    }

    output[index] = sum;
    out_product_sum[index] = sum;
    __syncthreads();
    if (index == 0) {
	    printf("sum = %f\n", sum);
        softmax(output, current_layer_size);
    }
}

__global__ void forward_phase(double *input, double *weights, double *output, double *out_product_sum, int current_layer_size, int prev_layer_size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;

    for (int j = 0; j < prev_layer_size; j++) {
        sum += weights[index * prev_layer_size + j] * input[j];
    }

    out_product_sum[index] = sum;
    output[index] = reLU(sum);
}

// reads input file and returns resized 64x64 image data
void readFileFromPath(char *path, unsigned char *resized_data) {
    DIR* FD;
    struct dirent* in_file;
    FILE    *common_file;
    FILE    *entry_file;
    char    buffer[14];

    FILE *file = fopen(path, "rb");
    int x_dim;
    int y_dim;
    int img_colors = 0;
    int is_asci = 0;
    int pnm_type = 0;
    int **biases;

    int isRandom = 1;
    if (file == NULL) {
        printf("file is null \n");
    } else {
        printf("file is not null  \n");
    }
    // Read contentis of ppm file header
    printf("Opening file to read \n");
    pnm_type = get_pnm_type(file);
    rewind(file);
    read_ppm_header(file, &x_dim, &y_dim, &img_colors, &is_asci);
    printf("Success\n");

    unsigned char *original_data = (unsigned char*)malloc(3 * x_dim * y_dim * sizeof(char));
    double *grayscale[LAYER_COUNT];
    double *product_sum[LAYER_COUNT];
    double *error_terms[LAYER_COUNT];
    int output = 0;

    int *image_data = (int*)malloc(3 * x_dim * y_dim * sizeof(int));
    // Read image data and resize it to 64x64px
    read_ppm_data(file, image_data, is_asci);
    for (int i = 0; i < 3 * x_dim * y_dim; i++) {
        original_data[i] = image_data[i];
    }

    stbir_resize_uint8(original_data, x_dim, y_dim, 0, resized_data, 64, 64, 0, 3);
    fclose(file);
}

void startTraining(char *path) {
    unsigned char *resized_data = (unsigned char*)malloc(3 * X_DIM * Y_DIM * sizeof(char));
    readFileFromPath("./Training/00000/01153_00000.ppm", resized_data);
    // Convert to grayscale
    printf("Image converted to grayscale\n");
    grayscale[0] = toGrayScale(resized_data, X_DIM, Y_DIM);

    // Initialize product sum and error terms
    product_sum[0] = (double *)malloc(getLayerSize(0) * sizeof(double));
    error_terms[0] = (double *)malloc(getLayerSize(0) * sizeof(double));

    for (int i = 1; i < LAYER_COUNT; i++) {
        grayscale[i] = (double *)malloc(getLayerSize(i) * sizeof(double));
        product_sum[i] = (double *)malloc(getLayerSize(i) * sizeof(double));
        error_terms[i] = (double *)malloc(getLayerSize(i) * sizeof(double));
    }

    // Initialize weights
    double *weights[LAYER_COUNT];
    createWeights(random, weights);
    biases = createInitialBiases();

    int input_number = 0;
    int iteration = 0;
    int num_iterations = 10;
    double expected_output[OUTPUT_LAYER_SIZE];

    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        expected_output[i] = 0.0;
        if (i == 0)
            expected_output[i] = 1;
    }

    while (++iteration < num_iterations) {
	    printf("iteration = %d\n", iteration);
        // forward phase
        for (int i = 1; i < LAYER_COUNT; i++) {
            printf("current layer %d\n", i);
	        double *weight;
            double *cuda_input;
            double *cuda_output;
            double *cuda_product_sum;
            int curr_layer_size = getLayerSize(i);
            int prev_layer_size = getLayerSize(i - 1);
            int curr_buff_size = curr_layer_size * sizeof(double);
            int prev_buff_size = prev_layer_size * sizeof(double);

            cudaMalloc((void**)&weight, curr_layer_size * prev_layer_size * sizeof(double));
            cudaMalloc((void**)&cuda_input, prev_buff_size);
            cudaMalloc((void**)&cuda_output, curr_buff_size);
            cudaMalloc((void**)&cuda_product_sum, curr_buff_size);

            cudaMemcpy(weight, weights[i - 1], curr_layer_size * prev_layer_size * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(cuda_input, grayscale[i - 1], prev_buff_size, cudaMemcpyHostToDevice);
//            printf("size = %d\n", curr_layer_size);
//	          for (int j = 0; j < 10; j++)
//                printf("input value = %lf\n", grayscale[i - 1][j]);

            if (i == LAYER_COUNT - 1) {
                forward_phase_output_layer<<<1, 62>>>(cuda_input, weight, cuda_output, cuda_product_sum, curr_layer_size, prev_layer_size);
            } else {
                forward_phase<<<curr_layer_size / 256, 256>>>(cuda_input, weight, cuda_output, cuda_product_sum, curr_layer_size, prev_layer_size);
            }
            cudaMemcpy(grayscale[i], cuda_output, curr_buff_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(product_sum[i], cuda_product_sum, curr_buff_size, cudaMemcpyDeviceToHost);

  //          for (int j = 0; j < 10; j++)
  //              printf("output value = %lf\n", grayscale[i][j]);

            cudaFree(cuda_input);
            cudaFree(cuda_output);
            cudaFree(weight);
            cudaFree(cuda_product_sum);
	        printf("memory freed\n");
        }

        // backward phase
        for (int i = LAYER_COUNT - 1; i > 0; i--) {
            printf("current layer %d\n", i);
            double *weight;
            double *cuda_input;
            double *deltas;
            double *out_deltas;
            double *cuda_expected;
            double *cuda_output;
            double *cuda_product_sum;
            int curr_layer_size = getLayerSize(i);
            int prev_layer_size = getLayerSize(i - 1);
            int next_layer_size = getLayerSize(i + 1);
            int curr_buff_size = curr_layer_size * sizeof(double);
            int prev_buff_size = prev_layer_size * sizeof(double);
            int next_buff_size = next_buff_size * sizeof(double);
            cudaMalloc((void**)&weight, curr_layer_size * next_layer_size * sizeof(double));
            cudaMalloc((void**)&cuda_input, curr_buff_size);
            cudaMalloc((void**)&cuda_output, prev_buff_size);
            cudaMalloc((void**)&cuda_expected, curr_buff_size);
            cudaMalloc((void**)&deltas, prev_buff_size);
            cudaMalloc((void**)&cuda_product_sum, curr_buff_size);

            printf("size = %d\n", curr_layer_size);

            if (i == LAYER_COUNT - 1) {
                cudaMemcpy(cuda_input, grayscale[i], curr_buff_size, cudaMemcpyHostToDevice); // activation function output
                backward_phase_output_layer<<<1, 62>>>(cuda_expected, cuda_input, cuda_output, curr_layer_size, prev_layer_size);
            } else {
                cudaMemcpy(cuda_input, grayscale[i - 1], prev_buff_size, cudaMemcpyHostToDevice); // activation function output
                cudaMemcpy(cuda_product_sum, product_sum[i], curr_buff_size, cudaMemcpyHostToDevice); // product sum
                cudaMemcpy(deltas, error_terms[i + 1], next_buff_size, cudaMemcpyHostToDevice); // error terms
                cudaMemcpy(weight, weights[i], curr_layer_size * next_buff_size * sizeof(double), cudaMemcpyHostToDevice); // weights
                backward_phase<<<curr_layer_size / 256, 256>>>(cuda_input, cuda_product_sum, deltas, weight, out_deltas, cuda_output, curr_layer_size, prev_layer_size, next_layer_size);
            }

            cudaMemcpy(error_terms[i], cuda_output, curr_buff_size, cudaMemcpyDeviceToHost);

            for (int j = 0; j < 10; j++)
                printf("output value = %lf\n", error_terms[i][j]);

            //cudaMemcpy(weights[i], weight, prev_buff_size, cudaMemcpyDeviceToHost);
            cudaFree(cuda_input);
            cudaFree(cuda_output);
            cudaFree(cuda_product_sum);
            cudaFree(cuda_expected);
            cudaFree(weight);
            cudaFree(deltas);
            printf("memory freed\n");
        }
    }
    // TODO: free for every malloc
    //free(weights);
    //free(biases);
    printf("programm execution complete");
}

int main(int argc, char **argv) {

    char *data_path = DEFAULT_PATH;
    double epsilon = DEFAULT_EPSILON;
    double rate = DEFAULT_RATE;
    int epoch = DEFAULT_EPOCH;

    if (argc % 2 != 0)
        printf("Not enough arguments passed");

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

    startTraining(data_path);
    return 0;
}
