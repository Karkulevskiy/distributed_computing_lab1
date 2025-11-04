#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX 10.0
#define MIN -10.0

void init_replaced(int *replaced, int size, int squared, int block_size)
{
    for (int i = 0; i < size; i++)
        replaced[i] = (i % squared) + i / squared * block_size * squared;
}

void init_counts(int *counts, int size)
{
    for (int i = 0; i < size; i++)
        counts[i] = 1;
}

int main(int argc, char **argv)
{
    int rank, size, squared;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    squared = (int)sqrt(size);
    if (squared * squared != size)
    {
        MPI_Finalize();
        return -1;
    }

    int n = atoi(argv[1]);
    if (n % squared != 0)
    {
        MPI_Finalize();
        return -1;
    }

    int block_size = n / squared;
    float *A = NULL, *B = NULL;
    if (rank == 0)
    {
        // генерим рандомно матрицы
        A = (float *)malloc(n * n * sizeof(float));
        B = (float *)malloc(n * n * sizeof(float));
        for (int i = 0; i < n * n; i++)
        {
            A[i] = 2 * MAX * (float)rand() / RAND_MAX - MIN;
            B[i] = 2 * MAX * (float)rand() / RAND_MAX - MIN;
        }
    }

    float *local_a = (float *)calloc(block_size * block_size, sizeof(float));
    float *local_b = (float *)calloc(block_size * block_size, sizeof(float));
    float *local_c = (float *)calloc(block_size * block_size, sizeof(float));

    float start = MPI_Wtime();

    MPI_Comm comm_cart;
    int dims[2] = {squared, squared};
    int periods[2] = {1, 1};
    // 1 2 3 4 => 1 2
    //            3 4
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_cart);

    int full_matrix_dims[2] = {n, n};
    int sub_start[2] = {0, 0};
    int sub_dim[2] = {block_size, block_size};

    MPI_Datatype block_type;
    MPI_Type_create_subarray(2, full_matrix_dims, sub_dim, sub_start, MPI_ORDER_C, MPI_FLOAT, &block_type);
    MPI_Type_create_resized(block_type, 0, block_size * sizeof(float), &block_type);
    MPI_Type_commit(&block_type);

    int replaced[size];
    init_replaced(replaced, size, squared, block_size);

    int counts[size];
    init_counts(counts, size);

    // shared
    MPI_Scatterv(A, counts, replaced, block_type, local_a, block_size * block_size, MPI_FLOAT, 0, comm_cart);
    MPI_Scatterv(B, counts, replaced, block_type, local_b, block_size * block_size, MPI_FLOAT, 0, comm_cart);

    MPI_Status status;
    int src, dest;
    int proc_coords[2];
    MPI_Cart_coords(comm_cart, rank, 2, proc_coords);

    // начальный сдвиг
    MPI_Cart_shift(comm_cart, 1, -proc_coords[0], &src, &dest);
    MPI_Sendrecv_replace(local_a, block_size * block_size, MPI_FLOAT, dest, 0, src, 0, comm_cart, &status);

    MPI_Cart_shift(comm_cart, 0, -proc_coords[1], &src, &dest);
    MPI_Sendrecv_replace(local_b, block_size * block_size, MPI_FLOAT, dest, 0, src, 0, comm_cart, &status);

    // canon alg
    for (int step = 0; step < squared; step++)
    {
        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                for (int k = 0; k < block_size; k++)
                    local_c[i * block_size + j] += local_a[i * block_size + k] * local_b[k * block_size + j];
            }
        }

        MPI_Cart_shift(comm_cart, 1, -1, &src, &dest);
        MPI_Sendrecv_replace(local_a, block_size * block_size, MPI_FLOAT, dest, 0, src, 0, comm_cart, &status);

        MPI_Cart_shift(comm_cart, 0, -1, &src, &dest);
        MPI_Sendrecv_replace(local_b, block_size * block_size, MPI_FLOAT, dest, 0, src, 0, comm_cart, &status);
    }

    float *C = NULL;
    if (rank == 0)
        C = (float *)malloc(n * n * sizeof(float));

    MPI_Gatherv(local_c, block_size * block_size, MPI_FLOAT, C, counts, replaced, block_type, 0, comm_cart);

    float end = MPI_Wtime();
    float duration = end - start;
    float total_time;

    MPI_Reduce(&duration, &total_time, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    double avg_time = total_time / size;
    if (rank == 0)
        printf("%lf\n", avg_time);

    free(local_a);
    free(local_b);
    free(local_c);
    if (rank == 0)
    {
        free(A);
        free(B);
        free(C);
    }
    MPI_Type_free(&block_type);
    MPI_Finalize();
    return 0;
}
