// Команды для запуска
// mpicc -o matrix_vector_mult matrix_vector_mult.c
// mpirun -np <количество_процессов> ./matrix_vector_mult <строки> <столбцы>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void InitVec(int *v, int n, int my_rank)
{
    if (my_rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            v[i] = rand() % 10;
        }
    }
    // Транслирует данные от одного участника группы всем участникам группы.
    MPI_Bcast(v, n, MPI_INT, 0, MPI_COMM_WORLD);
}

void BuildElemCountsDispls(int r, int comm_sz, const int *sizes, int *counts, int *displs_elems, int *total_elems_out) {
    int total = 0;
    for (int p = 0; p < comm_sz; p++) counts[p] = r * sizes[p];
    displs_elems[0] = 0;
    for (int p = 1; p < comm_sz; p++) displs_elems[p] = displs_elems[p-1] + counts[p-1];
    total = displs_elems[comm_sz-1] + counts[comm_sz-1];
    if (total_elems_out) *total_elems_out = total;
}

void InitAndShareMatrix(int r, int c, int *mat_local, int my_rank, int comm_sz, int *sizes, int *displs)
{
    if (my_rank == 0)
    {
        int *sendcounts = (int *)calloc(comm_sz, sizeof(int));
        int *displs_elems = (int *)calloc(comm_sz, sizeof(int));
        int total_elems = 0;
        BuildElemCountsDispls(r, comm_sz, sizes, sendcounts, displs_elems, &total_elems);

        // Генерим полную матрицу r×c
        int *mat = (int*)calloc(r * c, sizeof(int));
        for (int i = 0; i < r * c; i++) mat[i] = rand() % 10;

        // упакуем по процессам
        int *sendbuf = (int*)calloc(total_elems, sizeof(int));
        for (int p = 0; p < comm_sz; p++) {
            int kcols = sizes[p];
            int c0    = displs[p]; // стартовый столбец для p
            int base  = displs_elems[p];
            // копируем построчно подряд kcols столбцов
            for (int i = 0; i < r; i++) {
                // из A берём диапазон [i*c + c0, длиной kcols]
                memcpy(&sendbuf[base + i * kcols],
                       &mat[i * c + c0],
                       kcols * sizeof(int));
            }
        }

        // Распределим данные всем членам группы
        MPI_Scatterv(sendbuf, sendcounts, displs_elems, MPI_INT, mat_local, r * sizes[my_rank], MPI_INT, 0, MPI_COMM_WORLD);

        free(mat);
        free(sendbuf);
        free(sendcounts);
        free(displs_elems);
    } else {
        // Принимаем наш блок
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, mat_local, r * sizes[my_rank], MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void DistributeColumns(int c, int comm_sz, int *sizes)
{
    for (int i = 0; i < comm_sz; i++)
    {
        // Каждому процессу должно быть выделено c / comm_sz строк, и все r столбцов
        sizes[i] = (c / comm_sz);

        // остаток строк делится по процессам
        // если есть остаток, добавить строку
        if (i < c % comm_sz)  
            sizes[i] += 1;    
    }
}

void GetDisplacements(int comm_sz, int *displs, int *sizes)
{
    // Displacements - смещения
    displs[0] = 0;
    for (int i = 1; i < comm_sz; i++)
    {
        displs[i] = displs[i-1] + sizes[i-1];
    }
}

void MatrixVectorMultiply(int r, int local_cols, int *local_mat, int *vector, int *local_result)
{
    for (int i = 0; i < r; i++){
        int acc = 0;
        for (int j = 0; j < local_cols; j++)
        {
            acc += local_mat[i * local_cols + j] * vector[j];
        }
        local_result[i] = acc;
    }
}

int* GatherMatrix( int r, int c,int my_rank, int comm_sz,const int *mat_local,const int *sizes, int *displs) {
    int *mat = NULL;
    int *counts = (int*)calloc(comm_sz, sizeof(int));
    int *displs_elems = (int*)calloc(comm_sz, sizeof(int));
    int total_elems = 0;
    BuildElemCountsDispls(r, comm_sz, sizes, counts, displs_elems, &total_elems);

    // Сначала собираем в упакованный буфер (та же компоновка, что и при рассылке)
    int *packed = NULL;
    if (my_rank == 0) packed = (int*)calloc(total_elems, sizeof(int));

    MPI_Gatherv((void*)mat_local, r * sizes[my_rank], MPI_INT,
                packed, counts, displs_elems, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        mat = (int*)calloc((size_t)r * c, sizeof(int));
        for (int p = 0; p < comm_sz; p++) {
            int kcols = sizes[p], c0 = displs[p], base = displs_elems[p];
            for (int i = 0; i < r; i++) {
                memcpy(&mat[i * c + c0],
                       &packed[base + i * kcols],
                       (size_t)kcols * sizeof(int));
            }
        }
    }
    free(counts);
    free(displs_elems);
    free(packed);

    return mat; // только на ранге 0 — не NULL
}

void PrintVector(const char *title, const int *v, int n) {
    printf("%s\n", title);
    for (int i = 0; i < n; i++) printf("%d ", v[i]);
    printf("\n\n");
}

void PrintMatrixRowMajor(const char *title, const int *A, int r, int c) {
    printf("%s\n", title);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) printf("%d ", A[i * c + j]);
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    int comm_sz;
    int my_rank;

    MPI_Init(&argc, &argv);
    
    if (argc != 3) {
        if (my_rank == 0) {
            fprintf(stderr, "Usage: %s <rows> <cols>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); // Извлекает количество процессов, участвующих в коммуникаторе, или общее количество доступных процессов.
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // Извлекает ранг вызывающего процесса в группе указанного коммуникатора.

    int r = atoi(argv[1]); // <строки>
    int c = atoi(argv[2]); // <столбцы>
    
    MPI_Bcast(&r, 1, MPI_INT, 0, MPI_COMM_WORLD); //Передаём всем процессам количество строк
    MPI_Bcast(&c, 1, MPI_INT, 0, MPI_COMM_WORLD); //Передаём всем процессам количество столбцов

    int *sizes_vec = calloc(comm_sz, sizeof(int));
    int *displacements_vec = calloc(comm_sz, sizeof(int));

    DistributeColumns(c, comm_sz, sizes_vec);
    GetDisplacements(comm_sz, displacements_vec, sizes_vec);

    int local_cols = sizes_vec[my_rank];
    int *mat_local = (int *)calloc((size_t)r * local_cols, sizeof(int));

    InitAndShareMatrix(r,c, mat_local, my_rank, comm_sz, sizes_vec,displacements_vec );
    
    int *vector = calloc(c, sizeof(int));
    InitVec(vector, c, my_rank);

    int *local_result = calloc(r, sizeof(int));
    MatrixVectorMultiply(r, local_cols, mat_local, vector + displacements_vec[my_rank], local_result);

    int *result = NULL;
    if (my_rank == 0) result = (int*)calloc(r, sizeof(int));
    MPI_Reduce(local_result, (my_rank == 0 ? result : NULL), r, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Сборка полной матрицы у ранга 0
    int *mat = GatherMatrix(r, c, my_rank, comm_sz, mat_local, sizes_vec, displacements_vec);

    // Печать на корне исходного вектора, матрицы и результата
    if (my_rank == 0) {
        PrintVector("Vector:", vector, c);
        PrintMatrixRowMajor("Matrix(gathered):", mat, r, c);
        PrintVector("Result", result, r);
        free(result);
        free(mat);
    }

    free(mat_local);
    free(vector);
    free(local_result);
    free(sizes_vec);
    free(displacements_vec);

    MPI_Finalize();
}
