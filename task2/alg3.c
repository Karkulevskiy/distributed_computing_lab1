#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

static void fill_vector(int *v, int n, int my_rank) {
    if (my_rank == 0) {
        for (int i = 0; i < n; i++) v[i] = rand() % 10;
    }
    MPI_Bcast(v, n, MPI_INT, 0, MPI_COMM_WORLD);
}

static void print_vec(const char *title, const int *v, int n) {
    if (title) printf("%s\n", title);
    for (int i = 0; i < n; i++) printf("%d ", v[i]);
    printf("\n\n");
}

static void print_mat_rm(const char *title, const int *mat, int r, int c) {
    if (title) printf("%s\n", title);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) printf("%d ", mat[i*c + j]);
        printf("\n");
    }
    printf("\n");
}

// Разбивка r на P блоков: sizes_rows[i], displs_rows[i]
static void split_1d(int n, int parts, int *sizes, int *displs) {
    int base = n / parts, rem = n % parts;
    displs[0] = 0;
    for (int i = 0; i < parts; i++) {
        sizes[i] = base + (i < rem ? 1 : 0);
        if (i > 0) displs[i] = displs[i-1] + sizes[i-1];
    }
}

// Построить counts/disp по элементам для Scatterv/Gatherv для блочной матрицы
// Ранги нумеруются построчно: rank = pi * Q + pj
static void build_counts_displs_elems(int P, int Q,
                                      const int *rows_sz, const int *cols_sz,
                                      int *counts, int *displs_elems, int *total_out)
{
    int rtot = 0;
    for (int pi = 0; pi < P; pi++) {
        for (int pj = 0; pj < Q; pj++) {
            int rank = pi * Q + pj;
            counts[rank] = rows_sz[pi] * cols_sz[pj];
        }
    }
    displs_elems[0] = 0;
    for (int r = 1; r < P*Q; r++) displs_elems[r] = displs_elems[r-1] + counts[r-1];
    if (total_out) *total_out = displs_elems[P*Q - 1] + counts[P*Q - 1];
}

// Рассылка 2D-блоков
static void scatter_blocks(int r, int c, int P, int Q,
                           const int *rows_sz, const int *rows_disp,
                           const int *cols_sz, const int *cols_disp,
                           int *mat_local, int my_rank)
{
    int comm_sz;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    int *counts = (int*)calloc(comm_sz, sizeof(int));
    int *displs = (int*)calloc(comm_sz, sizeof(int));
    int total_elems = 0;
    build_counts_displs_elems(P, Q, rows_sz, cols_sz, counts, displs, &total_elems);

    if (my_rank == 0) {
        // Сгенерим полную матрицу
        int *mat = (int*)calloc((size_t)r * c, sizeof(int));
        for (int i = 0; i < r*c; i++) mat[i] = rand() % 10;

        // Упакуем блоки подряд друг за другом
        int *sendbuf = (int*)calloc((size_t)total_elems, sizeof(int));
        for (int pi = 0; pi < P; pi++) {
            for (int pj = 0; pj < Q; pj++) {
                int rank = pi * Q + pj;
                int rl = rows_sz[pi], cl = cols_sz[pj];
                int r0 = rows_disp[pi], c0 = cols_disp[pj];
                int base = displs[rank];

                for (int i = 0; i < rl; i++) {
                    memcpy(&sendbuf[base + i * cl],
                           &mat[(r0 + i) * c + c0],
                           (size_t)cl * sizeof(int));
                }
            }
        }

        MPI_Scatterv(sendbuf, counts, displs, MPI_INT,
                     mat_local, counts[0], MPI_INT, 0, MPI_COMM_WORLD);

        free(mat);
        free(sendbuf);
    } else {
        // Примем наш кусок (размер нам известен)
        int my_count = counts[my_rank];
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
                     mat_local, my_count, MPI_INT, 0, MPI_COMM_WORLD);
    }

    free(counts);
    free(displs);
}

// Сборка полной матрицы из 2D-блоков для печати/проверки (только на ранге 0)
static int* gather_blocks(int r, int c, int P, int Q,
                          const int *rows_sz, const int *rows_disp,
                          const int *cols_sz, const int *cols_disp,
                          const int *mat_local, int my_rank)
{
    int comm_sz;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    int *counts = (int*)calloc(comm_sz, sizeof(int));
    int *displs = (int*)calloc(comm_sz, sizeof(int));
    int total_elems = 0;
    build_counts_displs_elems(P, Q, rows_sz, cols_sz, counts, displs, &total_elems);

    int *packed = NULL;
    if (my_rank == 0) packed = (int*)calloc((size_t)total_elems, sizeof(int));

    MPI_Gatherv((void*)mat_local, counts[my_rank], MPI_INT,
                packed, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    int *mat = NULL;
    if (my_rank == 0) {
        mat = (int*)calloc((size_t)r * c, sizeof(int));
        for (int pi = 0; pi < P; pi++) {
            for (int pj = 0; pj < Q; pj++) {
                int rank = pi * Q + pj;
                int rl = rows_sz[pi], cl = cols_sz[pj];
                int r0 = rows_disp[pi], c0 = cols_disp[pj];
                int base = displs[rank];
                for (int i = 0; i < rl; i++) {
                    memcpy(&mat[(r0 + i) * c + c0],
                           &packed[base + i * cl],
                           (size_t)cl * sizeof(int));
                }
            }
        }
    }

    free(counts); free(displs); free(packed);
    return mat; // только на корне — не NULL
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int comm_sz = 0, my_rank = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (argc != 3) {
        if (my_rank == 0) fprintf(stderr, "Usage: %s <rows> <cols>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int r = atoi(argv[1]);
    int c = atoi(argv[2]);

    int dims[2] = {0, 0};
    MPI_Dims_create(comm_sz, 2, dims); // подберёт P,Q автоматически(не всегда квадратная сетка)
    int P = dims[0], Q = dims[1];

    // Координаты процесса в сетке
    int pi = my_rank / Q; // индекс по строкам сетки
    int pj = my_rank % Q; // индекс по столбцам сетки

    // Разбивка по строкам и столбцам 
    int *rows_sz  = (int*)calloc(P, sizeof(int));
    int *rows_disp= (int*)calloc(P, sizeof(int));
    int *cols_sz  = (int*)calloc(Q, sizeof(int));
    int *cols_disp= (int*)calloc(Q, sizeof(int));

    split_1d(r, P, rows_sz, rows_disp);
    split_1d(c, Q, cols_sz, cols_disp);

    int rows_local = rows_sz[pi];
    int cols_local = cols_sz[pj];

    // Получаем свой блок mat(pi,pj)
    int *mat_local = (int*)calloc((size_t)rows_local * cols_local, sizeof(int));
    scatter_blocks(r, c, P, Q, rows_sz, rows_disp, cols_sz, cols_disp, mat_local, my_rank);

    // Наш вектор
    int *vector = (int*)calloc(c, sizeof(int));
    fill_vector(vector, c, my_rank);
    const int c0 = cols_disp[pj]; // смещение по столбцам в глобальном vector
    const int *x_local = vector + c0;  // длина cols_local

    //  Локальное умножение: y_partial (rows_local) 
    int *y_partial = (int*)calloc(rows_local, sizeof(int));
    for (int i = 0; i < rows_local; i++) {
        int acc = 0;
        // mat_local — row-major: i * cols_local + j
        for (int j = 0; j < cols_local; j++) acc += mat_local[i * cols_local + j] * x_local[j];
        y_partial[i] = acc;
    }

    // Суммирование по процессам в строке сетки (фикс. pi, суммируем по pj) 
    // Разобьём на коммуникаторы строк и столбцов
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, pi, pj, &row_comm); // все с одним pi
    MPI_Comm_split(MPI_COMM_WORLD, pj, pi, &col_comm); // все с одним pj

    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    // Сложим по столбцам в "левом" процессе строки (тот, у кого pj == 0)
    int is_row_leader = (pj == 0);
    int *y_rowsum = (int*)calloc(rows_local, sizeof(int));
    MPI_Reduce(y_partial, (is_row_leader ? y_rowsum : NULL),
               rows_local, MPI_INT, MPI_SUM, 0, row_comm);

    // Собираем итоговый вектор result на процессе 0
    int *result = NULL;
    if (my_rank == 0) result = (int*)calloc(r, sizeof(int));

    // Участие в сборе только лидеров строк (pj==0). Остальные шлют 0 элементов.
    int *gcounts = NULL, *gdispls = NULL;
    if (my_rank == 0) {
        gcounts = (int*)calloc(comm_sz, sizeof(int));
        gdispls = (int*)calloc(comm_sz, sizeof(int));
        // Заполним только для рангов pj==0
        for (int ppi = 0; ppi < P; ppi++) {
            int rank_leader = ppi * Q + 0;
            gcounts[rank_leader] = rows_sz[ppi];
        }
        // displs в result — по глобальным строкам
        gdispls[0] = 0;
        for (int rnk = 1; rnk < comm_sz; rnk++) gdispls[rnk] = gdispls[rnk-1] + gcounts[rnk-1];
    }

    // Gatherv, нули для не-лидеров строк
    int sendcount = is_row_leader ? rows_local : 0;
    MPI_Gatherv(is_row_leader ? y_rowsum : NULL, sendcount, MPI_INT,
                result, gcounts, gdispls, MPI_INT, 0, MPI_COMM_WORLD);

    // Сборка mat для печати
    int *mat_full = gather_blocks(r, c, P, Q, rows_sz, rows_disp, cols_sz, cols_disp, mat_local, my_rank);

    // if (my_rank == 0) {
    //     print_vec("Vector:", vector, c);
    //     print_mat_rm("Matrix(gathered):", mat_full, r, c);
    //     print_vec("Result:", result, r);
    // }

    // Очистка 
    free(mat_local);
    free(vector);
    free(y_partial);
    free(y_rowsum);
    if (my_rank == 0) {
        free(result);
        free(mat_full);
        free(gcounts);
        free(gdispls);
    }
    free(rows_sz); free(rows_disp);
    free(cols_sz); free(cols_disp);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize();
    return 0;
}
