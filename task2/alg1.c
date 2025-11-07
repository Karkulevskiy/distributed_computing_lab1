#include <stdio.h>
#include <stdlib.h>
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

void InitAndShareMatrix(int r , int c , int *mat , int my_rank , int *sizes, int *displs)
{
    if (my_rank == 0)
    {
        int *temp = calloc(r * c, sizeof(int));
        for (int i = 0; i < r * c; i++)
            temp[i] = rand() % 10;

        // Распределяет данные из одного члена по всем членам группы. 
        MPI_Scatterv(temp, sizes, displs, MPI_INT, mat, sizes[my_rank], MPI_INT, 0, MPI_COMM_WORLD);
        free(temp);
    }
    else
    {
        // Распределяет данные из одного члена по всем членам группы. 
        MPI_Scatterv(NULL, sizes, displs, MPI_INT, mat, sizes[my_rank], MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void DistributeMatrix(int r, int c, int comm_sz, int *sizes)
{
    for (int i = 0; i < comm_sz; i++)
    {
        // Каждому процессу должно быть выделено r / comm_sz строк, и все c столбцов
        sizes[i] = (r / comm_sz) * c;

        // остаток строк делится по процессам
        // если есть остаток, добавить строку
        if (i < r % comm_sz)  
            sizes[i] += c;    
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

void MatVecMult(int *mat, int *vec, int *res, int local_r, int c)
{
    for (int i = 0; i < local_r; i++)
    {
        res[i] = 0;
        for (int j = 0; j < c; j++)
        {
            res[i] += mat[i * c + j] * vec[j];
        }
    }
}

void PrintResult(int *v, int *r_v, int r, int c, int my_rank)
{
    if (my_rank == 0)
    {
        printf("multiplied by vector:\n");
        for (int i = 0; i < c; i++)
            printf("%d ", v[i]);
        printf("\nequals vector:\n");
        for (int i = 0; i < r; i++)
            printf("%d ", r_v[i]);
        printf("\n");
    }
}

void PrintDistributedMatrix(int r, int c, int *local_mat, int my_rank, int *sizes, int *displs)
{
    if (my_rank == 0)
    {
        int *temp = calloc(r * c, sizeof(int));
        // Собирает переменные данные от всех членов группы к одному члену. 
        MPI_Gatherv(local_mat, sizes[my_rank], MPI_INT, temp, sizes, displs, MPI_INT, 0, MPI_COMM_WORLD);
        printf("matrix:\n");
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
                printf("%d ", temp[i * c + j]);
            printf("\n");
        }
        free(temp);
    }
    else
    {
        // Собирает переменные данные от всех членов группы к одному члену. 
        MPI_Gatherv(local_mat, sizes[my_rank], MPI_INT, NULL, sizes, displs, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char* argv[])
{
    int comm_sz;
    int my_rank;

    MPI_Init(NULL, NULL);

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

    int *vec = (int*)calloc((size_t)c, sizeof(int));
    InitVec(vec, c, my_rank);

    // Считаем размеры и смещения
    int *sizes_mat = calloc(comm_sz, sizeof(int));
    int *sizes_vec = calloc(comm_sz, sizeof(int));
    int *displacements_vec = calloc(comm_sz, sizeof(int));
    int *displacements_mat = calloc(comm_sz, sizeof(int));

    DistributeMatrix(r, c, comm_sz, sizes_mat);
    GetDisplacements(comm_sz, displacements_mat, sizes_mat);
    DistributeMatrix(r, 1, comm_sz, sizes_vec);
    GetDisplacements(comm_sz, displacements_vec, sizes_vec);
    
    int *mat = calloc(sizes_mat[my_rank], sizeof(int)); // Матрица размером ~(row*col / количество процессов)
    int *res = calloc(r, sizeof(int)); // Итоговый вектор размера row*1
    int *local_res = calloc(sizes_mat[my_rank]/c, sizeof(int)); // Итоговый вектор(локальный для процесса)

    InitAndShareMatrix(r, c, mat, my_rank, sizes_mat, displacements_mat); // Задаём случайную матрицу размера r*c
    
    // Умножение матрицы на вектор
    MatVecMult(mat, vec, local_res, sizes_mat[my_rank] / c, c);

    MPI_Gatherv(local_res, sizes_mat[my_rank] / c, MPI_INT, res, sizes_vec, displacements_vec, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Gatherv(что передаём, сколько, тип отправляемых данных, куда записываем, сколько записываем, на какое место записываем, тип получаемых данных, кто принимает, MPI_COMM_WORLD);

    Вывод в консоль
    PrintDistributedMatrix(r, c, mat, my_rank, sizes_mat, displacements_mat);
    PrintResult(vec, res, r, c, my_rank);

    free(mat);
    free(vec);
    free(res);
    free(local_res);
    free(sizes_mat);
    free(sizes_vec);
    free(displacements_mat);
    free(displacements_vec);

    MPI_Finalize();
}
