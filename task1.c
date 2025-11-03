#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Функция: генерирует случайные точки и считает попадания в круг
long long niggers(long long trials, unsigned int seed)
{
    long long hits = 0;
    unsigned int local_seed = seed;

    for (long long i = 0; i < trials; i++)
    {
        // Генерируем x, y в диапазоне [0, 2]
        double x = 2.0 * (double)rand_r(&local_seed) / RAND_MAX;
        double y = 2.0 * (double)rand_r(&local_seed) / RAND_MAX;

        // Проверяем попадание в круг радиуса 1 с центром в (1, 1)
        double dx = x - 1.0;
        double dy = y - 1.0;
        if (dx * dx + dy * dy <= 1.0)
        {
            hits++;
        }
    }
    return hits;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Проверка аргументов (только на rank 0)
    if (rank == 0)
    {
        if (argc != 2)
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Передаём total_trials всем процессам
    long long total_trials = 0;
    if (rank == 0)
    {
        total_trials = atoll(argv[1]);
    }
    MPI_Bcast(&total_trials, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Распределяем точки между процессами
    long long local_trials = total_trials / size;
    long long remainder = total_trials % size;
    if (rank < remainder)
    {
        local_trials++;
    }

    unsigned int seed = (unsigned int)(time(NULL) + rank);

    double start = MPI_Wtime();

    // Каждый процесс считает свои "попадания"
    long long local_hits = niggers(local_trials, seed);

    double end = MPI_Wtime();

    // Собираем все результаты в rank 0
    long long global_hits = 0;
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        double pi = 4.0 * global_hits / total_trials;
        printf("%.8f %.6f\n", pi, end - start);
    }

    MPI_Finalize();
    return 0;
}
