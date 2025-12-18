#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define G 6.67430e-11f  // гравитационная постоянная
#define BLOCK_SIZE 256   // размер блока в CUDA
#define MIN_DIST_SQ 1e-10f  // минимальное расстояние

// Структура для хранения состояния частицы на GPU
typedef struct {
    float mass;
    float x, y;          // положение
    float vx, vy;        // скорость
} ParticleGPU;

// Ядро CUDA для вычисления сил
__global__ void compute_forces_kernel_2d(ParticleGPU* particles, float* fx, float* fy, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    float total_fx = 0.0f;
    float total_fy = 0.0f;
    
    float my_x = particles[idx].x;
    float my_y = particles[idx].y;
    float my_mass = particles[idx].mass;
    
    // Вычисляем сумму сил от всех других частиц
    for (int j = 0; j < n; j++) {
        if (j == idx) continue;
        
        float dx = particles[j].x - my_x;
        float dy = particles[j].y - my_y;
        
        float dist_sq = dx*dx + dy*dy;
        if (dist_sq < MIN_DIST_SQ) dist_sq = MIN_DIST_SQ;
        
        float dist = sqrtf(dist_sq);
        
        // формула гравитации: F = G * m1 * m2 / r^2
        float force_magnitude = G * my_mass * particles[j].mass / dist_sq;
        
        total_fx += force_magnitude * dx / dist;
        total_fy += force_magnitude * dy / dist;
    }
    
    fx[idx] = total_fx;
    fy[idx] = total_fy;
}

// Ядро CUDA для обновления состояний частиц
__global__ void update_particles_kernel_2d(ParticleGPU* particles, float* fx, float* fy, float dt, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // Вычисляем ускорение: a = F/m
    float ax = fx[idx] / particles[idx].mass;
    float ay = fy[idx] / particles[idx].mass;
    
    // Обновляем скорости: v = v + a*dt
    particles[idx].vx += ax * dt;
    particles[idx].vy += ay * dt;
    
    // Обновляем положения: r = r + v*dt
    particles[idx].x += particles[idx].vx * dt;
    particles[idx].y += particles[idx].vy * dt;
}

// Ядро для копирования позиций для вывода
__global__ void copy_positions_kernel_2d(ParticleGPU* particles, float* output_x, float* output_y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    output_x[idx] = particles[idx].x;
    output_y[idx] = particles[idx].y;
}

// Ядро для обнуления сил
__global__ void reset_forces_kernel(float* fx, float* fy, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    fx[idx] = 0.0f;
    fy[idx] = 0.0f;
}

// Функция для проверки ошибок CUDA
#define CHECK_CUDA_ERROR(err) { \
    if (err != cudaSuccess) { \
        printf("CUDA ошибка: %s в %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

int main(int argc, char* argv[]) {
    float tend = atof(argv[1]);  // конечное время
    char* input_file = argv[2];   // файл с данными
    
    // Открываем файл с данными
    FILE* fp = fopen(input_file, "r");
    if (!fp) {
        printf("Ошибка открытия файла %s\n", input_file);
        return 1;
    }
    
    // Читаем количество частиц
    int n;
    if (fscanf(fp, "%d", &n) != 1) {
        printf("Ошибка чтения количества частиц\n");
        fclose(fp);
        return 1;
    }
    
    // Выделяем память на хосте
    ParticleGPU* h_particles = (ParticleGPU*)malloc(n * sizeof(ParticleGPU));
    if (!h_particles) {
        printf("Ошибка выделения памяти на хосте для %d частиц\n", n);
        fclose(fp);
        return 1;
    }
    
    // Читаем данные частиц
    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%f %f %f %f %f", 
               &h_particles[i].mass,
               &h_particles[i].x, &h_particles[i].y,
               &h_particles[i].vx, &h_particles[i].vy) != 5) {
            printf("Ошибка чтения данных частицы %d\n", i + 1);
            fclose(fp);
            free(h_particles);
            return 1;
        }
    }
    fclose(fp);
    
    // Выделяем память на GPU
    ParticleGPU* d_particles;
    float *d_fx, *d_fy;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_particles, n * sizeof(ParticleGPU)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fx, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fy, n * sizeof(float)));
    
    // Копируем данные на GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_particles, h_particles, n * sizeof(ParticleGPU), 
                               cudaMemcpyHostToDevice));
    
    // Параметры интегрирования
    float dt = 0.001f;  // шаг по времени
    int steps = (int)(tend / dt);  // количество шагов
    int output_interval = (int)(0.1f / dt);  // выводим каждые 0.1 секунды
    
    if (output_interval < 1) output_interval = 1;
    if (output_interval > 1000) output_interval = 1000;
    
    // Настройка CUDA grid и blocks
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Создаем выходной файл
    char output_file[256];
    snprintf(output_file, sizeof(output_file), "trajectories_cuda.csv");
    FILE* out_fp = fopen(output_file, "w");
    if (!out_fp) {
        printf("Ошибка создания выходного файла\n");
        free(h_particles);
        cudaFree(d_particles);
        cudaFree(d_fx);
        cudaFree(d_fy);
        return 1;
    }
    
    // Записываем заголовок CSV
    fprintf(out_fp, "t");
    for (int i = 0; i < n; i++) {
        fprintf(out_fp, ",x%d,y%d", i+1, i+1);
    }
    fprintf(out_fp, "\n");
    
    // Выделяем память на хосте для вывода
    float* h_output_x = (float*)malloc(n * sizeof(float));
    float* h_output_y = (float*)malloc(n * sizeof(float));
    if (!h_output_x || !h_output_y) {
        printf("Ошибка выделения памяти для вывода\n");
        fclose(out_fp);
        free(h_particles);
        free(h_output_x);
        free(h_output_y);
        cudaFree(d_particles);
        cudaFree(d_fx);
        cudaFree(d_fy);
        return 1;
    }
    
    // Создаем события CUDA для измерения времени
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    // Главный цикл интегрирования
    int output_counter = 0;
    for (int step = 0; step <= steps; step++) {
        float t = step * dt;
        
        // Выводим состояние
        if (step % output_interval == 0) {
            // Копируем позиции частиц
            copy_positions_kernel_2d<<<num_blocks, BLOCK_SIZE>>>(d_particles, d_fx, d_fy, n);
            CHECK_CUDA_ERROR(cudaGetLastError());
            
            // Копируем данные на хост
            CHECK_CUDA_ERROR(cudaMemcpy(h_output_x, d_fx, n * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(h_output_y, d_fy, n * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Записываем в файл
            fprintf(out_fp, "%.6f", t);
            for (int i = 0; i < n; i++) {
                fprintf(out_fp, ",%.6f,%.6f", h_output_x[i], h_output_y[i]);
            }
            fprintf(out_fp, "\n");
            
            output_counter++;
            
            // Прогресс (каждые 10% или каждые 10 выводов)
            if (output_counter % 10 == 0) {
                printf("  Прогресс: %.1f%% (t=%.2f)\n", 
                       (float)step / steps * 100, t);
            }
        }
        
        // Обнуляем силы
        reset_forces_kernel<<<num_blocks, BLOCK_SIZE>>>(d_fx, d_fy, n);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Вычисляем силы на GPU
        compute_forces_kernel_2d<<<num_blocks, BLOCK_SIZE>>>(d_particles, d_fx, d_fy, n);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Обновляем состояния частиц на GPU
        update_particles_kernel_2d<<<num_blocks, BLOCK_SIZE>>>(d_particles, d_fx, d_fy, dt, n);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Синхронизируем
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    
    // Измеряем время выполнения
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    float total_time = milliseconds / 1000.0f;
    
    printf("Время выполнения CUDA версии: %.2f секунд\n", total_time);
    
    // Определяем размер файла результатов
    fclose(out_fp);
    FILE* size_fp = fopen(output_file, "rb");
    if (size_fp) {
        fseek(size_fp, 0, SEEK_END);
        long file_size = ftell(size_fp);
        fclose(size_fp);
    }
    
    // Копируем финальные позиции для проверки
    copy_positions_kernel_2d<<<num_blocks, BLOCK_SIZE>>>(d_particles, d_fx, d_fy, n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_x, d_fx, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_y, d_fy, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Очистка
    free(h_particles);
    free(h_output_x);
    free(h_output_y);
    
    cudaFree(d_particles);
    cudaFree(d_fx);
    cudaFree(d_fy);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Сбрасываем устройство
    cudaDeviceReset();
    
    return 0;
}