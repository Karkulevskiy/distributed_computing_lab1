#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define G 6.67430e-11f  // гравитационная постоянная
#define BLOCK_SIZE 256
#define SOFTENING 1e-10f

// Структура для хранения состояния частицы
typedef struct {
    float mass;
    float x, y; // положение
    float vx, vy; // скорость
    float fx, fy; // сила
} Particle;

// Ядро для обнуления сил
__global__ void reset_forces_kernel(Particle* particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        particles[idx].fx = 0.0f;
        particles[idx].fy = 0.0f;
    }
}

// Ядро для вычисления сил
__global__ void compute_forces_kernel(Particle* particles, int n) {
    __shared__ Particle shared_particles[BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Particle* p_i = &particles[idx];
    
    float fx = 0.0f;
    float fy = 0.0f;
    
    // Проходим по всем блокам
    for (int tile = 0; tile < gridDim.x; tile++) {
        // Загружаем тайл частиц в shared memory
        int shared_idx = tile * blockDim.x + threadIdx.x;
        if (shared_idx < n) {
            shared_particles[threadIdx.x] = particles[shared_idx];
        } else {
            shared_particles[threadIdx.x].mass = 0.0f;
        }
        __syncthreads();
        
        // Вычисляем взаимодействия с частицами в shared memory
        #pragma unroll
        for (int j = 0; j < blockDim.x; j++) {
            Particle* p_j = &shared_particles[j];
            
            if (p_j->mass == 0.0f) continue;
            
            float dx = p_j->x - p_i->x;
            float dy = p_j->y - p_i->y;
            
            float dist_sq = dx*dx + dy*dy + SOFTENING;
            float inv_dist = rsqrtf(dist_sq);
            float inv_dist_cube = inv_dist * inv_dist * inv_dist;
            
            float force_magnitude = G * p_i->mass * p_j->mass * inv_dist_cube;
            
            fx += force_magnitude * dx;
            fy += force_magnitude * dy;
        }
        __syncthreads();
    }
    
    // Сохраняем вычисленные силы
    if (idx < n) {
        p_i->fx = fx;
        p_i->fy = fy;
    }
}

// Ядро для обновления состояний частиц
__global__ void update_particles_kernel(Particle* particles, int n, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Particle* p = &particles[idx];
        
        // Вычисляем ускорение: a = F/m
        float ax = p->fx / p->mass;
        float ay = p->fy / p->mass;
        
        // Обновляем скорости: v = v + a*dt
        p->vx += ax * dt;
        p->vy += ay * dt;
        
        // Обновляем положения: r = r + v*dt
        p->x += p->vx * dt;
        p->y += p->vy * dt;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    float t = atof(argv[1]);
    char* input_file = argv[2];
    
    FILE* fp = fopen(input_file, "r");
    if (!fp) {
        printf("Ошибка открытия файла %s\n", input_file);
        return 1;
    }
    
    int n;
    if (fscanf(fp, "%d", &n) != 1) {
        printf("Ошибка чтения количества частиц\n");
        fclose(fp);
        return 1;
    }
    
    // Выделяем память на хосте
    Particle* h_particles = (Particle*)malloc(n * sizeof(Particle));
    if (!h_particles) {
        printf("Ошибка выделения памяти для %d частиц\n", n);
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
        h_particles[i].fx = 0.0f;
        h_particles[i].fy = 0.0f;
    }
    fclose(fp);
    
    // Выделяем память на устройстве
    Particle* d_particles;
    size_t particle_size = n * sizeof(Particle);
    checkCudaError(cudaMalloc(&d_particles, particle_size), "cudaMalloc d_particles");
    
    // Копируем данные на устройство
    checkCudaError(cudaMemcpy(d_particles, h_particles, particle_size, 
                              cudaMemcpyHostToDevice), "cudaMemcpy HostToDevice");
    
    // Параметры интегрирования
    float dt = 0.001f;
    int steps = (int)(t / dt);
    int output_interval = 100;
    
    // Конфигурация запуска ядер
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Создаем выходной файл
    char output_file[256];
    snprintf(output_file, sizeof(output_file), "trajectories_cuda.csv");
    FILE* out_fp = fopen(output_file, "w");
    if (!out_fp) {
        printf("Ошибка создания выходного файла\n");
        free(h_particles);
        cudaFree(d_particles);
        return 1;
    }
    
    // Записываем заголовок CSV
    fprintf(out_fp, "t");
    for (int i = 0; i < n; i++) {
        fprintf(out_fp, ",x%d,y%d", i + 1, i + 1);
    }
    fprintf(out_fp, "\n");
    
    // Создаем события для измерения времени
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");
    
    float total_time = 0.0f;
    
    // Главный цикл интегрирования
    for (int step = 0; step <= steps; step++) {
        float t = step * dt;
        
        // Выводим состояние каждые output_interval шагов
        if (step % output_interval == 0) {
            // Копируем данные с устройства на хост для вывода
            checkCudaError(cudaMemcpy(h_particles, d_particles, particle_size, 
                                      cudaMemcpyDeviceToHost), "cudaMemcpy DeviceToHost");
            
            fprintf(out_fp, "%.6f", t);
            for (int i = 0; i < n; i++) {
                fprintf(out_fp, ",%.6f,%.6f", h_particles[i].x, h_particles[i].y);
            }
            fprintf(out_fp, "\n");
        }
        
        // Измеряем время вычислений
        checkCudaError(cudaEventRecord(start, 0), "cudaEventRecord start");
        
        // Вычисляем силы
        compute_forces_kernel<<<grid_dim, block_dim>>>(d_particles, n);
        checkCudaError(cudaGetLastError(), "compute_forces_kernel");
        
        // Обновляем состояния частиц
        update_particles_kernel<<<grid_dim, block_dim>>>(d_particles, n, dt);
        checkCudaError(cudaGetLastError(), "update_particles_kernel");
        
        checkCudaError(cudaEventRecord(stop, 0), "cudaEventRecord stop");
        checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize");
        
        float step_time = 0.0f;
        checkCudaError(cudaEventElapsedTime(&step_time, start, stop), "cudaEventElapsedTime");
        total_time += step_time;
    }
    
    printf("Время выполнения CUDA версии: %.2f секунд.\n", total_time / 1000.0f);
    
    // Очистка
    fclose(out_fp);
    free(h_particles);
    cudaFree(d_particles);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}