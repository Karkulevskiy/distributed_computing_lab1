#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define G 6.67430e-11  // гравитационная постоянная

// Структура для хранения состояния частицы 
typedef struct {
    double mass;
    double x, y;        // положение
    double vx, vy;      // скорость
    double fx, fy;      // сила
} Particle;

// Функция вычисления сил между всеми частицами
void compute_forces(Particle* particles, int n) {
    int i, j;
    
    // Обнуляем силы для всех частиц
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }
    
    // Вычисляем силы взаимодействия между всеми парами частиц
    // Используем третий закон Ньютона: F_ij = -F_ji
    #pragma omp parallel for private(j)
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            
            double dist_sq = dx*dx + dy*dy;
            if (dist_sq < 1e-10) dist_sq = 1e-10;  // защита от деления на ноль
            double dist = sqrt(dist_sq);
            
            double force_magnitude = G * particles[i].mass * particles[j].mass / dist_sq;
            
            double fx = force_magnitude * dx / dist;
            double fy = force_magnitude * dy / dist;
            
            // Применяем третий закон Ньютона
            #pragma omp atomic
            particles[i].fx += fx;
            #pragma omp atomic
            particles[i].fy += fy;
            
            #pragma omp atomic
            particles[j].fx -= fx;
            #pragma omp atomic
            particles[j].fy -= fy;
        }
    }
}

// Функция обновления состояний частиц по методу Эйлера
void update_particles(Particle* particles, int n, double dt) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        // Вычисляем ускорение: a = F/m
        double ax = particles[i].fx / particles[i].mass;
        double ay = particles[i].fy / particles[i].mass;
        
        // Обновляем скорости: v = v + a*dt
        particles[i].vx += ax * dt;
        particles[i].vy += ay * dt;
        
        // Обновляем положения: r = r + v*dt
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
    }
}

int main(int argc, char* argv[]) {
    double tend = atof(argv[1]);  // конечное время
    char* input_file = argv[2];    // файл с данными
    int num_threads = atoi(argv[3]); // число потоков

    omp_set_num_threads(num_threads);
    
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
    
    Particle* particles = (Particle*)malloc(n * sizeof(Particle));
    if (!particles) {
        printf("Ошибка выделения памяти для %d частиц\n", n);
        fclose(fp);
        return 1;
    }
    
    // Читаем данные частиц
    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%lf %lf %lf %lf %lf", 
               &particles[i].mass,
               &particles[i].x, &particles[i].y,
               &particles[i].vx, &particles[i].vy) != 5) {
            printf("Ошибка чтения данных частицы %d\n", i + 1);
            fclose(fp);
            free(particles);
            return 1;
        }
    }
    fclose(fp);
    
    // Параметры интегрирования
    double dt = 0.001;  // шаг по времени
    int steps = (int)(tend / dt);  // количество шагов
    int output_interval = 100;  // выводим каждый 100-й шаг
    
    // Создаем выходной файл
    char output_file[256];
    snprintf(output_file, sizeof(output_file), "trajectories_omp.csv");
    FILE* out_fp = fopen(output_file, "w");
    if (!out_fp) {
        printf("Ошибка создания выходного файла\n");
        free(particles);
        return 1;
    }
    
    // Записываем заголовок CSV
    fprintf(out_fp, "t");
    for (int i = 0; i < n; i++) {
        fprintf(out_fp, ",x%d,y%d", i + 1, i + 1);
    }
    fprintf(out_fp, "\n");
    
    double start_time = omp_get_wtime();
    
    // Главный цикл интегрирования
    for (int step = 0; step <= steps; step++) {
        double t = step * dt;
        
        // Выводим состояние каждые output_interval шагов
        if (step % output_interval == 0) {
            fprintf(out_fp, "%.6f", t);
            for (int i = 0; i < n; i++) {
                fprintf(out_fp, ",%.6f,%.6f", particles[i].x, particles[i].y);
            }
            fprintf(out_fp, "\n");
        }
        
        // Вычисляем силы
        compute_forces(particles, n);
        
        // Обновляем состояния частиц
        update_particles(particles, n, dt);
    }
    
    double end_time = omp_get_wtime();
    printf("Время выполнения OpenMP версии: %.2f секунд. Количество потоков: %d\n", end_time - start_time, num_threads);
    
    fclose(out_fp);
    free(particles);
    
    return 0;
}