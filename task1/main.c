#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define MAX_ITER 1000
#define X_MIN -3.0
#define X_MAX 3.0
#define Y_MIN -1.5
#define Y_MAX 1.5
#define VIS_WIDTH 200    // Ширина ASCII-визуализации
#define VIS_HEIGHT 60   // Высота ASCII-визуализации

typedef struct {
    double x;
    double y;
} Point;

typedef struct {
    double real;
    double imag;
} Complex;

void visualize_mandelbrot(const char* csv_filename, const char* output_filename) {
    FILE *csv_file = fopen(csv_filename, "r");
    if (csv_file == NULL) {
        fprintf(stderr, "Ошибка открытия файла %s для чтения\n", csv_filename);
        return;
    }
    
    // Открываем файл для записи визуализации
    FILE *vis_file = fopen(output_filename, "w");
    if (vis_file == NULL) {
        fprintf(stderr, "Ошибка открытия файла %s для записи\n", output_filename);
        fclose(csv_file);
        return;
    }
    
    // Пропускаем названия столбцов в csv
    char buffer[256];
    if (fgets(buffer, sizeof(buffer), csv_file) == NULL) {
        fclose(csv_file);
        fclose(vis_file);
        return;
    }
    
    // Создаем матрицу для визуализации
    char** canvas = (char**)malloc(VIS_HEIGHT * sizeof(char*));
    for (int i = 0; i < VIS_HEIGHT; i++) {
        canvas[i] = (char*)malloc((VIS_WIDTH + 1) * sizeof(char));
        for (int j = 0; j < VIS_WIDTH; j++) {
            canvas[i][j] = ' ';
        }
        canvas[i][VIS_WIDTH] = '\0';
    }
    
    int point_count = 0;
    double min_x = X_MAX, max_x = X_MIN;
    double min_y = Y_MAX, max_y = Y_MIN;
    
    // Читаем точки и заполняем canvas
    while (fgets(buffer, sizeof(buffer), csv_file) != NULL) {
        double x, y;
        if (sscanf(buffer, "%lf,%lf", &x, &y) == 2) {
            point_count++;
            
            // Обновляем границы
            if (x < min_x) min_x = x;
            if (x > max_x) max_x = x;
            if (y < min_y) min_y = y;
            if (y > max_y) max_y = y;
            
            // Преобразуем координаты в индексы canvas
            int col = (int)((x - X_MIN) / (X_MAX - X_MIN) * (VIS_WIDTH - 1));
            int row = VIS_HEIGHT - 1 - (int)((y - Y_MIN) / (Y_MAX - Y_MIN) * (VIS_HEIGHT - 1));
            
            // Проверяем границы
            if (col >= 0 && col < VIS_WIDTH && row >= 0 && row < VIS_HEIGHT) {
                // Используем разные символы для плотности
                if (canvas[row][col] == ' ') {
                    canvas[row][col] = '.';
                } else if (canvas[row][col] == '.') {
                    canvas[row][col] = ':';
                } else if (canvas[row][col] == ':') {
                    canvas[row][col] = '*';
                } else if (canvas[row][col] == '*') {
                    canvas[row][col] = '#';
                } else if (canvas[row][col] == '#') {
                    canvas[row][col] = '@';
                }
            }
        }
    }
    
    fclose(csv_file);
    
    fprintf(vis_file, "Количество найденных точек: %d\n", point_count);
    fprintf(vis_file, "Границы найденных точек: x=[%.3f, %.3f], y=[%.3f, %.3f]\n\n", min_x, max_x, min_y, max_y);
    
    // Рисуем ASCII-арт в файл
    fprintf(vis_file, "    y↑\n");
    
    // Выводим canvas с осью Y
    for (int i = 0; i < VIS_HEIGHT; i++) {
        // Метка оси Y через каждые 5 строк
        if (i % 5 == 0) {
            double y_val = Y_MAX - (double)i / (VIS_HEIGHT - 1) * (Y_MAX - Y_MIN);
            fprintf(vis_file, "%5.1f ", y_val);
        } else {
            fprintf(vis_file, "      ");
        }
        
        // Выводим строку canvas
        fprintf(vis_file, "%s\n", canvas[i]);
    }
    
    // Выводим ось X
    fprintf(vis_file, "\n     ");
    for (int j = 0; j < VIS_WIDTH; j++) {
        if (j % 10 == 0) {
            fprintf(vis_file, "|");
        } else {
            fprintf(vis_file, " ");
        }
    }
    fprintf(vis_file, "→ x\n      ");
    
    for (int j = 0; j < VIS_WIDTH; j += 10) {
        double x_val = X_MIN + (double)j / (VIS_WIDTH - 1) * (X_MAX - X_MIN);
        fprintf(vis_file, "%-10.1f", x_val);
    }
    fprintf(vis_file, "\n");
    
    // Легенда
    fprintf(vis_file, "\nЛегенда:\n");
    fprintf(vis_file, "  ' ' - нет точек\n");
    fprintf(vis_file, "  '.' - 1 точка\n");
    fprintf(vis_file, "  ':' - 2-3 точки\n");
    fprintf(vis_file, "  '*' - 4-5 точек\n");
    fprintf(vis_file, "  '#' - 6-10 точек\n");
    fprintf(vis_file, "  '@' - более 10 точек\n");
    
    fclose(vis_file);
    
    // Также выводим краткую информацию в консоль
    printf("Визуализация сохранена в файл: %s\n", output_filename);
    printf("Количество визуализированных точек: %d\n", point_count);
    printf("Использована область: x=[%.1f, %.1f], y=[%.1f, %.1f]\n", X_MIN, X_MAX, Y_MIN, Y_MAX);
    
    // Освобождаем память
    for (int i = 0; i < VIS_HEIGHT; i++) {
        free(canvas[i]);
    }
    free(canvas);
}

// Проверка принадлежности точки к множеству Мандельброта
int is_in_mandelbrot(Complex c) {
    Complex z = {0.0, 0.0};
    double temp_real;
    
    for (int i = 0; i < MAX_ITER; i++) {
        temp_real = z.real * z.real - z.imag * z.imag + c.real; // Пусть z = a + bi, a c = c + di, тогда
        z.imag = 2.0 * z.real * z.imag + c.imag;                // z=z^2 + c -> (a+bi)^2 + c + di = (a^2 - b^2) + c   +   (2*a*b + d)*i
        z.real = temp_real;                                     //                                  настоящая часть           мнимая
        
        // Если |z| > 2, точка не принадлежит множеству(чтобы не считать корень берём |z|^2 > 4)
        if (z.real * z.real + z.imag * z.imag > 4.0) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Использование: %s nthreads npoints\n", argv[0]);
        return 1;
    }
    
    int nthreads = atoi(argv[1]);
    int npoints = atoi(argv[2]);
    
    // Сетка для равномерного распределения точек по всем координатам
    int grid_size = (int)sqrt(npoints);
    if (grid_size * grid_size < npoints) {
        grid_size++;
    }
    
    // Шаги между точками для координат х и у 
    double dx = (X_MAX - X_MIN) / (grid_size - 1);
    double dy = (Y_MAX - Y_MIN) / (grid_size - 1);
    
    Point *points = (Point*)malloc(grid_size * grid_size * sizeof(Point));
    int *is_mandelbrot = (int*)malloc(grid_size * grid_size * sizeof(int));
    
    int index = 0;
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            points[index].x = X_MIN + i * dx; // Заполняем всю ось х в пределах наших констант точками
            points[index].y = Y_MIN + j * dy; // Заполняем всю ось y в пределах наших констант точками
            index++;
        }
    }

    #pragma omp barrier // Барьер для замера времени
    double start = omp_get_wtime();
    
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic) // Параллельный цикл на nthreads потоков с
    for (int idx = 0; idx < grid_size * grid_size; idx++) {          // динамическим распределением итераций
        Complex c = {points[idx].x, points[idx].y};
        is_mandelbrot[idx] = is_in_mandelbrot(c);
    }
    
    double end = omp_get_wtime();
    
    // Считаем точки в заданном множестве
    int count = 0;
    for (int i = 0; i < grid_size * grid_size; i++) {
        if (is_mandelbrot[i]) {
            count++;
        }
    }
    
    FILE *csv_file = fopen("mandelbrot_points.csv", "w");
    if (csv_file == NULL) {
        fprintf(stderr, "Ошибка открытия файла для записи\n");
        free(points);
        free(is_mandelbrot);
        return 1;
    }
    
    
    // Записываем найденные точки
    fprintf(csv_file, "x,y\n");
    for (int i = 0; i < grid_size * grid_size; i++) {
        if (is_mandelbrot[i]) {
            fprintf(csv_file, "%.15f,%.15f\n", points[i].x, points[i].y);
        }
    }
    
    fclose(csv_file);
    
    // Вывод статистики
    printf("Проверено точек: %d\n", grid_size * grid_size);
    printf("Найдено точек Мандельброта: %d\n", count);
    printf("Время выполнения: %.5f секунд\n", end - start);
    
    visualize_mandelbrot("mandelbrot_points.csv", "mandelbrot_visualization.txt");

    free(points);
    free(is_mandelbrot);
    return 0;
}

