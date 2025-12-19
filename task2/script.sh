# Конфигурация
T="5.0"
SIZES=(100 300 500 700 1000)
THREADS=(1 2 4 8)

# Генерация данных
for size in "${SIZES[@]}"; do
    python3 data_generator.py $size
done

# Компиляция
gcc -o nbody_omp open_mp.c -lm -fopenmp -O2
nvcc -o nbody_cuda cuda.cu -O2 -arch=sm_75

# Запуск тестов
for size in "${SIZES[@]}"; do
    echo "=== Тест с ${size} частицами ==="
    
    # OpenMP
    echo "OpenMP:"
    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t
        ./nbody_omp $T "data_${size}.txt" $t
    done
    
    # CUDA
    echo "CUDA:"
    ./nbody_cuda $T "data_${size}.txt"
    
    echo ""
done

# Очистка
rm -f nbody_omp nbody_cuda
echo "Тестирование завершено"