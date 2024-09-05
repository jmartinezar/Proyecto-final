#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <omp.h> // Include OpenMP header

int main(int argc, char *argv[]) {
    // Tamaño de los vectores
    int vector_size = std::atoi(argv[1]);
    size_t vector_memory_size = vector_size * sizeof(double);

    // Asignar memoria para los vectores en el host utilizando Eigen
    Eigen::VectorXd host_vector_A(vector_size);
    Eigen::VectorXd host_vector_B(vector_size);
    Eigen::VectorXd host_vector_C(vector_size);

    // Inicializar los vectores en el host (parallelized with OpenMP)
    #pragma omp parallel for
    for (int i = 0; i < vector_size; i++) {
        host_vector_A[i] = i;
        host_vector_B[i] = i * 2.0f;
    }

    // Medir el tiempo de ejecución
    auto start_time = std::chrono::system_clock::now(); // inicio del tiempo

    // Realizar la suma de vectores utilizando Eigen (can be parallelized as well)
    #pragma omp parallel for
    for (int i = 0; i < vector_size; i++) {
        host_vector_C[i] = host_vector_A[i] + host_vector_B[i];
    }

    auto end_time = std::chrono::system_clock::now(); // fin del tiempo
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Tiempo total de ejecución
    double execution_time = elapsed_time.count();
    
    // Imprimir algunos elementos del vector resultado
    for (int i = 0; i < 10; i++) {
        std::cerr << host_vector_A[i] << " + " << host_vector_B[i] << " = " << host_vector_C[i] << std::endl;
    }

    // Imprimir el tamaño y el tiempo transcurrido en la suma de vectores
    std::cout << vector_memory_size << "\t" << execution_time << std::endl;

    return 0;
}