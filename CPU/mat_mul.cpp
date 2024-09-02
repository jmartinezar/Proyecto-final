#include <iostream>
#include <chrono>
#include <Eigen/Dense>

int main(int argc, char *argv[])
{
    int matrix_size = std::atoi(argv[1]);

    // Definir las dimensiones de las matrices
    int matrix_A_width = matrix_size;
    int matrix_A_height = matrix_size;
    int matrix_B_width = matrix_size;
    int matrix_B_height = matrix_size;

    // Crear matrices usando Eigen
    Eigen::MatrixXd matrix_A = Eigen::MatrixXd::Constant(matrix_A_height, matrix_A_width, 1.0); // Matriz A con valores 1.0
    Eigen::MatrixXd matrix_B = Eigen::MatrixXd::Constant(matrix_B_height, matrix_B_width, 2.0); // Matriz B con valores 2.0
    Eigen::MatrixXd result_matrix(matrix_A_height, matrix_B_width); // Matriz C para almacenar el resultado

    // Medir el tiempo de ejecución
    auto start_time = std::chrono::system_clock::now(); // inicio del tiempo

    // Multiplicación de matrices usando Eigen
    result_matrix = matrix_A * matrix_B;

    auto end_time = std::chrono::system_clock::now(); // fin del tiempo
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Tiempo total
    double execution_time = elapsed_time.count();

    // Imprimir los primeros 10 valores de la matriz resultado
    for (int i = 0; i < 10; i++)
    {
        std::cerr << "result_matrix[" << i * matrix_B_width + 1 << "] = " << result_matrix(i, 0) << std::endl;
    }

    // Imprimir el tamaño y el tiempo transcurrido en la multiplicación de matrices
    std::cout << matrix_size << "\t" << execution_time << std::endl;

    return 0;
}
