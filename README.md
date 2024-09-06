# Análisis de Alto Rendimiento con CUDA

Este repositorio contiene implementaciones en CUDA para la suma de dos vectores y la multiplicación de dos matrices. El objetivo del proyecto es analizar conceptos clave de la computación de alto rendimiento, incluyendo escalamiento del problema, speedup y eficiencia.

## Estructura del Repositorio

El repositorio está organizado de la siguiente manera:

- `CPU/`: Contiene el código paralelizado con CPU.
- `GPU/`: Contiene la implementación realizada en CUDA.
- `aux/`: Guarda archivos auxiliales (ej: fuentes).
- `figs`: Directorio en que se guardan las gráficas.
- `data/`: Directorio para almacenar los datasets.
- `logs/`: Los archivos contienen la salida estándar de error.
- `report/`: Informe.
- `tmp/`: Almacena archivos temporales.
- `Makefile`: Archivo make que automatiza la ejecución.
- `plot.py`: Crealas gráficas.

## Dependencias

Para compilar y ejecutar el código en este repositorio, necesitarás:

- `nvcc 11.2.67`
- `python3 3.9.2`
- `Eigen 3.4.0`
- `g++ 10.2.1`
- `GNU Make 4.3`
- `OpenMP 4.5`

## Compilación

Para la generación de las gráficas de análisis es necesario ejecutar los siguientes comandos.

```
make
make plot
```

## Enlace al repositorio
https://github.com/jmartinezar/Proyecto-final