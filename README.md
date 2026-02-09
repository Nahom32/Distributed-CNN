# Distributed-CNN

An Implementation of a parallelized training for the CNN architecture. This project is a distributed and parallelized training scheme for Convolutional Neural Networks (CNNs) built using C, OpenMP, and MPI.

## Key Features

- **Distributed Training:** Utilizes MPI for distributed training across multiple nodes.
- **Parallelized Training:** Leverages OpenMP for parallelization within a single node.
- **C-based CNN:** The CNN is implemented from scratch in C.
- **Modular Design:** The project is organized into modules for layers, math operations, and data loading.

## Project Structure

The project is organized as follows:

- **`include/`**: Contains the header files for the project.
  - **`cnn.h`**: Header file for the CNN model.
  - **`math_ops.h`**: Header file for math operations.
  - **`mpi_ops.h`**: Header file for MPI operations.
  - **`layers/`**: Contains the header files for the different layers of the CNN.
    - **`conv.h`**: Header file for the convolutional layer.
    - **`full.h`**: Header file for the fully connected layer.
    - **`pool.h`**: Header file for the pooling layer.
  - **`utils/`**: Contains the header files for utility functions.
    - **`data_loader.h`**: Header file for the data loader.
- **`src/`**: Contains the source code for the project.
  - **`cnn.c`**: Source code for the CNN model.
  - **`main.c`**: Main entry point for the project.
  - **`layers/`**: Contains the source code for the different layers of the CNN.
    - **`conv.c`**: Source code for the convolutional layer.
    - **`full.c`**: Source code for the fully connected layer.
    - **`pool.c`**: Source code for the pooling layer.
  - **`utils/`**: Contains the source code for utility functions.
    - **`data_loader.c`**: Source code for the data loader.

## How to Build

The project uses CMake to build the source code. To build the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Nahom32/Distributed-CNN.git
   ```

2. **Create a build directory:**

   ```bash
   cd Distributed-CNN
   mkdir build
   cd build
   ```

3. **Run CMake and make:**

   ```bash
   cmake ..
   make
   ```

## How to Run

To run the project, you will need to have MPI installed on your system. You can then run the project using the `mpirun` command. The executable `cnn_dist` takes the number of epochs as a command line argument.

```bash
mpirun -np <num_processes> ./cnn_dist <epochs>
```

Example:
```bash
mpirun -np 4 ./cnn_dist 5
```

## Benchmarks

### Performance Comparison

| Execution Mode           | Processors | Total Time (s) | Speedup Factor |
| ------------------------ | ---------- | -------------- | -------------- |
| Serial Baseline          | 1          | 163.60         | 1.00x          |
| Distributed (MPI+OpenMP) | 4          | 45.97          | **3.56x**      |

### Training Convergence Comparison

| Epoch | Serial Training Loss | Serial Training Accuracy (%) | Distributed Training Loss | Distributed Training Accuracy (%) |
| ----- | -------------------- | ---------------------------- | ------------------------- | ------------------------------- |
| 1     | 0.5432               | 90.52                        | 1.0758                    | 88.19                           |
| 2     | 0.3061               | 91.50                        | 0.3996                    | 90.30                           |
| 3     | 0.2737               | 92.46                        | 0.3483                    | 90.90                           |
| 4     | 0.2452               | 93.21                        | 0.3262                    | 91.29                           |
| 5     | 0.2174               | 93.97                        | 0.3118                    | 91.55                           |