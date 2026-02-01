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

To run the project, you will need to have MPI installed on your system. You can then run the project using the `mpirun` command.

```bash
mpirun -np <num_processes> ./<executable_name>
```

