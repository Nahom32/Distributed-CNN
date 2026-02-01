cmake_minimum_required(VERSION 3.15)
project(CNNDistributedC C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# --- MACOS APPLE SILICON FIX ---
# We must set these variables BEFORE calling find_package(OpenMP)
if(APPLE)
  # 1. Point to the root of the installation
  set(OpenMP_ROOT "/opt/homebrew/opt/libomp")

  # 2. Explicitly define the library path (This fixes your specific error)
  set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")

  # 3. Define the flags manually just in case
  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_C_LIB_NAMES "omp")

  # 4. Help the compiler find headers/libs
  include_directories("/opt/homebrew/opt/libomp/include")
  link_directories("/opt/homebrew/opt/libomp/lib")
endif()
# -------------------------------

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${MPI_C_INCLUDE_PATH}
)

set(SOURCES
    src/main.c
    src/cnn.c
    src/layers/conv.c
    src/layers/pool.c
    src/layers/full.c
    src/utils/data_loader.c
)

add_executable(cnn_dist ${SOURCES})

# Explicitly link the specific library file we found/set
if(APPLE)
  target_link_libraries(cnn_dist PRIVATE MPI::MPI_C ${OpenMP_omp_LIBRARY})
else()
  target_link_libraries(cnn_dist PRIVATE MPI::MPI_C OpenMP::OpenMP_C)
endif()

message(STATUS "OpenMP Library Found at: ${OpenMP_omp_LIBRARY}")
