cmake_minimum_required(VERSION 3.5)

project(test LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_library(LIBAMGXSH amgxsh PATHS $ENV{AMGX_DIR}/build)

add_compile_options(-arch=sm_70)

include_directories(
    ${MPI_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
    $ENV{AMGX_DIR}/include
    $ENV{DF_ROOT}/src_gpu
)

add_library(${PROJECT_NAME} 
    STATIC 
        dfTestEqn.cu
        $ENV{DF_ROOT}/src_gpu/dfMatrixDataBase.cu)

target_link_libraries(${PROJECT_NAME}
    ${MPI_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${LIBAMGXSH}
)