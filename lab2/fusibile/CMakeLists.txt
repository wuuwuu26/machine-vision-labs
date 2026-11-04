# 现代版 CMakeLists（替换原 legacy 写法，兼容 CMake >= 3.18 + CUDA 11/12）
# 原版用 find_package(CUDA) + cuda_add_executable，该 API 已在 CMake >= 3.27 移除。
cmake_minimum_required(VERSION 3.18)
project(fusibile LANGUAGES CXX CUDA)

# ---- OpenCV（宿主机需安装 libopencv-dev，OpenCV 3/4 均可）----
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ---- CUDA 头文件：主机端 .cpp（main.cpp 含 #include <cuda_runtime.h>）也需要 ----
# nvcc 只对 .cu 自动加 CUDA 头路径；g++ 编 main.cpp 时必须显式给出。
# 从 nvcc 二进制位置反推 targets/x86_64-linux/include（CUDA 11/12 标准布局）。
if(DEFINED CMAKE_CUDA_COMPILER)
    get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" PATH)
    include_directories("${CUDA_COMPILER_BIN}/../targets/x86_64-linux/include")
elseif(DEFINED CUDAToolkit_ROOT)
    include_directories("${CUDAToolkit_ROOT}/targets/x86_64-linux/include")
endif()

# ---- 标准 ----
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- 目标 GPU 架构：60=原 Pascal；89=RTX 40 (Ada)。
#      若宿主机 CUDA < 11.8 不支持 sm_89，可改为 60（仍能在 4070 上兼容运行，略慢）----
set(CMAKE_CUDA_ARCHITECTURES 60 89)

# ---- NVCC 选项（贴近原版）----
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall")

# ---- OpenMP ----
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ---- 构建（头文件自动发现，仅列源文件）----
add_executable(fusibile
    fusibile.cu
    main.cpp
)

target_link_libraries(fusibile ${OpenCV_LIBS})
