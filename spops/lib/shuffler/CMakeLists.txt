project(shuffler CUDA CXX C)
SET(CMAKE_CXX_STANDARD 20)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    message("SETTING CUDA ARCH 1")
endif ()

cmake_minimum_required(VERSION 3.26)


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -G -O0")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

add_library(shuffler_lib INTERFACE)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g --generate-line-info")


#target_compile_options(shuffler_lib INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
#        --generate-line-info
#        -G
#        -g
#        -ggdb
#        >)


target_include_directories(shuffler_lib INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include")
target_sources(shuffler_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor_lib.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utility.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_timer.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/spadd.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/reduction.cu)


set(executables reduction bcast test_shuffler_spmm shuffle test_matmul test_v2 test_pipeline test_memcpy test_spadd)
# test_bits)
foreach (executable ${executables})
    add_executable(${executable} test/${executable}.cu)
    target_link_libraries(${executable} shuffler_lib)
endforeach ()


