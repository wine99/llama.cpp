find_package(OpenVINO REQUIRED COMPONENTS Runtime)

option(GGML_INPU_GPU "Enable OpenVINO GPU support in iNPU backend (requires OpenCL)" OFF)

if(OpenVINO_FOUND)
    message(STATUS "OpenVINO found for iNPU backend, version: ${OpenVINO_VERSION}")

    ggml_add_backend_library(ggml-inpu
        ggml-inpu.cpp
        ggml-inpu-cache.cpp
        ggml-inpu-debug.cpp
        ggml-inpu-translate.cpp
    )

    target_link_libraries(ggml-inpu PRIVATE openvino::runtime)
    target_include_directories(ggml-inpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    if(GGML_INPU_GPU)
        find_package(OpenCL REQUIRED)
        if(OpenCL_FOUND)
            message(STATUS "OpenCL found for iNPU GPU support, version: ${OpenCL_VERSION_STRING}")
            target_link_libraries(ggml-inpu PRIVATE OpenCL::OpenCL)
            target_compile_definitions(ggml-inpu PRIVATE GGML_INPU_GPU_ENABLED CL_TARGET_OPENCL_VERSION=300)
        else()
            message(FATAL_ERROR "OpenCL not found but GGML_INPU_GPU=ON requires it")
        endif()
    endif()
else()
    message(WARNING "OpenVINO not found, iNPU backend will not be built")
endif()
