find_package(OpenVINO REQUIRED COMPONENTS Runtime)

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
else()
    message(WARNING "OpenVINO not found, iNPU backend will not be built")
endif()
