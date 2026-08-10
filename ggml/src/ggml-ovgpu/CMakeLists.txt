find_package(OpenVINO REQUIRED COMPONENTS Runtime Threading)
find_package(OpenCL REQUIRED)

# PoC: consume the clDNN layer of the OV GPU plugin directly (static libs from
# the OV build tree, not the installed plugin .so)
set(GGML_OVGPU_OV_SOURCE_DIR "" CACHE PATH "OpenVINO source root")
set(GGML_OVGPU_OV_BIN_DIR    "" CACHE PATH "OpenVINO build bin dir (bin/intel64/<cfg>)")
if (NOT GGML_OVGPU_OV_SOURCE_DIR)
    message(FATAL_ERROR "GGML_OVGPU requires -DGGML_OVGPU_OV_SOURCE_DIR=<openvino source root>")
endif()
if (NOT GGML_OVGPU_OV_BIN_DIR)
    set(GGML_OVGPU_OV_BIN_DIR "${GGML_OVGPU_OV_SOURCE_DIR}/bin/intel64/Release")
endif()
set(GGML_OVGPU_ONEDNN_GPU_LIB "" CACHE FILEPATH "path to libopenvino_onednn_gpu.a")
if (NOT GGML_OVGPU_ONEDNN_GPU_LIB)
    set(GGML_OVGPU_ONEDNN_GPU_LIB "${GGML_OVGPU_OV_SOURCE_DIR}/build/Release/src/plugins/intel_gpu/thirdparty/onednn_gpu_install/lib/libopenvino_onednn_gpu.a")
endif()

set(OVGPU_INTEL_GPU ${GGML_OVGPU_OV_SOURCE_DIR}/src/plugins/intel_gpu)

# minimal plugin glue sources needed for stray symbols referenced by the graph
# lib (mirrors intel_gpu/tests/unit/CMakeLists.txt)
set(OVGPU_GLUE_SOURCES
    ${OVGPU_INTEL_GPU}/src/plugin/variable_state.cpp
    ${OVGPU_INTEL_GPU}/src/plugin/multi_tensor_variable_state.cpp
    ${OVGPU_INTEL_GPU}/src/plugin/remote_context.cpp
    ${OVGPU_INTEL_GPU}/src/plugin/remote_tensor.cpp
    ${OVGPU_INTEL_GPU}/src/plugin/usm_host_tensor.cpp
    ${OVGPU_INTEL_GPU}/src/plugin/common_utils.cpp
    ${OVGPU_INTEL_GPU}/src/plugin/simple_math.cpp
)

# internal op definitions referenced by the graph lib (kv_cache, gemm, sdpa, ...)
file(GLOB OVGPU_OP_DEF_SOURCES ${OVGPU_INTEL_GPU}/src/plugin/transformations/op/*.cpp)

ggml_add_backend_library(ggml-ovgpu
    ggml-ovgpu.cpp
    ggml-ovgpu-ops.cpp
    ggml-ovgpu-smoke.cpp
    ${OVGPU_GLUE_SOURCES}
    ${OVGPU_OP_DEF_SOURCES}
)

# compile defines must match what the OV graph lib was built with (ABI)
target_compile_definitions(ggml-ovgpu PRIVATE
    ENABLE_ONEDNN_FOR_GPU
    ENABLE_DEBUG_CAPS
    GPU_DEBUG_CONFIG=1
    OV_GPU_WITH_OCL_RT=1
    OV_GPU_USE_OPENCL_HPP
    OV_GPU_OPENCL_HPP_HAS_BUS_INFO
    OV_GPU_OPENCL_HPP_HAS_UUID
    CL_TARGET_OPENCL_VERSION=300
    ENABLE_PROFILING_ITT
    OV_THREAD=OV_THREAD_TBB_ADAPTIVE
    DNNL_DLL
    DNNL_DLL_EXPORTS
    DNNL_ENABLE_CPU_ISA_HINTS
    DNNL_ENABLE_MAX_CPU_ISA
    DNNL_X64=1
    NGEN_CONFIG
)

target_include_directories(ggml-ovgpu SYSTEM PRIVATE
    ${OVGPU_INTEL_GPU}/include
    ${OVGPU_INTEL_GPU}/src
    ${OVGPU_INTEL_GPU}/src/runtime
    ${OVGPU_INTEL_GPU}/src/graph
    ${OVGPU_INTEL_GPU}/src/graph/include
    ${GGML_OVGPU_OV_SOURCE_DIR}/build/Release/src/plugins/intel_gpu/thirdparty/onednn_gpu_install/include
    ${GGML_OVGPU_OV_SOURCE_DIR}/src/inference/dev_api
    ${GGML_OVGPU_OV_SOURCE_DIR}/src/common/util/include
    ${GGML_OVGPU_OV_SOURCE_DIR}/src/core/dev_api
    ${GGML_OVGPU_OV_SOURCE_DIR}/src/common/itt/include
    ${GGML_OVGPU_OV_SOURCE_DIR}/src/common/transformations/include
    ${GGML_OVGPU_OV_SOURCE_DIR}/src/common/shape_inference/include
    ${GGML_OVGPU_OV_SOURCE_DIR}/src/core/shape_inference/include
    ${GGML_OVGPU_OV_SOURCE_DIR}/thirdparty/ocl/clhpp_headers/include
    ${GGML_OVGPU_OV_SOURCE_DIR}/thirdparty/ocl/cl_headers
)

function(ovgpu_find_lib out name)
    find_library(${out} NAMES ${name} PATHS ${GGML_OVGPU_OV_BIN_DIR} REQUIRED NO_DEFAULT_PATH)
endfunction()

ovgpu_find_lib(OVGPU_GRAPH_LIB  openvino_intel_gpu_graph)
ovgpu_find_lib(OVGPU_RUNTIME_LIB openvino_intel_gpu_runtime)
ovgpu_find_lib(OVGPU_KERNELS_LIB openvino_intel_gpu_kernels)
ovgpu_find_lib(OVGPU_ITT_LIB    openvino_itt)
ovgpu_find_lib(OVGPU_SHAPE_INFERENCE_LIB openvino_shape_inference)
ovgpu_find_lib(OVGPU_PUGIXML_LIB pugixml)
ovgpu_find_lib(OVGPU_REFERENCE_LIB openvino_reference)
ovgpu_find_lib(OVGPU_UTIL_LIB openvino_util)
ovgpu_find_lib(OVGPU_SHUTDOWN_LIB openvino_shutdown)

target_link_libraries(ggml-ovgpu PRIVATE
    -Wl,--start-group
    ${OVGPU_GRAPH_LIB}
    ${OVGPU_KERNELS_LIB}
    ${OVGPU_RUNTIME_LIB}
    ${GGML_OVGPU_ONEDNN_GPU_LIB}
    ${OVGPU_SHAPE_INFERENCE_LIB}
    ${OVGPU_ITT_LIB}
    ${OVGPU_PUGIXML_LIB}
    ${OVGPU_REFERENCE_LIB}
    ${OVGPU_UTIL_LIB}
    ${OVGPU_SHUTDOWN_LIB}
    -Wl,--end-group
    openvino::runtime
    openvino::threading
    OpenCL::OpenCL
)
