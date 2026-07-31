#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_OVGPU_NAME "OVGPU"

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_ovgpu_reg(void);

#ifdef __cplusplus
}
#endif
