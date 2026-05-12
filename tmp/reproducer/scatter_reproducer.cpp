// Minimal reproducer for OpenVINO GPU ScatterUpdate bug with remote tensors
//
// The bug: when using a remote tensor as both input[0] and output[0] (in-place
// ScatterUpdate), GPU produces wrong results even on the first infer call.
// The written row AND other rows are corrupted. CPU produces correct results.
//
// Model: single ScatterUpdate node
//   data:    [1,1,256,3072] f16 (KV cache - loaded from file)
//   indices: scalar = 5
//   updates: [1,1,1,3072] f16 (random values, seed=42)
//   axis:    2
//
// Build:
//   mkdir build && cd build
//   cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenVINO_DIR=<path>
//   make
//
// Usage:
//   ./scatter_reproducer <path_to_kv_cache_data.bin>

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/op/scatter_update.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/constant.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <random>

struct TensorData {
    std::string element_type_str;
    std::vector<size_t> shape;
    std::vector<char> data;

    ov::element::Type get_ov_type() const {
        if (element_type_str == "f16") return ov::element::f16;
        if (element_type_str == "f32") return ov::element::f32;
        if (element_type_str == "i32") return ov::element::i32;
        if (element_type_str == "i64") return ov::element::i64;
        throw std::runtime_error("Unknown element type: " + element_type_str);
    }

    ov::Shape get_ov_shape() const {
        return ov::Shape(shape.begin(), shape.end());
    }
};

TensorData load_tensor_bin(const std::string& filepath) {
    TensorData td;
    std::ifstream fin(filepath, std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open: " + filepath);
    }

    uint32_t et_len;
    fin.read(reinterpret_cast<char*>(&et_len), sizeof(et_len));
    td.element_type_str.resize(et_len);
    fin.read(td.element_type_str.data(), et_len);

    uint32_t ndims;
    fin.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
    td.shape.resize(ndims);
    for (uint32_t i = 0; i < ndims; i++) {
        uint64_t dim;
        fin.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        td.shape[i] = static_cast<size_t>(dim);
    }

    auto pos = fin.tellg();
    fin.seekg(0, std::ios::end);
    auto end = fin.tellg();
    size_t data_size = end - pos;
    fin.seekg(pos);
    td.data.resize(data_size);
    fin.read(td.data.data(), data_size);

    return td;
}

// Build a model with a single ScatterUpdate node:
//   Parameter(data) [1,1,-1,3072] f16
//   Parameter(updates) [1,1,1,3072] f16
//   Constant(indices) = 5, i32
//   Constant(axis) = 2, i32
//   ScatterUpdate(data, indices, updates, axis) -> Result
std::shared_ptr<ov::Model> build_scatter_model() {
    auto data_param = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f16, ov::PartialShape{1, 1, -1, 3072});
    data_param->set_friendly_name("data");

    auto updates_param = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f16, ov::PartialShape{1, 1, 1, 3072});
    updates_param->set_friendly_name("updates");

    auto indices_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::i32, ov::Shape{1}, std::vector<int32_t>{5});

    auto axis_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::i32, ov::Shape{1}, std::vector<int32_t>{2});

    auto scatter = std::make_shared<ov::op::v3::ScatterUpdate>(
        data_param, indices_const, updates_param, axis_const);

    auto result = std::make_shared<ov::op::v0::Result>(scatter);
    result->set_friendly_name("output");

    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{result},
        ov::ParameterVector{data_param, updates_param},
        "scatter_update_reproducer");

    return model;
}

void dump_row(const std::string& label, const ov::Tensor& tensor, size_t row, size_t cols = 10) {
    auto* data = tensor.data<ov::float16>();
    size_t total_cols = tensor.get_shape()[3];  // [1,1,256,3072]
    size_t offset = row * total_cols;
    std::cout << label << " row " << row << " (first " << cols << " elements): ";
    for (size_t i = 0; i < cols && i < total_cols; i++) {
        std::cout << std::fixed << std::setprecision(4)
                  << static_cast<float>(data[offset + i]) << " ";
    }
    std::cout << std::endl;
}

void dump_tensor_txt(const std::string& filepath, const std::string& name,
                     const ov::Tensor& tensor) {
    std::ofstream out(filepath);
    out << std::setprecision(8);
    auto shape = tensor.get_shape();
    out << "name: " << name << ", type: " << tensor.get_element_type().get_type_name()
        << ", shape: " << shape.to_string() << "\n";
    size_t n = tensor.get_size();
    auto et = tensor.get_element_type();
    if (et == ov::element::f16) {
        auto* data = tensor.data<ov::float16>();
        for (size_t i = 0; i < n; i++) out << static_cast<float>(data[i]) << '\n';
    } else if (et == ov::element::f32) {
        auto* data = tensor.data<float>();
        for (size_t i = 0; i < n; i++) out << data[i] << '\n';
    }
}

float max_diff_tensors(const ov::Tensor& a, const ov::Tensor& b) {
    auto* da = a.data<ov::float16>();
    auto* db = b.data<ov::float16>();
    size_t n = a.get_size();
    float max_d = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = std::abs(static_cast<float>(da[i]) - static_cast<float>(db[i]));
        max_d = std::max(max_d, diff);
    }
    return max_d;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <reproducer_input_1_cache_k_l0.bin>" << std::endl;
        return 1;
    }

    std::string data_file = argv[1];

    // Load KV cache data
    std::cout << "Loading data from: " << data_file << std::endl;
    auto td = load_tensor_bin(data_file);
    std::cout << "  type=" << td.element_type_str
              << " shape=[" << td.shape[0] << "," << td.shape[1] << ","
              << td.shape[2] << "," << td.shape[3] << "]" << std::endl;

    // Create fixed random updates [1,1,1,3072] f16
    std::vector<ov::float16> updates_data(3072);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : updates_data) {
        v = ov::float16(dist(rng));
    }

    // Build model
    auto model = build_scatter_model();
    std::cout << "Model built: ScatterUpdate(data[1,1,256,3072], indices=5, updates[1,1,1,3072], axis=2)"
              << std::endl;

    ov::Core core;

    // ==================== CPU test ====================
    std::cout << "\n=== CPU: single infer ===" << std::endl;
    ov::Tensor cpu_result;
    {
        auto compiled = core.compile_model(model, "CPU");
        auto infer_request = compiled.create_infer_request();

        ov::Tensor data_tensor(ov::element::f16, td.get_ov_shape());
        std::memcpy(data_tensor.data(), td.data.data(), td.data.size());

        ov::Tensor updates_tensor(ov::element::f16, ov::Shape{1, 1, 1, 3072});
        std::memcpy(updates_tensor.data(), updates_data.data(), 3072 * sizeof(ov::float16));

        infer_request.set_input_tensor(0, data_tensor);
        infer_request.set_input_tensor(1, updates_tensor);
        infer_request.set_output_tensor(0, data_tensor);

        infer_request.infer();
        dump_row("  After infer:", data_tensor, 5);
        dump_tensor_txt("cpu_result.txt", "cpu_result", data_tensor);

        cpu_result = ov::Tensor(ov::element::f16, data_tensor.get_shape());
        std::memcpy(cpu_result.data(), data_tensor.data(), data_tensor.get_byte_size());
    }

    // ==================== GPU test with remote tensor ====================
    std::cout << "\n=== GPU (remote tensor in-place): single infer ===" << std::endl;
    {
        auto remote_context = core.get_default_context("GPU");
        auto compiled = core.compile_model(model, remote_context);
        auto infer_request = compiled.create_infer_request();

        // Allocate remote tensor for data (used as both input and output)
        ov::Shape data_shape = td.get_ov_shape();
        auto remote_data = remote_context.create_tensor(ov::element::f16, data_shape);

        // Copy initial data to remote tensor
        ov::Tensor host_init(ov::element::f16, data_shape, td.data.data());
        host_init.copy_to(remote_data);

        // Set updates as host tensor
        ov::Tensor updates_tensor(ov::element::f16, ov::Shape{1, 1, 1, 3072});
        std::memcpy(updates_tensor.data(), updates_data.data(), 3072 * sizeof(ov::float16));

        // Set remote tensor as both input and output (in-place)
        infer_request.set_input_tensor(0, remote_data);
        infer_request.set_input_tensor(1, updates_tensor);
        infer_request.set_output_tensor(0, remote_data);

        infer_request.infer();

        // Copy result to host
        ov::Tensor gpu_result(ov::element::f16, data_shape);
        remote_data.copy_to(gpu_result);
        dump_row("  After infer:", gpu_result, 5);
        dump_tensor_txt("gpu_result.txt", "gpu_result", gpu_result);

        // Compare CPU vs GPU
        float diff = max_diff_tensors(cpu_result, gpu_result);
        std::cout << "\n  CPU vs GPU max_diff = " << diff << std::endl;
        if (diff > 0.001f) {
            std::cout << "  BUG: GPU result differs from CPU!" << std::endl;
        } else {
            std::cout << "  OK: results match." << std::endl;
        }
    }

    return 0;
}
