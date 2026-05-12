// OpenVINO GPU Remote Tensor KV Cache Bug Reproducer
//
// This reproducer demonstrates a bug where using the same remote tensor as both
// input and output (in-place KV cache update) produces incorrect results on GPU,
// while CPU produces correct results.
//
// Bug description:
// - Model: Phi-3-mini-4k-instruct (32 layers, 3072 head_dim, f16 KV cache)
// - The model takes KV cache tensors as both inputs and outputs (in-place update)
// - On CPU: outputs are correct
// - On GPU with remote tensors shared between input/output: outputs are WRONG
//   - The newly written row in the KV cache is corrupted
//   - All layers, all K and V caches are affected
//
// Build:
//   g++ -std=c++17 -O2 reproducer.cpp -o reproducer \
//       $(pkg-config --cflags --libs openvino) -I/path/to/openvino/include
//
// Or with cmake (see CMakeLists.txt)
//
// Usage:
//   ./reproducer <model.xml> <input_dir> [output_dir]
//
// The input_dir should contain the reproducer_input_*.bin files.
// KV cache outputs are dumped to output_dir (default: ./output_cpu and ./output_gpu)

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace fs = std::filesystem;

// Read a binary tensor file written by the dump code
// Format: [uint32 et_str_len][char[] et_str][uint32 ndims][uint64[] dims][raw data]
struct TensorData {
    std::string element_type_str;
    std::vector<size_t> shape;
    std::vector<char> data;

    ov::element::Type get_ov_type() const {
        if (element_type_str == "f16") return ov::element::f16;
        if (element_type_str == "f32") return ov::element::f32;
        if (element_type_str == "i32") return ov::element::i32;
        if (element_type_str == "i64") return ov::element::i64;
        if (element_type_str == "bf16") return ov::element::bf16;
        if (element_type_str == "u8") return ov::element::u8;
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

    // Read remaining data
    auto pos = fin.tellg();
    fin.seekg(0, std::ios::end);
    auto end = fin.tellg();
    size_t data_size = end - pos;
    fin.seekg(pos);
    td.data.resize(data_size);
    fin.read(td.data.data(), data_size);

    return td;
}

// Parse reproducer_meta.txt
struct MetaInfo {
    size_t n_inputs;
    size_t n_outputs;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

MetaInfo load_meta(const std::string& filepath) {
    MetaInfo meta;
    std::ifstream fin(filepath);
    std::string line;
    while (std::getline(fin, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        auto key = line.substr(0, eq);
        auto val = line.substr(eq + 1);
        if (key == "n_inputs") meta.n_inputs = std::stoull(val);
        else if (key == "n_outputs") meta.n_outputs = std::stoull(val);
        else if (key.rfind("input_", 0) == 0) meta.input_names.push_back(val);
        else if (key.rfind("output_", 0) == 0) meta.output_names.push_back(val);
    }
    return meta;
}

void dump_kv_output(const std::string& name, const ov::Tensor& tensor,
                    const std::string& output_dir) {
    fs::create_directories(output_dir);
    // Dump binary
    std::string bin_filename = output_dir + "/" + name + ".bin";
    std::ofstream fout(bin_filename, std::ios::binary);
    fout.write(static_cast<const char*>(tensor.data()), tensor.get_byte_size());
    fout.close();

    // Dump text
    std::string txt_filename = output_dir + "/" + name + ".txt";
    std::ofstream tout(txt_filename);
    tout << std::setprecision(8);
    auto shape = tensor.get_shape();
    tout << "name: " << name << ", type: " << tensor.get_element_type().get_type_name()
         << ", shape: " << shape.to_string() << "\n";
    size_t n = tensor.get_size();
    auto et = tensor.get_element_type();
    if (et == ov::element::f16) {
        auto* data = tensor.data<ov::float16>();
        for (size_t i = 0; i < n; i++) tout << static_cast<float>(data[i]) << '\n';
    } else if (et == ov::element::f32) {
        auto* data = tensor.data<float>();
        for (size_t i = 0; i < n; i++) tout << data[i] << '\n';
    } else if (et == ov::element::bf16) {
        auto* data = tensor.data<ov::bfloat16>();
        for (size_t i = 0; i < n; i++) tout << static_cast<float>(data[i]) << '\n';
    }
}

float compute_max_diff_f16(const ov::Tensor& a, const ov::Tensor& b) {
    auto* da = a.data<ov::float16>();
    auto* db = b.data<ov::float16>();
    size_t n = a.get_size();
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = std::abs(static_cast<float>(da[i]) - static_cast<float>(db[i]));
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

void run_inference(const std::string& device,
                   ov::Core& core,
                   std::shared_ptr<ov::Model> model,
                   const MetaInfo& meta,
                   const std::string& input_dir,
                   const std::string& output_dir,
                   bool use_remote_tensor_inplace) {
    std::cout << "\n=== Running on " << device
              << " (remote_inplace=" << use_remote_tensor_inplace << ") ===" << std::endl;

    ov::CompiledModel compiled;
    ov::InferRequest infer_request;

    if (device == "GPU" && use_remote_tensor_inplace) {
        // Use GPU remote context
        auto remote_context = core.get_default_context(device);
        compiled = core.compile_model(model, remote_context);
        infer_request = compiled.create_infer_request();

        // For GPU with remote tensors: allocate remote tensors for KV cache
        // and use the SAME tensor as both input and output (in-place update)
        // This is the bug scenario.
        for (size_t i = 0; i < meta.n_inputs; i++) {
            const auto& name = meta.input_names[i];
            if (name.find("cache_k_") != std::string::npos ||
                name.find("cache_v_") != std::string::npos) {
                // Load initial data
                std::string bin_file = input_dir + "/reproducer_input_" +
                                       std::to_string(i) + "_" + name + ".bin";
                auto td = load_tensor_bin(bin_file);
                auto ov_shape = td.get_ov_shape();
                auto ov_type = td.get_ov_type();

                // Allocate a remote tensor on GPU
                auto remote_tensor = remote_context.create_tensor(ov_type, ov_shape);

                // Copy initial data to the remote tensor
                ov::Tensor host_tensor(ov_type, ov_shape, td.data.data());
                host_tensor.copy_to(remote_tensor);

                // Set as input
                infer_request.set_tensor(name, remote_tensor);

                // Find matching output and set the SAME remote tensor as output
                // (this is the in-place update pattern that triggers the bug)
                for (size_t j = 0; j < meta.n_outputs; j++) {
                    if (meta.output_names[j] == name) {
                        infer_request.set_output_tensor(j, remote_tensor);
                        break;
                    }
                }
            }
        }

        // Set non-KV inputs from files (as host tensors)
        for (size_t i = 0; i < meta.n_inputs; i++) {
            const auto& name = meta.input_names[i];
            if (name.find("cache_k_") == std::string::npos &&
                name.find("cache_v_") == std::string::npos) {
                std::string bin_file = input_dir + "/reproducer_input_" +
                                       std::to_string(i) + "_" + name + ".bin";
                auto td = load_tensor_bin(bin_file);
                ov::Tensor tensor(td.get_ov_type(), td.get_ov_shape(), td.data.data());
                // Need to copy since td.data will go out of scope
                ov::Tensor persistent_tensor(td.get_ov_type(), td.get_ov_shape());
                std::memcpy(persistent_tensor.data(), td.data.data(), td.data.size());
                infer_request.set_tensor(name, persistent_tensor);
            }
        }
    } else {
        // CPU path or GPU without remote tensor inplace
        compiled = core.compile_model(model, device);
        infer_request = compiled.create_infer_request();

        // Load all inputs from files
        for (size_t i = 0; i < meta.n_inputs; i++) {
            const auto& name = meta.input_names[i];
            std::string bin_file = input_dir + "/reproducer_input_" +
                                   std::to_string(i) + "_" + name + ".bin";
            auto td = load_tensor_bin(bin_file);
            ov::Tensor tensor(td.get_ov_type(), td.get_ov_shape());
            std::memcpy(tensor.data(), td.data.data(), td.data.size());
            infer_request.set_tensor(name, tensor);

            // For KV cache outputs: use the SAME host tensor as output (in-place)
            if (name.find("cache_k_") != std::string::npos ||
                name.find("cache_v_") != std::string::npos) {
                for (size_t j = 0; j < meta.n_outputs; j++) {
                    if (meta.output_names[j] == name) {
                        infer_request.set_output_tensor(j, tensor);
                        break;
                    }
                }
            }
        }
    }

    std::cout << "Running inference..." << std::endl;
    infer_request.infer();
    infer_request.infer();
    std::cout << "Inference done." << std::endl;

    // Dump KV cache outputs
    for (size_t i = 0; i < meta.n_outputs; i++) {
        const auto& name = meta.output_names[i];
        if (name.find("cache_k_") != std::string::npos ||
            name.find("cache_v_") != std::string::npos) {
            ov::Tensor out_tensor = infer_request.get_output_tensor(i);
            // For remote tensors, copy to host first
            ov::Tensor host_tensor;
            try {
                (void)out_tensor.data();
                host_tensor = out_tensor;
            } catch (...) {
                host_tensor = ov::Tensor(out_tensor.get_element_type(), out_tensor.get_shape());
                out_tensor.copy_to(host_tensor);
            }
            dump_kv_output(name, host_tensor, output_dir);
        }
    }
    std::cout << "KV cache outputs dumped to: " << output_dir << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.xml> <input_dir> [output_base_dir]" << std::endl;
        std::cerr << "  input_dir should contain reproducer_input_*.bin and reproducer_meta.txt" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_dir = argv[2];
    std::string output_base = argc > 3 ? argv[3] : ".";
    std::string cpu_output_dir = output_base + "/output_cpu";
    std::string gpu_output_dir = output_base + "/output_gpu";

    // Load meta
    std::string meta_path = input_dir + "/reproducer_meta.txt";
    auto meta = load_meta(meta_path);
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Inputs: " << meta.n_inputs << ", Outputs: " << meta.n_outputs << std::endl;

    // Load model
    ov::Core core;
    auto model = core.read_model(model_path);
    std::cout << "Model loaded successfully." << std::endl;

    // Run on CPU (baseline - always correct)
    run_inference("CPU", core, model, meta, input_dir, cpu_output_dir, false);

    // Run on GPU with remote tensor in-place (buggy path)
    run_inference("GPU", core, model, meta, input_dir, gpu_output_dir, true);

    // Compare outputs
    std::cout << "\n=== Comparing CPU vs GPU KV cache outputs ===" << std::endl;
    int mismatches = 0;
    for (size_t i = 0; i < meta.n_outputs; i++) {
        const auto& name = meta.output_names[i];
        if (name.find("cache_k_") == std::string::npos &&
            name.find("cache_v_") == std::string::npos) continue;

        std::string cpu_file = cpu_output_dir + "/" + name + ".bin";
        std::string gpu_file = gpu_output_dir + "/" + name + ".bin";

        // Read and compare
        std::ifstream cpu_fin(cpu_file, std::ios::binary);
        std::ifstream gpu_fin(gpu_file, std::ios::binary);
        if (!cpu_fin.is_open() || !gpu_fin.is_open()) {
            std::cerr << "Cannot open output files for: " << name << std::endl;
            continue;
        }

        cpu_fin.seekg(0, std::ios::end);
        size_t size = cpu_fin.tellg();
        cpu_fin.seekg(0);

        std::vector<char> cpu_data(size), gpu_data(size);
        cpu_fin.read(cpu_data.data(), size);
        gpu_fin.read(gpu_data.data(), size);

        // Compute max diff (assuming f16)
        size_t n_elements = size / 2;  // f16 = 2 bytes
        float max_diff = 0.0f;
        size_t max_diff_idx = 0;
        auto* cpu_f16 = reinterpret_cast<ov::float16*>(cpu_data.data());
        auto* gpu_f16 = reinterpret_cast<ov::float16*>(gpu_data.data());
        for (size_t j = 0; j < n_elements; j++) {
            float diff = std::abs(static_cast<float>(cpu_f16[j]) - static_cast<float>(gpu_f16[j]));
            if (diff > max_diff) {
                max_diff = diff;
                max_diff_idx = j;
            }
        }

        if (max_diff > 0.1f) {
            // shape is [1,1,256,3072], so row = idx / 3072
            size_t row = max_diff_idx / 3072;
            std::cout << "MISMATCH " << name
                      << ": max_diff=" << std::fixed << std::setprecision(4) << max_diff
                      << " at element " << max_diff_idx
                      << " (row " << row << ")"
                      << std::endl;
            mismatches++;
        }
    }

    if (mismatches == 0) {
        std::cout << "All KV cache outputs match between CPU and GPU." << std::endl;
    } else {
        std::cout << "\nTotal mismatches: " << mismatches << " / "
                  << meta.n_outputs - 1 << " KV outputs" << std::endl;
        std::cout << "\nBUG CONFIRMED: GPU remote tensor in-place KV cache update "
                  << "produces different results from CPU." << std::endl;
    }

    return mismatches > 0 ? 1 : 0;
}
