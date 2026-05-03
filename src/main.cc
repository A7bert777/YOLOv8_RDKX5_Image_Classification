/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

Copyright (c) 2024-2025, WuChao && MaChao D-Robotics.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// 注意: 此程序在RDK板端运行
// Attention: This program runs on RDK board.

// ============================================================================
// Configuration Parameters
// ============================================================================

// D-Robotics *.bin 模型路径
// Path to D-Robotics *.bin model
#define MODEL_PATH "/home/sunrise/RDKX5/rdk_model_zoo-rdk_x5/YOLOv8_RDKX5_Image_Classification/model/yolov8ncls_kongkai.bin"

// 测试图片输入文件夹路径
#define INPUT_FOLDER_PATH "../inputimage"

// 结果保存文件夹路径
#define OUTPUT_FOLDER_PATH "../outputimage"

// ----------------------------------------------------------------------------
// 核心参数：选择 .bin 模型的类型
// 1 = 模型去除了反量化节点 (使用修改后的绝对偏移+动态类型转换逻辑)
// 0 = 模型未去除反量化节点 (使用原始偏移+Float逻辑)
// ----------------------------------------------------------------------------
#define REMOVE_DEQUANT_NODE 0

// 前处理方式: 0=Resize, 1=LetterBox
// Preprocessing method: 0=Resize, 1=LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

// Top K 结果数量
// Number of top K results to display
#define TOP_K 5

// ============================================================================
// Includes
// ============================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstring>
#include <fstream>

// OpenCV
#include <opencv2/opencv.hpp>

// RDK BPU libDNN API
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"

// ============================================================================
// Macros
// ============================================================================

#define CHECK_SUCCESS(value, errmsg)                                         \
    do {                                                                     \
        auto ret_code = value;                                               \
        if (ret_code != 0) {                                                 \
            std::cerr << "\033[1;31m[ERROR]\033[0m " << __FILE__ << ":"     \
                      << __LINE__ << " " << errmsg                           \
                      << ", error code: " << ret_code << std::endl;          \
            return ret_code;                                                 \
        }                                                                    \
    } while (0)

#define LOG_INFO(msg) \
    std::cout << "\033[1;32m[INFO]\033[0m " << msg << std::endl

#define LOG_WARN(msg) \
    std::cout << "\033[1;33m[WARN]\033[0m " << msg << std::endl

#define LOG_ERROR(msg) \
    std::cerr << "\033[1;31m[ERROR]\033[0m " << msg << std::endl

#define LOG_TIME(msg, duration) \
    std::cout << "\033[1;31m" << msg << " = " << std::fixed            \
              << std::setprecision(2) << (duration) << " ms\033[0m"    \
              << std::endl

// ============================================================================
// ImageNet 1000 Classes (abbreviated for brevity, full list in Python demo)
// ============================================================================

const std::vector<std::string> IMAGENET_CLASSES = {
    "down", "up"
};

// ============================================================================
// Classification Result Structure
// ============================================================================

struct ClassificationResult {
    int class_id;
    float probability;
    std::string class_name;

    ClassificationResult(int id, float prob, const std::string& name)
        : class_id(id), probability(prob), class_name(name) {}
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Extract filename without extension from a full path
 */
std::string extractFileNameWithoutExtension(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);
    pos = filename.find_last_of(".");
    if (pos != std::string::npos) {
        filename = filename.substr(0, pos);
    }
    return filename;
}

/**
 * @brief Convert BGR image to NV12 format
 */
cv::Mat bgr2nv12(const cv::Mat& bgr_img) {
    auto start = std::chrono::high_resolution_clock::now();

    int height = bgr_img.rows;
    int width = bgr_img.cols;

    // BGR to YUV420P
    cv::Mat yuv_mat;
    cv::cvtColor(bgr_img, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t* yuv = yuv_mat.ptr<uint8_t>();

    // Allocate NV12 image
    cv::Mat nv12_img(height * 3 / 2, width, CV_8UC1);
    uint8_t* nv12 = nv12_img.ptr<uint8_t>();

    // Copy Y plane
    int y_size = height * width;
    memcpy(nv12, yuv, y_size);

    // Convert UV planar to UV packed (NV12)
    int uv_height = height / 2;
    int uv_width = width / 2;
    uint8_t* nv12_uv = nv12 + y_size;
    uint8_t* u_data = yuv + y_size;
    uint8_t* v_data = u_data + uv_height * uv_width;

    for (int i = 0; i < uv_width * uv_height; i++) {
        *nv12_uv++ = *u_data++;
        *nv12_uv++ = *v_data++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    LOG_TIME("BGR to NV12 time", duration);

    return nv12_img;
}

/**
 * @brief Preprocess image with letterbox or resize
 */
cv::Mat preprocess_image(const cv::Mat& img, int input_h, int input_w,
                         float& x_scale, float& y_scale,
                         int& x_shift, int& y_shift) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result;

    if (PREPROCESS_TYPE == LETTERBOX_TYPE) {
        // Letterbox preprocessing
        x_scale = std::min(1.0f * input_h / img.rows, 1.0f * input_w / img.cols);
        y_scale = x_scale;

        if (x_scale <= 0 || y_scale <= 0) {
            throw std::runtime_error("Invalid scale factor");
        }

        int new_w = static_cast<int>(img.cols * x_scale);
        int new_h = static_cast<int>(img.rows * y_scale);

        x_shift = (input_w - new_w) / 2;
        y_shift = (input_h - new_h) / 2;
        int x_other = input_w - new_w - x_shift;
        int y_other = input_h - new_h - y_shift;

        cv::resize(img, result, cv::Size(new_w, new_h));
        cv::copyMakeBorder(result, result, y_shift, y_other, x_shift, x_other,
                          cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        LOG_TIME("Preprocess (LetterBox) time", duration);

    } else if (PREPROCESS_TYPE == RESIZE_TYPE) {
        // Resize preprocessing
        cv::resize(img, result, cv::Size(input_w, input_h));

        x_scale = 1.0f * input_w / img.cols;
        y_scale = 1.0f * input_h / img.rows;
        x_shift = 0;
        y_shift = 0;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        LOG_TIME("Preprocess (Resize) time", duration);
    }

    LOG_INFO("Scale: x=" << x_scale << ", y=" << y_scale);
    LOG_INFO("Shift: x=" << x_shift << ", y=" << y_shift);

    return result;
}

/**
 * @brief Softmax function
 */
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> result(logits.size());

    // Find max for numerical stability
    float max_val = *std::max_element(logits.begin(), logits.end());

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        result[i] = std::exp(logits[i] - max_val);
        sum += result[i];
    }

    // Normalize
    for (size_t i = 0; i < logits.size(); i++) {
        result[i] /= sum;
    }

    return result;
}

/**
 * @brief Get top K classification results
 * @note The actual k is min(k, probabilities.size()) to avoid out-of-bounds access
 */
std::vector<ClassificationResult> get_topk_results(
    const std::vector<float>& probabilities, int k) {

    int actual_k = std::min(k, static_cast<int>(probabilities.size()));

    // Create index array
    std::vector<int> indices(probabilities.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Partial sort to get top k
    std::partial_sort(indices.begin(), indices.begin() + actual_k, indices.end(),
        [&probabilities](int a, int b) {
            return probabilities[a] > probabilities[b];
        });

    // Create results
    std::vector<ClassificationResult> results;
    for (int i = 0; i < actual_k; i++) {
        int idx = indices[i];
        std::string class_name = (idx < IMAGENET_CLASSES.size())
            ? IMAGENET_CLASSES[idx]
            : "class_" + std::to_string(idx);
        results.emplace_back(idx, probabilities[idx], class_name);
    }

    return results;
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    LOG_INFO("=== Ultralytics YOLO Classify Batch Demo (C++) ===");
    LOG_INFO("OpenCV Version: " << CV_VERSION);

    // ========================================================================
    // 0. Parse command line arguments
    // ========================================================================

    std::string model_path = MODEL_PATH;
    std::string input_folder = INPUT_FOLDER_PATH;
    std::string output_folder = OUTPUT_FOLDER_PATH;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) input_folder = argv[2];
    if (argc >= 4) output_folder = argv[3];

    // ========================================================================
    // 1. Load BPU model
    // ========================================================================

    LOG_INFO("Loading model: " << model_path);
    auto start_time = std::chrono::high_resolution_clock::now();

    hbPackedDNNHandle_t packed_dnn_handle;
    const char* model_file_name = model_path.c_str();
    CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "Failed to initialize model from file");

    auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    LOG_TIME("Load model time", load_duration);

    // ========================================================================
    // 2. Get model handle
    // ========================================================================

    const char** model_name_list;
    int model_count = 0;
    CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "Failed to get model name list");

    const char* model_name = model_name_list[0];
    LOG_INFO("Model name: " << model_name);

    hbDNNHandle_t dnn_handle;
    CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "Failed to get model handle");

    // ========================================================================
    // 3. Check model input
    // ========================================================================

    int32_t input_count = 0;
    CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "Failed to get input count");

    if (input_count != 1) {
        LOG_ERROR("Model should have exactly 1 input, but has " << input_count);
        return -1;
    }

    hbDNNTensorProperties input_properties;
    CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "Failed to get input tensor properties");

    // Check tensor type
    if (input_properties.tensorType != HB_DNN_IMG_TYPE_NV12) {
        LOG_ERROR("Input tensor type is not HB_DNN_IMG_TYPE_NV12");
        return -1;
    }
    LOG_INFO("Input tensor type: HB_DNN_IMG_TYPE_NV12");

    // Check tensor layout
    if (input_properties.tensorLayout != HB_DNN_LAYOUT_NCHW) {
        LOG_ERROR("Input tensor layout is not HB_DNN_LAYOUT_NCHW");
        return -1;
    }
    LOG_INFO("Input tensor layout: HB_DNN_LAYOUT_NCHW");

    // Get input shape
    if (input_properties.validShape.numDimensions != 4) {
        LOG_ERROR("Input tensor should have 4 dimensions");
        return -1;
    }

    int32_t input_h = input_properties.validShape.dimensionSize[2];
    int32_t input_w = input_properties.validShape.dimensionSize[3];
    LOG_INFO("Input shape: (1, 3, " << input_h << ", " << input_w << ")");

    // ========================================================================
    // 4. Check model outputs
    // ========================================================================

    int32_t output_count = 0;
    CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "Failed to get output count");

    if (output_count != 1) {
        LOG_ERROR("Classification model should have exactly 1 output, but has " << output_count);
        return -1;
    }

    hbDNNTensorProperties output_properties;
    CHECK_SUCCESS(
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, 0),
        "Failed to get output tensor properties");

    LOG_INFO("Output shape: ("
             << output_properties.validShape.dimensionSize[0] << ", "
             << output_properties.validShape.dimensionSize[1] << ", "
             << output_properties.validShape.dimensionSize[2] << ", "
             << output_properties.validShape.dimensionSize[3] << ")");

    int num_classes = output_properties.validShape.dimensionSize[1];
    LOG_INFO("Number of classes: " << num_classes);

    // Print quantization information
    if (output_properties.quantiType == SHIFT)
        LOG_INFO("Output quantiType: SHIFT");
    else if (output_properties.quantiType == SCALE)
        LOG_INFO("Output quantiType: SCALE");
    else if (output_properties.quantiType == NONE)
        LOG_INFO("Output quantiType: NONE");

    // ========================================================================
    // 5. Allocate system memory (reused in batch loop)
    // ========================================================================

    hbDNNTensor input;
    input.properties = input_properties;
    int input_memSize = input_h * input_w * 3 / 2;
    hbSysAllocCachedMem(&input.sysMem[0], input_memSize);

    hbDNNTensor* output = new hbDNNTensor[output_count];
    for (int i = 0; i < output_count; i++) {
        hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
        int out_size = output[i].properties.alignedByteSize;
        hbSysAllocCachedMem(&output[i].sysMem[0], out_size);
    }

    // ========================================================================
    // 6. Batch Process Images in Folder
    // ========================================================================

    DIR *dir = opendir(input_folder.c_str());
    if (dir == nullptr) {
        LOG_ERROR("Failed to open input directory: " << input_folder);
        return -1;
    }

    struct dirent *entry;
    int image_idx = 0;

    while ((entry = readdir(dir)) != nullptr) {
        std::string fileName = entry->d_name;
        std::string fullPath = input_folder + "/" + fileName;

        if ((fileName.size() >= 4 && strcmp(fileName.c_str() + fileName.size() - 4, ".jpg") == 0) ||
            (fileName.size() >= 5 && strcmp(fileName.c_str() + fileName.size() - 5, ".jpeg") == 0) ||
            (fileName.size() >= 4 && strcmp(fileName.c_str() + fileName.size() - 4, ".png") == 0)) {

            image_idx++;
            std::string baseName = extractFileNameWithoutExtension(fullPath);

            LOG_INFO("---------------------------------------------------------");
            LOG_INFO("[" << image_idx << "] Processing: " << fileName);

            cv::Mat img = cv::imread(fullPath);
            if (img.empty()) {
                LOG_ERROR("Failed to load image: " << fullPath);
                continue;
            }
            LOG_INFO("Image size: " << img.cols << "x" << img.rows);

            // Preprocess image
            float x_scale, y_scale;
            int x_shift, y_shift;
            cv::Mat preprocessed = preprocess_image(img, input_h, input_w,
                                                   x_scale, y_scale, x_shift, y_shift);

            // Convert to NV12
            cv::Mat nv12_img = bgr2nv12(preprocessed);

            // Prepare input tensor (reuse memory)
            memcpy(input.sysMem[0].virAddr, nv12_img.ptr<uint8_t>(), input_memSize);
            hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

            // Run inference
            hbDNNTaskHandle_t task_handle = nullptr;
            hbDNNInferCtrlParam infer_ctrl_param;
            HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

            start_time = std::chrono::high_resolution_clock::now();
            hbDNNInfer(&task_handle, &output, &input, dnn_handle, &infer_ctrl_param);
            hbDNNWaitTaskDone(task_handle, 0);

            auto infer_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
            LOG_TIME("BPU inference time", infer_duration);

            // Release task handle
            hbDNNReleaseTask(task_handle);

            // ====================================================================
            // 7. Post-process
            // ====================================================================

            LOG_INFO("Post-processing...");
            start_time = std::chrono::high_resolution_clock::now();

            // Flush memory
            hbSysFlushMem(&output[0].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);

            // Get output data
            std::vector<float> logits(num_classes);

#if REMOVE_DEQUANT_NODE == 1
            // 模型去除了反量化节点，输出为量化数据 (int8/int32)，
            // 需要手动反量化：读取原始量化值 * scale
            {
                float output_scale = 1.0f;
                if (output[0].properties.quantiType == SCALE) {
                    output_scale = output[0].properties.scale.scaleData[0];
                } else if (output[0].properties.quantiType == SHIFT) {
                    output_scale = 1.0f / (1 << output[0].properties.shift.shiftData[0]);
                }

                int output_type = output[0].properties.tensorType;
                void* output_base = output[0].sysMem[0].virAddr;

                // 使用 aligned shape 计算内存偏移（处理硬件内存对齐）
                int aligned_h = output[0].properties.alignedShape.dimensionSize[2];
                int aligned_w = output[0].properties.alignedShape.dimensionSize[3];

                for (int c = 0; c < num_classes; c++) {
                    int offset = c * aligned_h * aligned_w;
                    float val = 0.0f;
                    if (output_type == HB_DNN_TENSOR_TYPE_S32) {
                        val = static_cast<float>(((int32_t*)output_base)[offset]) * output_scale;
                    } else if (output_type == HB_DNN_TENSOR_TYPE_S8) {
                        val = static_cast<float>(((int8_t*)output_base)[offset]) * output_scale;
                    } else {
                        // 回退到 float 读取
                        val = ((float*)output_base)[offset];
                    }
                    logits[c] = val;
                }
            }
#else
            // 模型未去除反量化节点，输出为 float 类型，直接读取
            {
                float* output_data = reinterpret_cast<float*>(output[0].sysMem[0].virAddr);
                for (int i = 0; i < num_classes; i++) {
                    logits[i] = output_data[i];
                }
            }
#endif

            // Apply softmax
            std::vector<float> probabilities = softmax(logits);

            // Get top K results
            std::vector<ClassificationResult> results = get_topk_results(probabilities, TOP_K);

            auto post_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
            LOG_TIME("Post-processing time", post_duration);

            // ====================================================================
            // 8. Display and save results
            // ====================================================================

            LOG_INFO("Classification results:");
            LOG_INFO("Image: " << fullPath);
            std::cout << std::endl;

            for (size_t i = 0; i < results.size(); i++) {
                const auto& res = results[i];
                std::cout << "\033[1;32m"
                          << "TOP" << (i + 1) << " -> "
                          << "id: " << res.class_id << ", "
                          << "score: " << std::fixed << std::setprecision(3) << res.probability << ", "
                          << "name: " << res.class_name
                          << "\033[0m" << std::endl;
            }

            // Save result image with classification text drawn on it
            cv::Mat result_img = img.clone();
            for (size_t i = 0; i < results.size(); i++) {
                const auto& res = results[i];
                std::string text = "TOP" + std::to_string(i + 1) + ": [" +
                                   std::to_string(res.class_id) + "] " +
                                   res.class_name + " (" +
                                   std::to_string(static_cast<int>(res.probability * 100)) + "%)";
                cv::putText(result_img, text, cv::Point(10, 30 + static_cast<int>(i) * 30),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }
            std::string result_img_path = output_folder + "/" + baseName + "_out.jpg";
            cv::imwrite(result_img_path, result_img);
            LOG_INFO("Result image saved to: " << result_img_path);

            // Save result to text file in output folder
            std::string result_txt_path = output_folder + "/" + baseName + "_out.txt";
            std::ofstream result_file(result_txt_path);
            if (result_file.is_open()) {
                result_file << "Image: " << fileName << std::endl;
                result_file << "Classification results (Top " << results.size() << "):" << std::endl;
                for (size_t i = 0; i < results.size(); i++) {
                    const auto& res = results[i];
                    result_file << "TOP" << (i + 1) << " -> "
                               << "id: " << res.class_id << ", "
                               << "score: " << std::fixed << std::setprecision(3) << res.probability << ", "
                               << "name: " << res.class_name << std::endl;
                }
                result_file.close();
                LOG_INFO("Result txt saved to: " << result_txt_path);
            }

            std::cout << std::endl;
        }
    }

    closedir(dir);

    // ========================================================================
    // 9. Cleanup
    // ========================================================================

    for (int i = 0; i < output_count; i++) {
        hbSysFreeMem(&output[i].sysMem[0]);
    }
    delete[] output;
    hbSysFreeMem(&input.sysMem[0]);
    hbDNNRelease(packed_dnn_handle);

    LOG_INFO("=========================================================");
    LOG_INFO("Batch processing complete. Processed " << image_idx << " images.");
    LOG_INFO("=========================================================");

    return 0;
}
