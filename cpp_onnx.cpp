#include <vector>
#include <random>
#include <iostream>
#include <onnxruntime_cxx_api.h>


std::vector<float> run_infer_features(Ort::Session* features_onnx) {
    // Check for null pointer
    if (features_onnx == nullptr) {
        std::cerr << "Error: features_onnx session is null." << std::endl;
        return {}; // Return an empty vector
    }

    // Add random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<std::vector<float>> all_features;
    std::cout<<"Running features onnx"<<std::endl;
    // Process each frame
    for(int i = 0; i < 5; i++) {
        // Prepare input tensor
        std::vector<float> frame_input(1 * 3 * 224 * 224);
        // initialize frame_input with random values
        for(auto& val : frame_input) val = dis(gen);
        std::cout<<"frame_input data sample 10,";
        for (int j = 0; j < 10; j++) {
            std::cout<<frame_input[j]<<",";
        }
        std::cout<<std::endl;
        // Setup input tensor
        std::vector<int64_t> input_shape = {1, 3, 224, 224};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, frame_input.data(), frame_input.size(), 
            input_shape.data(), input_shape.size());
        std::cout<<"Running features prepare"<<std::endl;

        // Run inference
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};
        Ort::RunOptions run_options;
        auto features = features_onnx->Run(
            run_options, 
            input_names, &input_tensor, 1, 
            output_names, 1);
        std::cout<<"Running after infer"<<std::endl;

        // Store features
        float* features_data = features[0].GetTensorMutableData<float>();
        std::vector<float> features_vec(features_data, 
            features_data + features[0].GetTensorTypeAndShapeInfo().GetElementCount());
        all_features.push_back(features_vec);
    }

    // Concatenate features
    std::vector<float> concatenated_features;
    for(const auto& feat : all_features) {
        concatenated_features.insert(concatenated_features.end(), feat.begin(), feat.end());
    }
    return concatenated_features;
}

void run_infer_classifier(Ort::Session* classifier_onnx, std::vector<float> concatenated_features) {
    // Run classifier
    std::vector<int64_t> classifier_input_shape = {1, 1280*5, 7, 7};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value classifier_input = Ort::Value::CreateTensor<float>(
        memory_info, concatenated_features.data(), concatenated_features.size(),
        classifier_input_shape.data(), classifier_input_shape.size());

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    auto outputs = classifier_onnx->Run(
        Ort::RunOptions{nullptr},
        input_names, &classifier_input, 1,
        output_names, 1);

    // Process results
    float* output_data = outputs[0].GetTensorMutableData<float>();
    size_t output_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
}

void predict_onnx(Ort::Session* features_onnx, Ort::Session* classifier_onnx) {
    // Random number generation
    std::vector<float> concatenated_features = run_infer_features(features_onnx);
    
    run_infer_classifier(classifier_onnx, concatenated_features);
    

}

int main() {
    Ort::SessionOptions options;
    Ort::Session features_onnx(nullptr, "features_224_224.onnx", options);
    Ort::Session classifier_onnx(nullptr, "classifier_224_224.onnx", options);
    auto shape = features_onnx.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cout<<"model features_onnx input shape,"<< shape[0] << "," << shape[1] << "," << shape[2] << "," << shape[3] <<std::endl;
    auto shape2 = classifier_onnx.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cout<<"model classifier_onnx input shape,"<< shape2[0] << "," << shape2[1] << "," << shape2[2] << "," << shape2[3] <<std::endl;
    predict_onnx(&features_onnx, &classifier_onnx);
    return 0;
}

