#include <iostream>
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "BatchStream.h"
#include "EntropyCalibrator.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"

using namespace nvinfer1;
using namespace nvonnxparser;

// class Logger : public ILogger           
//  {
//      void log(Severity severity, const char* msg) override
//      {
//          // suppress info-level messages
//          if (severity != Severity::kINFO)
//              std::cout << msg << std::endl;
//      }
//  } gLogger;


int main() {
        std::cout<<"hello world"<<std::endl;
        samplesCommon::Args args;

        // 1 加载onnx模型
        IBuilder* builder = createInferBuilder(sample::gLogger.getTRTLogger());
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());

        const char* onnx_filename="/home/case/Desktop/wza/mobileNetRT/weights/mobileNet.onnx";
        parser->parseFromFile(onnx_filename, static_cast<int>(sample::gLogger.getReportableSeverity()));
        for (int i = 0; i < parser->getNbErrors(); ++i)
        {
                std::cout << parser->getError(i)->desc() << std::endl;
        }
        std::cout << "successfully load the onnx model" << std::endl;

        // 2、build the engine
        unsigned int maxBatchSize=1;
        builder->setMaxBatchSize(maxBatchSize);
        IBuilderConfig* config = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1 << 20);
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

        // 3、serialize Model
        IHostMemory *gieModelStream = engine->serialize();
        std::string serialize_str;
        std::ofstream serialize_output_stream;
        serialize_str.resize(gieModelStream->size());   
        memcpy((void*)serialize_str.data(),gieModelStream->data(),gieModelStream->size());
        serialize_output_stream.open("./serialize_engine_output.trt");
        serialize_output_stream<<serialize_str;
        serialize_output_stream.close();


        // 4、deserialize model
        IRuntime* runtime = createInferRuntime(sample::gLogger.getTRTLogger());
        std::string cached_path = "/home/case/Desktop/wza/mobileNetRT/build/serialize_engine_output.trt";
        std::ifstream fin(cached_path);
        std::string cached_engine = "";
        while (fin.peek() != EOF){ 
                std::stringstream buffer;
                buffer << fin.rdbuf();
                cached_engine.append(buffer.str());
        }
        fin.close();
        ICudaEngine* re_engine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);

        std::cout << "Hello, World!" << std::endl;
        return 0;
}