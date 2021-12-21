SET(common_dir /usr/src/tensorrt/samples/common)
INCLUDE_DIRECTORIES(${common_dir})

SET(cuda_dir /usr/local/cuda-10.2/include)
INCLUDE_DIRECTORIES(${cuda_dir})

set(ONNX_PARSE /home/case/Desktop/wza/onnx-tensorrt)
INCLUDE_DIRECTORIES(${ONNX_PARSE})

SET(LOG_CPP /usr/src/tensorrt/samples/common/logger.cpp)

ADD_EXECUTABLE(trt_test main.cpp ${LOG_CPP})

find_library(LIBONNX_PATH nvonnxparser /home/case/Desktop/wza/onnx-tensorrt/build)
TARGET_LINK_LIBRARIES(trt_test ${LIBONNX_PATH})

find_library(LIBNVINFER nvinfer /usr/lib/aarch64-linux-gnu/)
TARGET_LINK_LIBRARIES(trt_test ${LIBNVINFER})

find_library(LIBCUDART cudart /usr/local/cuda-10.2/lib64/)
TARGET_LINK_LIBRARIES(trt_test ${LIBCUDART})