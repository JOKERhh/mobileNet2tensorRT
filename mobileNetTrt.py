import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import sys
import os
import common
import random
import torch
from PIL import Image
class ModelData(object):
    MODEL_PATH = "ResNet50.onnx"
    INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
def build_engine(model_file,engine_path):
    builder=trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = common.GiB(1)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, 'rb') as model:
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))
            return None
    engine= builder.build_engine(network, config)
    with open(engine_path,"wb") as f:
        f.write(engine.serialize())
    return engine
def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return (image_arr / 255.0 - 0.45) / 0.225

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    # np.copyto(pagelocked_buffer, normalize_image(test_image))
    return test_image

if __name__ =="__main__":
    _, data_files = common.find_sample_data(description="Runs a ResNet50 network with a TensorRT inference engine.", subfolder="resnet50", find_files=["binoculars.jpeg", "reflex_camera.jpeg", "tabby_tiger_cat.jpg", ModelData.MODEL_PATH, "class_labels.txt"])
    # Get test images, models and labels.
    test_images = data_files[0:3]
    onnx_path = 'weights/mobileNet.onnx'
    engine_path = 'weights/mobileNet.engine'
    engine=build_engine(onnx_path,engine_path)
    context = engine.create_execution_context()
    test_image = random.choice(test_images)
    engine=build_engine(onnx_path,engine_path)
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    test_case = load_normalized_test_case(test_image, inputs[0].host)
    trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    print(len(trt_outputs[0]))