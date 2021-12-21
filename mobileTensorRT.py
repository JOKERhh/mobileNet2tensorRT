import onnxruntime as ort
import time
from PIL import Image
import numpy as np
from torchvision import transforms as T
from log import timer, logger
import torch
import torchvision

dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
model_alex = torchvision.models.alexnet(pretrained=True).cuda()

trans = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
img = Image.open('mobileNet/test_photo.jpg')
img_tensor = trans(img).unsqueeze(0)
img_np = np.array(img_tensor)
logger('Image : {} >>> {}'.format(np.shape(img) , np.shape(img_tensor)))
load_onnx = timer('Load ONNX Model')
ort_session = ort.InferenceSession('mobileNet/mobileNet.onnx')
load_onnx.end()

infer_alex=timer('Run alex')
model_alex.eval()
outputs=model_alex(img_tensor.cuda())
infer_alex.end()

input_name = ort_session.get_inputs()[0].name
infer_onnx = timer('Run MobileNet')
outputs = ort_session.run(None, {input_name: img_np})[0]
infer_onnx.end()
