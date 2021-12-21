# mobileNet2tensorRT
convert Pytorch mobileNet to tensorRt
if you want to use c++ code,please follow:
- cd build
- cmake ..
- make
then you can run compiled program which can generate .onnx
Then you could run mobileNetTrt.py which can use tensorRt inference.
