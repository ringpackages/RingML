# File: src/ringml.ring
# Description: Main entry point

load "consolecolors.ring"
load "fastpro.ring"
load "core/tensor.ring"
load "utils/serializer.ring"
load "data/dataset.ring"
load "data/datasplitter.ring"
load "utils/visualizer.ring"
load "layers/layer.ring"
load "layers/dense.ring"
load "layers/activation.ring"
load "layers/softmax.ring"
load "layers/dropout.ring" 
load "model/sequential.ring"
load "loss/mse.ring"
load "loss/crossentropy.ring"
load "optim/sgd.ring"
load "optim/adam.ring"         

func RingML_Info
    see "RingML Library v1.4 - Adam Optimizer Added" + nl