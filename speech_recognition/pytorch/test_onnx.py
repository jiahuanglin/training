import onnx
import caffe2.python.onnx.backend as backend
# import onnx_caffe2.backend as backend
import numpy as np
import time

def check_model():
    # Load the ONNX model
    model = onnx.load("models/deepspeech_9.onnx")

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)


def test_inference():
    model = onnx.load("models/deepspeech_9.onnx")
    print("checking onnx model!")
    onnx.checker.check_model(model)
    print("model checked, prepareing backend!")
    rep = backend.prepare(model, device="CPU") # or "CPU"
    # For the Caffe2 backend:
    #     rep.predict_net is the Caffe2 protobuf for the network
    #     rep.workspace is the Caffe2 workspace for the network
    #       (see the class caffe2.python.onnx.backend.Workspace)
    print("runing inference!")
    input = np.random.randn(16, 1, 161, 129).astype(np.float32)
    # input = np.random.randn(16, 1, 124, 124).astype(np.float32)
    # W = {model.graph.input[0].name: input}
    start = time.time()
    outputs = rep.run(input)
    print("time used: {}".format(time.time() - start))
    # To run networks with more than one input, pass a tuple
    # rather than a single numpy ndarray.
    print(outputs[0])


if __name__ == "__main__":
    test_inference()

