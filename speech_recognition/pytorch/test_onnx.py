import onnx
import caffe2.python.onnx.backend as backend
import numpy as np

def check_model():
    # Load the ONNX model
    model = onnx.load("models/deepspeech_9.onnx")

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)


def test_inference():
    model = onnx.load("models/deepspeech_9.onnx")

    rep = backend.prepare(model, device="CUDA:0") # or "CPU"
    # For the Caffe2 backend:
    #     rep.predict_net is the Caffe2 protobuf for the network
    #     rep.workspace is the Caffe2 workspace for the network
    #       (see the class caffe2.python.onnx.backend.Workspace)
    outputs = rep.run(np.random.randn(16, 1, 224, 224).astype(np.float32))
    # To run networks with more than one input, pass a tuple
    # rather than a single numpy ndarray.
    print(outputs[0])


if __name__ == "__main__":
    test_inference()
