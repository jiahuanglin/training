import onnx

# Load the ONNX model
model = onnx.load("models/deepspeech_9.onnx")

# Check that the IR is well formed
print(onnx.checker.check_model(model))

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
