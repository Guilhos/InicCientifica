

import onnx
onnx_model = onnx.load("NNMPC/libs/modelo.onnx")
print(onnx.helper.printable_graph(onnx_model.graph))

