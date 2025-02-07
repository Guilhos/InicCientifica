import numpy as np
import onnx
import do_mpc
import casadi as ca


model = onnx.load("NNMPC/libs/modelo.onnx")
casadi_converter = do_mpc.sysid.ONNXConversion(model)

x = ca.SX.sym('x',3,4)
casadi_converter.convert(input=x)

print(casadi_converter)