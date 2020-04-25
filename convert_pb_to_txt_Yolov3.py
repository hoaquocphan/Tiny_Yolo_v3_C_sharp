import numpy as np
import onnx
import os
import glob
import onnx as backend

from onnx import numpy_helper

test_data_dir = 'test_data_set_0'



inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))

# Load input_0
input_0 = []
input_file_0 = os.path.join(test_data_dir, 'input_0.pb')
output_file_0 = os.path.join(test_data_dir, 'input_0.txt')

tensor = onnx.TensorProto()
with open(input_file_0, 'rb') as f:
    tensor.ParseFromString(f.read())
input_0.append(numpy_helper.to_array(tensor))

in_data_0 = np.array(input_0)
print('shape: {}'.format(in_data_0.shape))

f0= open(output_file_0,"w+")
for a in range(1):
    for b in range(1):
        for c in range(3):
            for d in range(416):
                for e in range(416):
                    f0.write('{}\n'.format(in_data_0[a][b][c][d][e]))
f0.close()

# Load input_1
input_1 = []
input_file_1 = os.path.join(test_data_dir, 'input_1.pb')
output_file_1 = os.path.join(test_data_dir, 'input_1.txt')

tensor = onnx.TensorProto()
with open(input_file_1, 'rb') as f:
    tensor.ParseFromString(f.read())
input_1.append(numpy_helper.to_array(tensor))

in_data_1 = np.array(input_1)
print('shape: {}'.format(in_data_1.shape))

f0= open(output_file_1,"w+")
for a in range(1):
    for b in range(1):
        for c in range(2):
            f0.write('{}\n'.format(in_data_1[a][b][c]))
f0.close()

# Load reference outputs
ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))

# Load output_0
input_0 = []
input_file_0 = os.path.join(test_data_dir, 'output_0.pb')
output_file_0 = os.path.join(test_data_dir, 'output_0.txt')

tensor = onnx.TensorProto()
with open(input_file_0, 'rb') as f:
    tensor.ParseFromString(f.read())
input_0.append(numpy_helper.to_array(tensor))

in_data_0 = np.array(input_0)
print('shape: {}'.format(in_data_0.shape))

f0= open(output_file_0,"w+")
for a in range(1):
    for b in range(1):
        for c in range(10647):
            for d in range(4):
                f0.write('{}\n'.format(in_data_0[a][b][c][d]))
f0.close()

# Load output_1
input_1 = []
input_file_1 = os.path.join(test_data_dir, 'output_1.pb')
output_file_1 = os.path.join(test_data_dir, 'output_1.txt')

tensor = onnx.TensorProto()
with open(input_file_1, 'rb') as f:
    tensor.ParseFromString(f.read())
input_1.append(numpy_helper.to_array(tensor))

in_data_1 = np.array(input_1)
print('shape: {}'.format(in_data_1.shape))

f0= open(output_file_1,"w+")
for a in range(1):
    for b in range(1):
        for c in range(80):
            for d in range(10647):
                f0.write('{}\n'.format(in_data_1[a][b][c][d]))
f0.close()

# Load output_2
input_2 = []
input_file_2 = os.path.join(test_data_dir, 'output_2.pb')
output_file_2 = os.path.join(test_data_dir, 'output_2.txt')

tensor = onnx.TensorProto()
with open(input_file_2, 'rb') as f:
    tensor.ParseFromString(f.read())
input_2.append(numpy_helper.to_array(tensor))

in_data_2 = np.array(input_2)
print('shape: {}'.format(in_data_2.shape))

f0= open(output_file_2,"w+")
for a in range(1):
    for b in range(1):
        for c in range(3):
            f0.write('{}\n'.format(in_data_2[a][b][c]))
f0.close()