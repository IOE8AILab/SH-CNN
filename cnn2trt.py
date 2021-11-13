import tensorrt as trt
import time
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt

import numpy as np
import random

# For our custom calibrator
import torch

from calibrator import load_data, CNNEntropyCalibrator

# For ../common.py
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
import common

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# This function builds an engine from a Caffe model.
def build_int8_engine(calib, onnx_file_path, engine_file_path=""):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, \
            trt.Runtime(TRT_LOGGER) as runtime:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = 1
        config.max_workspace_size = common.GiB(1)
        # config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib
        # Parse Onnx model
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        network.get_input(0).shape = [1, 1, 240, 240]
        # network.get_input(0).shape = [1, 1, 28, 28]
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        return engine

def compute_rmse(out_pre, test_label):
    num = np.shape(out_pre)[0]
    mse = np.zeros(num)
    for i in range(num):
        mse[i] = np.sum((out_pre[i] - test_label[i])**2)/299
    meanmse = np.sum(mse)/num
    meanrmse = np.sum(np.sqrt(mse))/num
    print('mse:', meanmse, '              rmse:', meanrmse)

def plot_one(out_pre, test_label):
    x = np.arange(3, 300)
    plt.plot(x, out_pre[1, 2:], 'g.', x, test_label[1, 2:], 'rx')
    plt.show()

def main():
    onnx_file_path = 'CNN.onnx'
    engine_file_path = "CNN.trt"
    test_num = 10000
    test_set, test_label = load_data(test_num)
    # Now we create a calibrator and give it the location of our calibration data.
    # We also allow it to cache calibration data for faster engine building.
    calibration_cache = "cnn_calibration.cache"
    cache_num = 500
    calib = CNNEntropyCalibrator(cache_file=calibration_cache, cache_num=cache_num)

    out_pre = np.zeros((test_num, 299))
    total_time = 0
    with build_int8_engine(calib, onnx_file_path, engine_file_path) as engine, \
            engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        for i in range(test_num):
            inputs[0].host = test_set[i:i+1].astype(np.float32)
            st = time.time()
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            out_pre[i:i+1] = trt_outputs
            et = time.time()
            total_time += (et - st)
        print('mean time:' + str(total_time*1000/test_num) + 'ms')
        compute_rmse(out_pre, test_label)
        print(out_pre[0])


if __name__ == '__main__':
    main()
