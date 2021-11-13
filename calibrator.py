import tensorrt as trt
import os
import h5py
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def load_data(test_num):
    f = h5py.File("Sum_NewData_299_100.h5", 'r')
    temp = 90000 + test_num
    X = f['Xtrain'][90000:temp]
    Y = f['Ytrain'][90000:temp]
    data = np.expand_dims(X, axis=1)
    return np.ascontiguousarray(data), Y   # test_num,1,240,240


class CNNEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, cache_num, batch_size=1):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        self.data, _ = load_data(cache_num)
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]


    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

