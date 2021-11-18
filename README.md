# SH-CNN
This repository contains PyTorch implementation for the paper: "Deep Phase Retrieval for Astronomical Shack-Hartmann Wavefront Sensors". It is a high-speed deep learning based phase retrieval approach for Shack-Hartmann wavefront sensor used in astronomical adaptive optics.

### Setup:
Clone this project using:
```
git clone https://github.com/IOE8AILab/SH-CNN.git
```
The code is developed using Python 3.9, PyTorch 1.9.0 and TensorRT 8.0.1.6. The GPU we used is NVIDIA RTX 3090. 

### Dataset 
To generate dateset by matlab：
```
cd SH-CNN/TurbPhaseGeneration/
matlab -nodesktop -nosplash -r TurbPhaseGenerate
```

### Training
To preprocess dataset and train the entire framework:
```
cd SH-CNN/
python DataMake.py
python Train_Model.py
```

### Accelerating
To accelerate the model by tensorrt:
```
python cnn2trt.py
```

Please make sure your path is set properly for the dataset.
