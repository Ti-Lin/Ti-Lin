# Ti-Lin

Codes for reproducing the experimental results in "Certifying Robustness of Convolutional Neural Networks with Tightest Linear Approximation".

This repository contains our tools Ti-Lin and Ti-Lin-R. Concretely, for general CNNs/FNNs  we compare  Ti-Lin's linear bounds of Sigmoid/Tanh/Arctan function with VeriNet and DeepCert, and compared Ti-Lin's linear bounds of Maxpool function with CNN-Cert. For non-negative CNNs, we compared Ti-Lin-R with VeriNet. The experiment results show that Ti-Lin gains tighter non-trivial certified robustness bounds with almost the same computational complexity.

## Setup

1. The code is tested with python3 and TensorFlow v1.10 or v1.12 (TensorFlow v1.8 is not compatible). The following packages are required.
   
   ```
   conda create --name tilin python=3.6
   conda activate tilin
   conda install pillow numpy scipy pandas h5py tensorflow=1.12.0 numba posix_ipc matplotlib
   ```

2. Clone this repository
   
   ```
   git clone https://github.com/xiaobailearn/Ti-Lin.git
   cd Ti-Lin
   ```

3. Download the [models](https://www.dropbox.com/s/byvlywiqnocx51d/models.zip?dl=0) in this paper and unzip the zip file.
   
   ```
   unzip models.zip
   ```

4. Reproduce the experiments result in our paper.
- In DeepCert, VeriNet, and Ti-Lin, you can get certified robustness bounds and runtime on ```pure CNNs``` with Sigmoid/Tanh/Arctan functions in two steps. Firstly, you need to set ```table``` in the main function of pymain_new_network.py to ```1```. Secondly, you need to run pymain_new_network.py. The result are saved into log_pymain_new_network_{timestamp}.txt. For example,
  
  ```
  cd Ti-Lin
  python pymain_new_network.py 
  ```

- In DeepCert, VeriNet, and Ti-Lin, you can get certified robustness bounds and runtime on ```FNNs``` with Sigmoid/Tanh/Arctan functions by setting ```table``` in the main function of pymain_fnn.py to ```2``` and running pymain_fnn.py. For example,
  
  ```
  cd Ti-Lin
  python pymain_fnn.py 
  ```

- In Cnncert and Ti-Lin, you can get certified robustness bounds and runtime on ```CNNs with max pooling with ReLU``` by setting ```table``` in the main function of pymain_new_network.py to ```3``` and running pymain_fnn.py. For example,
  
  ```
  cd Cnncert
  python pymain_new_network.py 
  ```

- In VeriNet and Ti-Lin-R, you can get certified robustness bounds and runtime on ```non-negative``` pure CNNs with Sigmoid/Tanh/Arctan functions by setting ```table``` in the main function of pymain_new_network_postive.py to ```4``` and running pymain_new_network_postive.py.
  
  ```
  cd VeriNet
  python pymain_new_network_postive.py 
  ```
  
  Results of certified robusntess bounds and runtime are saved into log_pymain_new_network_{timestamp}.txt

### References

Boopathy, A.; Weng, T.-W.; Chen, P.-Y.; Liu, S.; and Daniel,L. 2019. Cnn-cert: An efficient framework for certifying robustness of convolutional neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, 3240–3247.
