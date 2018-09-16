# mnist-CuDNNLSTM

This code comapres accuracy and time results for vanills LSTM and CuDNNLSTM.

## CuDNNLSTM

The CuDNN-LSTM layer is defined within CuDNN_rnn layer of tensorflow which is specifically compiled to work with CuDNN package.
Its almost 10 times faster than regular LSTM.

Cudnn RNNs have two major differences from other platform-independent RNNs tf
  provides:
  * Cudnn LSTM and GRU are mathematically different from their tf counterparts.
    (e.g. @{tf.contrib.rnn.LSTMBlockCell} and @{tf.nn.rnn_cell.GRUCell}.
  * Cudnn-trained checkpoints are not directly compatible with tf RNNs:
    * They use a single opaque parameter buffer for the entire (possibly)
      multi-layer multi-directional RNN; Whereas tf RNN weights are per-cell and
      layer.
    * The size and layout of the parameter buffers may change between
      CUDA/CuDNN/GPU generations. Because of that, the opaque parameter variable
      does not have a static shape and is not partitionable. Instead of using
      partitioning to alleviate the PS's traffic load, try building a
      multi-tower model and do gradient aggregation locally within the host
      before updating the PS. See [tensorflow parameter server variables](https://www.tensorflow.org/performance/performance_models#parameter_server_variables)
      for a detailed performance guide.

## Accuracy(acc, val_acc)
|              | Tesla K80        | Pascal P100      |
|--------------|------------------|------------------|
| vanilla LSTM | (0.9693, 0.9714) | (0.9710, 0.9788) |
| CuDNNLSTM    | (0.9776, 0.9803) | (0.9878, 0.985)  |

## Time Taken(in Secs)
|              | Tesla K80 | Pascal P100 |
|--------------|-----------|-------------|
| vanilla LSTM | 984.2     | 496.82      |
| CuDNNLSTM    | 130.84    | 39.82       |


## Classes found in `tf.contrib.cudnn_rnn`
class CudnnCompatibleGRUCell: Cudnn Compatible GRUCell.     
class CudnnCompatibleLSTMCell: Cudnn Compatible LSTMCell.       
class CudnnGRU: Cudnn implementation of the GRU layer.    
class CudnnGRUSaveable: SaveableObject implementation handling Cudnn GRU opaque params.     
class CudnnLSTM: Cudnn implementation of LSTM layer.        
class CudnnLSTMSaveable: SaveableObject implementation handling Cudnn LSTM opaque params.         
class CudnnRNNRelu: Cudnn implementation of the RNN-relu layer.         
class CudnnRNNReluSaveable: SaveableObject implementation handling Cudnn RNN Relu opaque params.          
class CudnnRNNTanh: Cudnn implementation of the RNN-tanh layer.         
class CudnnRNNTanhSaveable: SaveableObject implementation handling Cudnn RNN Tanh opaque params.            

## Reference :
This code is taken from different open-source platforms and [tensorflow.org](www.tensorflow.org)
