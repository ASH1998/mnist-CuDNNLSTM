# mnist-CuDNNLSTM


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
