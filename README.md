# pytorch-char-rnn
**A text generating neural network implemented in PyTorch**

The program trains a recurrent neural network (RNN) on a text corpus. The text is fed into the network in sequences, and is encoded using One-hot encoding. 
The supported RNN types are LSTM, GRU, RNN_TANH, and RNN_RELU. All of them have support for bidirection.
Bidirectional RNNs perform very well because it gives the network the ability to think ahead.

This program is very flexible and there are lots of available arguments.


## Requirements
```
numpy
torch
```
You can install these requirements with:
```
pip install -r requirements.txt
```

## Training
To train the network, run ```python train.py``` with the desired arguments.

To view the options for training, run ```python train.py --help```

There are a lot of arguments, but here is a simple **example**:

```python train.py --infile <some text file to train on> --save_every 2 --rnntype GRU --bidirectional --rnnsize 1024 --batchsize 64 --print_every 100 --cuda```

## Sampling
If you don't know what sampling is, it's generating text from the model by giving it a starting sequence, and the model predicts the next N chars (N is defined by the argument ```--length```)

To sample from a saved model, run ```python sample.py <saved model>``` with the desired arguments.

To view the options for sampling, simply run ```python sample.py --help```

