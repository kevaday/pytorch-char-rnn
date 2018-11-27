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

## Example
Here is an example I sampled after training on Mysterious Island by Jules Vernes with the following command:

```python train.py ..\text_data\mysterious_island_vernes.txt --rnntype GRU --rnnsize 786 --nlayers 4 --batchsize 100 --seqlength 136 --nepochs 25 --print_every 75 --sample_every 2 --lrdiv 22 --dropout 0.2 --cuda --fastest```

> The engineer was not half-past
so. He had gained more caresult, in a dark step by the light of the
sand. There was no reason for staring anything. Put off in his
delight, the necessity of reporting among the works to go.

Here is another better example after training it with:

```python train.py ..\text_data\mysterious_island_vernes.txt --rnntype LSTM --rnnsize 1500 --nlayers 4 --batchsize 64 --seqlength 128 --nepochs 30 --print_every 75 --sample_every 2 --lrdiv 22 --dropout 0.1 --cuda --fastest```

> “I beg your pardon!” returned Pencroft.
> “Why?”
> “Because that pleases me!”
> “Are you very fond of pig then, Pencroft?”
> “I am very fond of pig,” replied the sailor, dragging after him
> the body of the animal.

> While Neb skinned the jaguar, his companions had reached the
> marsh of the first magnitude.

If you've read Mysterious Island, you would see that some lines from the above sample are memorized. This is due to the low dropout. Although it memorized some things above, it also added in some twists if its own. For example, the line '“I am very fond of pig,” replied'... is memorized, but it put together 2 different lines, and actually did it in the right context.

[Here's](https://drive.google.com/open?id=1h2q9rEHLE_g_z_rvCq2SCs1Pif0eIo-Q) the link for the trained model on Mysterious Island
