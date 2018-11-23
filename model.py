''' by Kevi Aday 2018
'''

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from utils import get_batches, one_hot_encode, get_lookup_tables
from os.path import basename
from math import exp


class CharRNN(nn.Module):
    def __init__(self, text, rnn_type='LSTM', bidirectional=False, n_hidden=512, n_layers=4, dropout=0.3, lr=2e-3, initrange=1, cuda=False, cudnn_fastest=False, cudnn_benchmark=False):
        super().__init__()

        self.rnn_type = rnn_type.upper()
        self.bidirectional = bidirectional
        self.drop = dropout
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lr = lr
        self.initrange = initrange

        self.use_cuda = cuda
        self.fastest = cudnn_fastest
        self.benchmark = cudnn_benchmark
        if cudnn_fastest: torch.backends.cudnn.fastest = True
        if cudnn_benchmark: torch.backends.cudnn.benchmark = True

        self.text = text
        self.int2char, self.char2int = get_lookup_tables(text)
        self.chars = tuple(self.char2int.keys())

        self.dropout = nn.Dropout(dropout)
        if rnn_type in ('LSTM', 'GRU'):
            self.rnn = getattr(nn, rnn_type)(len(self.chars), n_hidden, n_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        else:
            try:
                nonlin = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError('An invalid option for `--rnntype` was supplied, valid options are `LSTM`, `GRU`, `RNN_TANH`, or `RNN_RELU`')

            self.rnn = nn.RNN(len(self.chars), n_hidden, n_layers, nonlinearity=nonlin, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(n_hidden*2 if bidirectional else n_hidden, len(self.chars))

        self.init_weights()

        if not cuda and torch.cuda.is_available():
            print('WARNING: CUDA argument was set to false. Your device supports CUDA, you should use it.')
        if cuda:
            self.cuda()
        else:
            self.cpu()

    def forward(self, x, hc):
        ''' Forward pass through the network '''
              
        if self.bidirectional:
            x, h = self.rnn(x, hc)
            x = self.dropout(x)
            x = self.decoder(x.view(x.size(0)*x.size(1), x.size(2)))
        else:
            x, h = self.rnn(x, hc)
            x = self.dropout(x)
            x = x.view(x.size(0)*x.size(1), self.n_hidden)
            x = self.decoder(x)

        return x, h

    def init_weights(self):
        ''' Initialize weights of decoder (fully connected layer) '''

        # Apply bias tensor to all zeros
        self.decoder.bias.data.fill_(0)

        # Apply random uniform weights to decoder
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)

    def init_hidden(self, batch_size):
        ''' Initialize hidden state of rnn

            arguments:
                batch_size: batch size

            returns:
                new weights torch Tensor
                if rnn type is an LSTM, returns a tuple of 2 of these weights
        '''
        
        # Create two new tensors with size of n_layers (x2 if bidirectional) x seq_len x n_hidden,
        # initialized to zero, for hidden state and cell state of RNN
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.n_layers*2 if self.bidirectional else self.n_layers, batch_size, self.n_hidden),
                    weight.new_zeros(self.n_layers*2 if self.bidirectional else self.n_layers, batch_size, self.n_hidden))
        else:
            return weight.new_zeros(self.n_layers*2 if self.bidirectional else self.n_layers, batch_size, self.n_hidden)

    def predict(self, char, h=None, cuda=False, top_k=None):
        ''' Predict the character after the given character

            arguments:
                char: starting character
                h: hidden state
                cuda: use cuda
                top_k: finds most probable next char from the output softmax

            returns:
                predicted character
                h: hidden state
        '''

        if h is None: # If hidden state is not supplied, use a hidden state of sequence length of 1
            h = self.init_hidden(1)

        x = np.array([[self.char2int[char]]])
        x = one_hot_encode(x, len(self.chars))
        with torch.no_grad():
            inputs = torch.from_numpy(x)
        if self.use_cuda:
            inputs = inputs.cuda()
        
        with torch.no_grad():
            h = detach_hidden(h)
        out, h = self.forward(inputs, h)

        p = F.softmax(out).data
        if self.use_cuda:
            p = p.cpu()

        if not top_k:
            top_ch = np.arange(len(self.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())

        return self.int2char[char], h

def save_model(filename, model):
    with open(filename, 'wb') as f:
        torch.save(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        model = torch.load(f)

    return model

def detach_hidden(h):
    ''' Detach hidden state from their training history.
    '''
    
    if isinstance(h, tuple):
        return tuple([detach_hidden(t) for t in h])
    else:
        return h.detach()

def train(model, data, textfile, epochs=25, batch_size=48, seq_len=100, lr=2e-3, lrdiv=[20], clip=5, val_fract=0.1, cuda=False, print_every=25, sample_every=5, save_every=1, save_dir='save'):
    ''' Train a language network

        arguments:
            net: network to train
            data: encoded text data
            textfile: basename of text file to train on, only being used for saving
            epochs: number of epochs to train
            batch_size: number of mini-sequences per mini-batch
            seq_len: number of characters per mini-batch
            lr: learning rate
            clip: gradient clipping
            val_fract: fraction of text data to put aside for validation
            cuda: Train the network with CUDA on GPU
            print_every: number of iterations to print training stats after
    '''
    
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # separate training and validation data
    val_idx = int(len(data)*(1-val_fract))
    data, val_data = data[:val_idx], data[val_idx:]

    if cuda:
        model.cuda()

    counter = 0
    total_loss = 0
    n_chars = len(model.chars)
    for e in range(epochs):
        h = model.init_hidden(batch_size)
        for x, y in get_batches(data, batch_size, seq_len):
            counter += 1

            # one-hot encode training data and make the Torch Tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # creat new Tensors for hidden state
            h = detach_hidden(h)

            model.zero_grad()

            output, h = model(inputs, h)
            loss = criterion(output, targets.view(batch_size*seq_len).long())
            total_loss += loss.item()
            
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            opt.step()

            if counter % print_every == 0:
                # validation section

                # get hidden state for validation
                val_h = model.init_hidden(batch_size)
                val_losses = []
                for x, y in get_batches(val_data, batch_size, seq_len):
                    # the validation is almost the same thing except we do not backprop,
                    # we only calculate some statistics from our model and validate it
                    # on the validation set
                    
                    # create Torch tensors from our data by one-hot encoding it
                    x = one_hot_encode(x, n_chars)
                    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                    # creat new Tensors for hidden state
                    with torch.no_grad():
                        val_h = detach_hidden(val_h)
                    if cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = model(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_len).long())

                    val_losses.append(val_loss.item())

                # print the stats
                print('Epoch: {}/{}\t'.format(e+1, epochs),
                      'Step: {}\t'.format(counter),
                      'Loss: {:.4f}\t'.format(loss.item()),
                      'Val loss: {:.4f}'.format(np.mean(val_losses)),
                      'Perplexity: {:.4f}'.format(exp(total_loss/print_every)),
                      'Val perplexity: {:.4f}'.format(exp(sum(val_losses)/print_every)))

        # divide learning rate by 10 if it is the specified epoch(s)
        for i in lrdiv:
            if (e+1) % i == 0:
                lr /= 10
                
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
                break

        if (e+1) % sample_every == 0:
            print(f'Epoch {e+1}: Sampling from model:')
            print('-'*50)
            print(sample(model, length=100, cuda=cuda))
            print('-'*50)
            model.train()

        if (e+1) % save_every == 0:
            # save the model

            save_file = f'{save_dir}/{basename(textfile)}_{loss}_{e+1}.ckpt'
            save_model(save_file, model)
            print(f'Network saved as {save_file}')

    return np.mean(val_losses)

def sample(model, length, prime='The ', top_k=None, cuda=False, file=None, encoding=None):
    ''' Sample characters from a model

        arguments:
            model: the model to sample from
            length: number of characters to sample from the model
            prime: the starting string to sample with, must not be empty string
            top_k: finds most probable next char from the output softmax
            cuda: use cuda for sampling
            file: file to save predicted text to
            encoding: encoding of text file to write to

        returns:
            predicted next chars
    '''

    if not prime or prime == '':
        raise ValueError('Prime string must not be void')

    elif not file and encoding:
        encoding = None
    
    if cuda:
        model.cuda()
    else:
        model.cpu()

    model.eval()
    chars = [ch for ch in prime]
    h = model.init_hidden(1)
    for ch in prime:
        char, h = model.predict(ch, h, cuda=cuda, top_k=top_k)

    chars.append(char)

    for i in range(length):
        char, h = model.predict(chars[-1], h, cuda=cuda, top_k=top_k)
        chars.append(char)
        if file:
            if i % 10 == 0:
                print(f'Predicted {len(chars)-len(prime)-1}/{length} chars\r', end='', flush=True)

    if file:
        if encoding:
            with open(file, 'w', encoding=encoding) as f:
                f.write(''.join(chars))
        else:
            with open(file, 'w') as f:
                f.write(''.join(chars))

    return ''.join(chars)
