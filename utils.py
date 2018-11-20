''' by Kevi Aday 2018
'''

import numpy as np


def get_lookup_tables(text):
    ''' Helper function for creating int to char and char to int encodings

        arguments:
            text: text to base lookup tables on
        returns:
            int2char: int to char lookup table
            char2int: char to int lookup table
    '''
    int2char = dict(enumerate(tuple(set(text))))
    char2int = {c: i for i, c in int2char.items()}

    return int2char, char2int

def get_batches(data, n_seqs, seq_len):
    ''' a batch generator that returns batches of size
        n_seqs x seq_len from data
    '''  

    # calculate number of batches
    batch_size = n_seqs*seq_len
    n_batches = len(data)//batch_size
    
    # keep only enough characters to make full batches
    data = data[:n_batches * batch_size]
    # reshape into n_seqs rows
    data = data.reshape((n_seqs, -1))

    for n in range(0, data.shape[1], seq_len):
        # features
        x = data[:, n:n+seq_len]
        # targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], data[:, n+seq_len]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], data[:, 0]
        yield x, y

def one_hot_encode(data, n_labels):
    ''' encodes data with one-hot encoding
    '''
    
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*data.shape), n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), data.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*data.shape, n_labels))
    
    return one_hot
