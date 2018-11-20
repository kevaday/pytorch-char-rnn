import argparse
import os
import numpy as np
#import codecs

from model import CharRNN, train
from utils import get_lookup_tables

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', type=str, help='text file to train on')
    parser.add_argument('--encoding', type=str, default=None, help='encoding of the text file')
    parser.add_argument('--savepath', type=str, default='save', help='directory to save models')
    parser.add_argument('--save_every', type=int, default=1, help='save every SAVE_EVERY epochs')
    parser.add_argument('--rnntype', type=str, default='LSTM', help='type of rnn to use for model. Options are `LSTM`, `GRU`, `RNN_TANH`, or `RNN_RELU`')
    parser.add_argument('--bidirectional', action='store_true', default=False, help='use bidirectional RNN for model')
    parser.add_argument('--rnnsize', type=int, default=512, help='size of the hidden state of RNN')
    parser.add_argument('--nlayers', type=int, default=4, help='number of RNN layers')
    parser.add_argument('--batchsize', type=int, default=48, help='minibatch size')
    parser.add_argument('--seqlength', type=int, default=100, help='length of text sequences to train on')
    parser.add_argument('--nepochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--print_every', type=int, default=25, help='print frequency on batch iterations')
    parser.add_argument('--sample_every', type=int, default=5, help='sample from the model every SAMPLE_EVERY epochs')
    parser.add_argument('--valfract', type=float, default=0.1, help='fraction of text data to put aside for validation')
    parser.add_argument('--gradclip', type=float, default=5., help='value to clip gradients at')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--lrdiv', type=int, nargs='*', default=None, help='divide learning rate by 10 after LRDIV epochs. supports multiple')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout to apply to weights')
    parser.add_argument('--initrange', type=float, default=1., help='range to initialize decoder weights with')
    parser.add_argument('--cuda', action='store_true', default=False, help='train the network on GPU with CUDA')
    parser.add_argument('--fastest', action='store_true', default=False, help='set CuDNN to fastest mode')
    parser.add_argument('--benchmark', action='store_true', default=False, help='use CuDNN benchmark mode to find optimal algorithms')
    parser.add_argument('--init_from', type=str, default=None, help='directory of model file to initialize network from')
    args = parser.parse_args()

    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
        print(f'Directory {args.savepath} created')
    
    if not args.encoding:
        with open(args.infile, 'r') as f:
            text = f.read()
    else:
        with open(args.infile, 'r', encoding=args.encoding) as f:
            text = f.read()
    
    int2char, char2int = get_lookup_tables(text)
    encoded = np.array([char2int[c] for c in text])
    chars = tuple(char2int.keys())

    if not args.init_from:
        model = CharRNN(chars, int2char, char2int, rnn_type=args.rnntype,
                        bidirectional=args.bidirectional,
                        n_hidden=args.rnnsize, n_layers=args.nlayers, dropout=args.dropout,
                        lr=args.lr, initrange=args.initrange, cuda=args.cuda,
                        cudnn_fastest=args.fastest, cudnn_benchmark=args.benchmark)
    else:
        model = CharRNN.load(args.init_from, chars, int2char, char2int,
                             dropout=args.dropout, lr=args.lr, initrange=args.initrange, cuda=args.cuda,
                             cudnn_fastest=args.fastest, cudnn_benchmark=args.benchmark)
    
    val_loss = train(model, encoded, os.path.basename(args.infile), epochs=args.nepochs,
                     batch_size=args.batchsize, seq_len=args.seqlength, lr=args.lr,
                     clip=args.gradclip, val_fract=args.valfract, cuda=args.cuda,
                     print_every=args.print_every, sample_every=args.sample_every,
                     save_every=args.save_every, save_dir=args.savepath)

    savefile = f'{os.path.basename(args.infile)}_final_{val_loss:.4f}.ckpt'
    model.save(os.path.join(args.savepath, savefile))
    print(f'Network saved as {savefile}')

if __name__ == '__main__':
    main()
