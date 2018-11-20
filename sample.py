''' by Kevi Aday 2018
'''

import argparse

from model import CharRNN, sample

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', type=str, default=None, help='checkpointed model to sample from')
    parser.add_argument('--cuda', action='store_true', default=False, help='run the network on the GPU with CUDA')
    parser.add_argument('--length', type=int, default=256, help='number of characters to sample from the model')
    parser.add_argument('--prime', type=str, default='The ', help='start sampling from network with these starting characters')
    parser.add_argument('--top_k', type=int, default=10, help='find most probable next char from the output softmax with top_k')
    parser.add_argument('--savefile', type=str, default=None, help='optional, save sampled text to this file')
    parser.add_argument('--encoding', type=str, default=None, help='optional, encoding of text file to save as')

    args = parser.parse_args()

    model = CharRNN.load(args.model, args.cuda)

    if not args.savefile:
        print(sample(model, length=args.length, prime=args.prime, cuda=args.cuda, top_k=args.top_k))
    else:
        sample(model, length=args.length, prime=args.prime, cuda=args.cuda, top_k=args.top_k, file=args.savefile, encoding=args.encoding)

if __name__ == '__main__':
    main()
