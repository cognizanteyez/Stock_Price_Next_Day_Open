import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--corpusFile', default='data/000001SH_index.csv')


# TODO 
parser.add_argument('--gpu', default=0, type=int) # gpu 
parser.add_argument('--epochs', default=100, type=int) #
parser.add_argument('--layers', default=2, type=int) # LSTM
parser.add_argument('--input_size', default=5, type=int) # 
parser.add_argument('--hidden_size', default=32, type=int) # 
parser.add_argument('--lr', default=0.0001, type=float) # learning rate 
parser.add_argument('--sequence_length', default=5, type=int) # sequence
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--useGPU', default=False, type=bool) # GPU
parser.add_argument('--batch_first', default=True, type=bool) # batch_size
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--save_file', default='model/stock.pkl') # save file


args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
args.device = device