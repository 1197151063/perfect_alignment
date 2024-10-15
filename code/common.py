import torch
from parse import parse_args
import multiprocessing

args = parse_args()
GPU = torch.cuda.is_available()
torch.cuda.set_device(args.gpu_id)
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed
TRAIN_epochs = args.epochs

def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

def bprint(words:str):
    print(f"\033[0;30;45m{words}\033[0m")