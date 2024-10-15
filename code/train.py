#The module of early stopping 

from dataloader import Loader
import common
from common import cprint,bprint
from model import LightGCN
import utils
import procedure
import random
import torch
device = common.device
seed = common.seed
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
args = common.args
Neg_k = 1
w = None
model_name = 'Robust-' + args.model
save_path = '/root/autodl-tmp/models/'
dataset = Loader(args=args)
noise_ratio = round(args.noise_rate,2)
save_file_name = model_name + '-' + args.dataset + '-' + str(args.seed) + '-' + str(noise_ratio) + '.pth.tar'
edge_index = dataset.adj_mat.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
Recmodel = LightGCN(num_users=num_users,num_items=num_items,edge_index=edge_index)
bpr = utils.BPR(Recmodel)   
Recmodel = Recmodel.to(device)
max_score = 0.
patience = 0.
best = ""
for epoch in range(common.TRAIN_epochs):
    output_information = procedure.trainer(dataset, Recmodel, bpr)
    cprint("[TEST]")
    recall_20,ndcg_20,recall_50,ndcg_50 = procedure.test(dataset=dataset, model=Recmodel, k=[20,50])
    recall_20 = round(recall_20,4)
    ndcg_20 = round(ndcg_20,4)
    recall_50 = round(recall_50,4)
    ndcg_50 = round(ndcg_50,4)
    score = recall_20 + ndcg_20
    if score > max_score:
        patience = 0
        max_score = score
        bprint('es model selected , best epoch saved')
        best = f'Best Epoch Status[(r@20,n@20,r@50,n@50):{recall_20},{ndcg_20},{recall_50},{ndcg_50}]'
        torch.save(Recmodel.state_dict(),save_path+save_file_name)
    else:
        patience += 1
    if patience > 1000:
        break
    topk_txt = f'Testing EPOCH[{epoch + 1}/{common.TRAIN_epochs}]  {output_information} | Results Top-k (r@20, n@20, r@50, n@50):{recall_20}, {ndcg_20} , {recall_50}, {ndcg_50} '
    print(topk_txt)
bprint(best)