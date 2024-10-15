from world import cprint,bprint
import world
from re_model import LightGCN
from re_dataloader import Loader
import torch
import numpy as np
import re_procedure as Procedure
import random 
import utils
import numpy as np




config = world.config
seed = config['seed']
device = world.device
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
dataset = Loader()
num_users = dataset.num_users
num_items = dataset.num_items
edge_index = dataset.getSparseGraph().to(device)
Recmodel = LightGCN(num_users=num_users,num_items=num_items,edge_index=edge_index)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss1(Recmodel,config)
save_path = './models/'
noise_ratio = round(config['noise_rate'],2)
save_file_name = 'LightGCN' + '-' + config['dataset'] + '-' + str(config['seed']) + '-' + str(noise_ratio) +'-'+str(config['latent_dim_rec'])+ '.pth.tar'
Neg_k = 1
w = None
best = ""
max_score = 0
patience = 0
for epoch in range(world.TRAIN_epochs):
    output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
    cprint("[TEST]")
    results = Procedure.Valid(dataset, Recmodel, epoch, w, world.config['multicore'], val=False)
    recall_20 = round(results['recall'][0], 4)
    ndcg_20 = round(results['ndcg'][0], 4)
    recall_50 = round(results['recall'][1], 4)
    ndcg_50 = round(results['ndcg'][1], 4)
    score = recall_20 + ndcg_20
    if score > max_score:
        patience = 0
        max_score = score
        torch.save(Recmodel.state_dict(),save_path+save_file_name)
        bprint('es model selected , best epoch saved')
        best = f'Best Validation Status[(r@20,n@20,r@50,n@50):{recall_20},{ndcg_20},{recall_50},{ndcg_50}]'
    else:
        patience += 1
    if patience > 150:
        break
    topk_txt = f'Testing EPOCH[{epoch + 1}/{world.TRAIN_epochs}]  {output_information} | Results Top-k (r@20,n@20,r@50,n@50): {recall_20}, {ndcg_20},{recall_50},{ndcg_50}'
    print(topk_txt)
del Recmodel
Recmodel = LightGCN(num_users=num_users,num_items=num_items,edge_index=edge_index)
# Recmodel = GTN(config,dataset,args = world.args)
Recmodel = Recmodel.to(world.device)
Recmodel.load_state_dict(torch.load(save_path + save_file_name,map_location=torch.device('cpu')))
results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], val=False)
recall_20 = round(results['recall'][0], 4)
ndcg_20 = round(results['ndcg'][0], 4)
recall_50 = round(results['recall'][1], 4)
ndcg_50 = round(results['ndcg'][1], 4)
test = f'Testing Status[(r@20,n@20,r@50,n@50):{recall_20},{ndcg_20},{recall_50},{ndcg_50}]'
bprint(test)
