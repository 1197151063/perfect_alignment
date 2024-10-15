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
save_path = '/root/autodl-tmp/models/'
noise_ratio = round(config['noise_rate'],2)
save_file_name = 'LightGCN1' + '-' + config['dataset'] + '-' + str(config['seed']) + '-' + str(noise_ratio) +'-'+str(config['latent_dim_rec'])+ '.pth.tar'
Neg_k = 1
w = None
best = ""
best_test=""
max_score = 0
patience = 0
for epoch in range(world.TRAIN_epochs):
    output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
    cprint("[TEST]")
    results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], val=False)
    recall_20 = round(results['recall'][0], 4)
    ndcg_20 = round(results['ndcg'][0], 4)
    recall_50 = round(results['recall'][1], 4)
    ndcg_50 = round(results['ndcg'][1], 4)
    recall_20_v = round(results['recall_v'][0], 4)
    ndcg_20_v = round(results['ndcg_v'][0], 4)
    recall_50_v = round(results['recall_v'][1], 4)
    ndcg_50_v = round(results['ndcg_v'][1], 4)
    score = recall_20_v + ndcg_20_v
    val_info = f'Val[(r@20,n@20,r@50,n@50):{recall_20_v},{ndcg_20_v},{recall_50_v},{ndcg_50_v}]'
    test_info = f'Test[(r@20,n@20,r@50,n@50):{recall_20},{ndcg_20},{recall_50},{ndcg_50}]'
    if score > max_score:
        patience = 0
        max_score = score
        # torch.save(Recmodel.state_dict(),save_path+save_file_name)
        bprint('es model selected , best epoch saved')
        best = val_info
        best_test = test_info
    else:
        patience += 1
    if patience > 100:
        break
    topk_txt = f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information} {val_info} {test_info}'
    print(topk_txt)
bprint(best_test)

