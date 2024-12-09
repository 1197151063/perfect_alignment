import torch
from torch import Tensor
import numpy as np
from torch_geometric.utils import degree
import world
import utils
import multiprocessing
import model
from model import SGL,SimGCL,CenNorm,LightGCN
device = world.device
config = world.config
CORES = multiprocessing.cpu_count() // 2
"""
define evaluation metrics here
Already implemented:[Recall@K,NDCG@K]
"""
def train_bpr(dataset,model:LightGCN,opt):
    model = model
    model.train()
    S = utils.Fast_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(S)
    for edge_label_index in S:
        opt.zero_grad()

        pos_rank,neg_rank = model(edge_label_index)
        bpr_loss,reg_loss = model.recommendation_loss(pos_rank,neg_rank,edge_label_index)
        i_loss = model.item_constraint_loss(edge_label_index)
        hn_loss =  model.hn_loss(edge_label_index)
        loss = bpr_loss + reg_loss + i_loss + hn_loss
        loss.backward()

        # uni_loss = model.uniformity_loss(edge_label_index)
        # loss_1 = hn_loss+uni_loss
        # loss_1.backward()
        opt.step()   
        aver_loss += (loss)
    aver_loss /= total_batch
    return f"average loss {aver_loss:5f}"

def train_bpr_aligngcn(dataset,model:CenNorm,opt):
    model = model
    model.train()
    S = utils.Fast_Sampling(dataset=dataset)
    aver_loss = 0.
    aver_align_loss = 0.
    aver_uni_loss = 0.
    total_batch = len(S)
    for edge_label_index in S:
        alignment_loss = model.alignment_loss(edge_label_index)
        uniformity_loss = model.uniformity_loss(edge_label_index)
        # decor_loss = model.cal_decorelation_loss(edge_label_index)
        loss = alignment_loss + uniformity_loss
        aver_align_loss += (alignment_loss)
        aver_uni_loss += (uniformity_loss)
        opt.zero_grad()
        loss.backward()
        opt.step()   
        aver_loss += (loss)
    aver_loss /= total_batch
    aver_align_loss /= total_batch
    aver_uni_loss /= total_batch
    return f"loss:{aver_loss:.3f},alignment:{aver_align_loss:.3f},uniformity:{aver_uni_loss:.3f}"


def train_bpr_sgl(dataset,
                  model:SGL,
                  opt,
                  edge_index1,
                  edge_index2):
    model = model
    model.train()
    S = utils.Fast_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(S)
    for edge_label_index in S:
        pos_rank,neg_rank = model(edge_label_index)
        bpr_loss = model.bpr_loss(pos_rank,neg_rank)
        ssl_loss = model.ssl_loss(edge_index1,edge_index2,edge_label_index)
        L2_reg = model.L2_reg(edge_label_index)
        i_i_cons = model.item_constraint_loss(edge_label_index)
        loss = bpr_loss + ssl_loss + L2_reg + i_i_cons
        opt.zero_grad()
        loss.backward()
        opt.step()    
        aver_loss += (bpr_loss + ssl_loss + L2_reg)
    aver_loss /= total_batch
    return f"average loss {aver_loss:5f}"


def train_bpr_simgcl(dataset,
                  model:SimGCL,
                  opt):
    model = model
    model.train()
    S = utils.Fast_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(S)
    for edge_label_index in S:
        pos_rank,neg_rank = model(edge_label_index)
        bpr_loss = model.bpr_loss(pos_rank,neg_rank)
        ssl_loss = model.ssl_loss(edge_label_index)
        L2_reg = model.L2_reg(edge_label_index)
        loss = bpr_loss + ssl_loss + L2_reg
        opt.zero_grad()
        loss.backward()
        opt.step()    
        aver_loss += (bpr_loss + ssl_loss + L2_reg)
    aver_loss /= total_batch
    return f"average loss {aver_loss:5f}"


@torch.no_grad()
def test(k_values:list,
         model,
         train_edge_index,
         test_edge_index,
         num_users,
         ):
    model.eval()
    recall = {k: 0 for k in k_values}
    ndcg = {k: 0 for k in k_values}
    total_examples = 0
    for start in range(0, num_users, 2048):
        end = start + 2048
        if end > num_users:
            end = num_users
        src_index=torch.arange(start,end).long().to(device)
        logits = model.link_prediction(src_index=src_index,dst_index=None)

        # Exclude training edges:
        mask = ((train_edge_index[0] >= start) &
                (train_edge_index[0] < end))
        masked_interactions = train_edge_index[:,mask]
        logits[masked_interactions[0] - start,masked_interactions[1]] = float('-inf')
        # Generate ground truth matrix
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((test_edge_index[0] >= start) &
                (test_edge_index[0] < end))
        masked_interactions = test_edge_index[:,mask]
        ground_truth[masked_interactions[0] - start,masked_interactions[1]] = True
        node_count = degree(test_edge_index[0, mask] - start,
                            num_nodes=logits.size(0))
        topk_indices = logits.topk(max(k_values),dim=-1).indices
        for k in k_values:
            topk_index = topk_indices[:,:k]
            isin_mat = ground_truth.gather(1, topk_index)
            # Calculate recall
            recall[k] += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
            # Calculate NDCG
            log_positions = torch.log2(torch.arange(2, k + 2, device=logits.device).float())
            dcg = (isin_mat / log_positions).sum(dim=-1)
            ideal_dcg = torch.zeros_like(dcg)
            for i in range(len(dcg)):
                ideal_dcg[i] = (1.0 / log_positions[:node_count[i].clamp(0, k).int()]).sum()
            ndcg[k] += float((dcg / ideal_dcg.clamp(min=1e-6)).sum())

        total_examples += int((node_count > 0).sum())

    recall = {k: recall[k] / total_examples for k in k_values}
    ndcg = {k: ndcg[k] / total_examples for k in k_values}

    return recall,ndcg

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Test(dataset, Recmodel, epoch, w=None, multicore=1, val=False):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    valDict: dict = dataset.valDict

    Recmodel: model.LightGCN
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'precision_v': np.zeros(len(world.topks)),
               'recall_v': np.zeros(len(world.topks)),
               'ndcg_v': np.zeros(len(world.topks)),
               }
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        valid_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)  # can speed up: self._allPos
            val_list = [valDict[u] for u in batch_users]
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.link_prediction(batch_users_gpu,None)   
            # rating = rating.cpu()
            exclude_index = []  
            exclude_items = []
            # posivite instances
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating[exclude_index, exclude_items] = -(1 << 10)

            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            valid_list.append(val_list)
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        X_val = zip(rating_list,valid_list)
        pre_results = pool.map(test_one_batch, X)
        val_results = pool.map(test_one_batch,X_val)  
        for result in pre_results:
            results['recall'] += result['recall']
            # results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        # results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        for result in val_results:
            results['recall_v'] += result['recall']
            # results['precision_v'] += result['precision']
            results['ndcg_v'] += result['ndcg']
        results['recall_v'] /= float(len(users))
        # results['precision_v'] /= float(len(users))
        results['ndcg_v'] /= float(len(users))
        if multicore == 1:
            pool.close()

        return results
    
def Valid(dataset, Recmodel, epoch, w=None, multicore=1, val=False):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    # testDict: dict = dataset.testDict
    valDict: dict = dataset.valDict

    Recmodel: model.LightGCN
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(valDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)  # can speed up: self._allPos
            groundTrue = [valDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)   
            # rating = rating.cpu()
            exclude_index = []  
            exclude_items = []
            # posivite instances
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating[exclude_index, exclude_items] = -(1 << 10)

            _, rating_K = torch.topk(rating, k=max_K)
            # rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
                
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if multicore == 1:
            pool.close()

        return results
def Recall_K(ground_truth,r,k):
    num_correct_pred = torch.sum(r,dim=-1)
    user_num_liked = Tensor([len(ground_truth[i]) for i in range(len(ground_truth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    return recall.item()

def NDCG_K(ground_truth,r,k):
    assert len(r) == len(ground_truth)
    test_matrix = torch.zeros((len(r),k))
    for i,item in enumerate(ground_truth):
        length = min(len(item),k)
        test_matrix[i,:length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2,k + 2)),axis = 1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()

def get_user_positive_items(edge_index):
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items

    # user_embedding = np.array(model.user_emb.weight.cpu().detach().numpy())
    # item_embedding = np.array(model.item_emb.weight.cpu().detach().numpy())
    # rating = torch.tensor(np.matmul(user_embedding,item_embedding.T))

    # user_embedding = model.user_emb.weight
    # item_embedding = model.item_emb.weight
    # rating = torch.matmul(user_embedding,item_embedding.T)
    e_index = model.dataset.getSparseGraph().to(world.device)
    user_embedding,_,item_embedding,_ = model.forward(e_index)
    rating = torch.matmul(user_embedding,item_embedding.T)
    user_pos_items = get_user_positive_items(exclude_edge_index)
    exclude_user = []
    exclude_item = []
    for user,items in user_pos_items.items():
        exclude_user.extend([user]*len(items))
        exclude_item.extend(items)
    rating[exclude_user,exclude_item] = -(1 << 10)
    _, top_k_items = torch.topk(rating,k=k)

    users = edge_index[0].unique()
    test_user_pos_item = get_user_positive_items(edge_index)
    test_user_pos_item_list = [test_user_pos_item[user.item()] for user in users]

    r = []
    for user in users:
        ground_truth_items = test_user_pos_item[user.item()]
        label = list(map(lambda x : x in ground_truth_items,top_k_items[user]))
        r.append(label)
    r = Tensor(np.array(r).astype('float'))

    recall = Recall_K(test_user_pos_item_list,r,k)
    ndcg = NDCG_K(test_user_pos_item_list,r,k)

    return recall,ndcg
