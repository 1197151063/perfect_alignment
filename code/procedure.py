import common
import numpy as np
import torch
from dataloader import Loader
from utils import BPR
from model import LightGCN
import torch_geometric.utils as pyg_utils
args = common.args
device = common.device
batch_size = args.bpr_batch
def trainer(dataset:Loader,rec_model,loss:BPR):
    train_loader = dataset.train_loader
    train_edge_index = dataset.train_edge_index.to(device)
    edge_index = dataset.edge_index.to(device)
    batches = len(train_loader.dataset) // args.bpr_batch
    if len(train_loader.dataset) % args.bpr_batch != 0:
        batches += 1
    rec_model = rec_model
    loss = loss
    rec_loss = 0
    for index in train_loader:
        pos = train_edge_index[:,index]
        all_interacted = dataset.get_user_all_interacted(pos[0])
        while True:
            neg = np.random.randint(0,dataset.num_items,size=len(pos[0]))
            if neg in all_interacted:
                 continue
            else:
                 break
        neg = torch.from_numpy(neg).to(device)
        users = pos[0].to(device)
        pos = pos[1].to(device)
        # neg = neg.to(device)
        bpr_loss = loss.stageOne(users,pos,neg)
        rec_loss += bpr_loss
    rec_loss /= batches
    return f"loss {rec_loss:.5f}"



def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
	"""
    Mean Reciprocal Rank
    """
	pred_data = r[:, :k]
	scores = np.log2(1. / np.arange(1, k + 1))
	pred_data = pred_data / scores
	pred_data = pred_data.sum(1)
	return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)
    return  ret['recall'], NDCGatK_r(groundTrue,r,k)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def test(model:LightGCN, dataset:Loader,k:list):
    topk = k
    n_user = dataset.num_users
    test_loader = dataset.test_loader
    test_ground_truth_list = dataset.test_ground_truth_list
    mask = dataset.mask
    rating_list = []
    groundTrue_list = []

    with torch.no_grad():
        model.eval()
        Recall_20, NDCG_20 = 0,0
        Recall_50, NDCG_50 = 0,0
        for k in topk:
            for idx, batch_users in enumerate(test_loader):            
                batch_users = batch_users.to(device)
                rating = model.getUsersRating(batch_users)
                rating = rating.cpu()
                rating += mask[batch_users.cpu()]
                _, rating_K = torch.topk(rating, k=k)
                rating_list.append(rating_K)
                groundTrue_list.append([test_ground_truth_list[u] for u in batch_users])
            X = zip(rating_list, groundTrue_list)
            for i,x in enumerate(X):
                recall,ndcg = test_one_batch(x,k)
                if k == 20:
                      Recall_20 += recall
                      NDCG_20 += ndcg
                if k == 50:
                      Recall_50 += recall
                      NDCG_50 += ndcg
        Recall_20 /= n_user
        NDCG_20 /= n_user
        Recall_50 /= n_user
        NDCG_50 /= n_user

    return Recall_20, NDCG_20,Recall_50,NDCG_50
