import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载pth文件
embeddings_pth = torch.load('/root/cyj/NRGCF_Pytorch/code/au==1.0 dataset==yelp2018_0.pth')  # 修改为你的文件路径

# 假设 pth 文件包含 'user_embeddings' 和 'item_embeddings' 两个键，分别对应用户和物品的嵌入
num_users_embeddings = embeddings_pth['user_emb'].cpu().numpy()  # 转换为 NumPy 数组
num_items_embeddings = embeddings_pth['item_emb'].cpu().numpy()  # 转换为 NumPy 数组

# 合并用户和物品的嵌入
embeddings = np.vstack((num_users_embeddings, num_items_embeddings))

# 使用 t-SNE 降维到2维
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# 分离降维后的用户和物品嵌入
num_users_2d = embeddings_2d[:num_users_embeddings.shape[0]]
num_items_2d = embeddings_2d[num_users_embeddings.shape[0]:]

# 绘制用户和物品的嵌入点，使用不同的颜色
plt.figure(figsize=(10, 6))
plt.scatter(num_users_2d[:, 0], num_users_2d[:, 1], c='blue', label='Users', alpha=0.6)
plt.scatter(num_items_2d[:, 0], num_items_2d[:, 1], c='red', label='Items', alpha=0.6)
plt.legend()
plt.title('t-SNE Visualization of User and Item Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()
