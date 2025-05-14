import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import wandb
import os
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bottleneck = nn.Linear(hidden_dim, bottleneck_dim)  # 添加瓶颈层

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = self.bottleneck(x)  # 瓶颈表征
        return z

# 数据集封装
class XYFeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx] if isinstance(self.X[idx], torch.Tensor) else torch.tensor(self.X[idx], dtype=torch.float32)
        y = self.y[idx] if isinstance(self.y[idx], torch.Tensor) else torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y

# 类别采样器
class CategoriesSampler:
    def __init__(self, X, y, epoch_size, K_way, N_shot_query):
        self.X = X
        self.y = y
        self.epoch_size = epoch_size
        self.K_way = K_way
        self.N_shot_query = N_shot_query
        self.indexes_per_class = self._prepare_class_indexes()

    def _prepare_class_indexes(self):
        class_indexes = {}
        for idx, label in enumerate(self.y):
            if label not in class_indexes:
                class_indexes[label] = []
            class_indexes[label].append(idx)
        return class_indexes

    def __iter__(self):
        for _ in range(self.epoch_size):
            batch_indexes = []
            classes = np.random.choice(list(self.indexes_per_class.keys()), self.K_way, replace=False)
            for c in classes:
                indexes = np.random.choice(self.indexes_per_class[c], self.N_shot_query, replace=False)
                batch_indexes.extend(indexes)
            yield batch_indexes

    def __len__(self):
        return self.epoch_size

# 原型网络
class PrototypicalNet:
    def __init__(self, X_train, X_test, y_train, y_test, input_dim, hidden_dim, input_texts_test, bottleneck_dim, device):
        self.device = device
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, bottleneck_dim).to(self.device)
        self.train_dataset = XYFeatureDataset(X_train, y_train)
        self.test_dataset = XYFeatureDataset(X_test, y_test)
        self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), weight_decay=1e-5, lr=0.0003)
        self.bottleneck_dim = bottleneck_dim

    def bottleneck_loss(self, z, beta=1e-3):
        # 假设 z ~ N(0, σ^2)，直接计算约束信息量
        z_var = torch.var(z, dim=0)  # 每个维度的方差
        ib_loss = beta * torch.sum(torch.log(1 + z_var))  # 近似的压缩损失
        return ib_loss

    def cal_euc_distance(self, query_z, center, K_way, N_query):
        center = center.unsqueeze(0).expand(K_way * N_query, K_way, self.bottleneck_dim)
        query_z = query_z.unsqueeze(1).expand(K_way * N_query, K_way, self.bottleneck_dim)
        return torch.pow(query_z - center, 2).sum(2)

    def cal_cosine_distance(self, query_z, prototypes, K_way, N_query):
        """
        计算余弦距离：
        - query_z: 查询样本的特征 (K_way * N_query, z_dim)
        - prototypes: 类别原型 (K_way, z_dim)
        """
        # 归一化
        query_z_norm = F.normalize(query_z, p=2, dim=1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)

        # 计算余弦相似度
        cosine_similarity = torch.mm(query_z_norm, prototypes_norm.t())  # (K_way * N_query, K_way)

        # 将相似度转为距离
        cosine_distance = 1 - cosine_similarity
        return cosine_distance  # 返回距离矩
    
    def cal_manhattan_distance(self, query_z, prototypes, K_way, N_query):
        """
        计算曼哈顿距离：
        - query_z: 查询样本的特征，形状为 (K_way * N_query, z_dim)
        - prototypes: 类别原型，形状为 (K_way, z_dim)
        - K_way: 类别数
        - N_query: 每类查询样本数
        """
        assert query_z.size(0) == K_way * N_query, "query_z 的形状不符合 K_way * N_query 的要求"
        assert prototypes.size(0) == K_way, "prototypes 的形状不符合 K_way 的要求"

        # 扩展维度以便进行广播
        query_z_expanded = query_z.unsqueeze(1).expand(K_way * N_query, K_way, -1)  # (K_way * N_query, K_way, z_dim)
        prototypes_expanded = prototypes.unsqueeze(0).expand(K_way * N_query, K_way, -1)  # (K_way * N_query, K_way, z_dim)

        # 计算曼哈顿距离
        manhattan_distance = torch.abs(query_z_expanded - prototypes_expanded).sum(dim=2)  # (K_way * N_query, K_way)

        return manhattan_distance
    
    def set_forward_loss(self, K_way, N_shot, N_query, sample_datas):
        z = self.feature_extractor(sample_datas)
        z = z.view(K_way, N_shot + N_query, -1)

        support_z = z[:, :N_shot]
        query_z = z[:, N_shot:].contiguous().view(K_way * N_query, -1)
        center = torch.mean(support_z, dim=1)

        # 分类损失与信息瓶颈损失
        loss, acc = self.loss_acc(query_z, center, K_way, N_query)
        ib_loss = self.bottleneck_loss(query_z)
        total_loss = loss + ib_loss
        return total_loss, acc

    def loss_acc(self, query_z, prototypes, K_way, N_query):
        target_inds = torch.arange(0, K_way).view(K_way, 1).expand(K_way, N_query).long().to(self.device)
        distance = self.cal_euc_distance(query_z, prototypes, K_way, N_query)
        predict_label = torch.argmin(distance, dim=1)
        acc = torch.eq(target_inds.contiguous().view(-1), predict_label).float().mean()

        loss = F.log_softmax(-distance, dim=1).view(K_way, N_query, K_way)
        loss = -loss.gather(dim=2, index=target_inds.unsqueeze(2)).view(-1).mean()
        return loss, acc
    
    def train(self, epochs, epoch_size, X_train, X_test, y_train, y_test, input_texts_test):
        K_way = 2
        N_shot = 5
        N_query = 5
        self.feature_extractor.train()
        train_sampler = CategoriesSampler(X_train, y_train, epoch_size, K_way, N_shot + N_query)
        train_loader = DataLoader(dataset=self.train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0

            for i, batch_data in enumerate(train_loader):
                imgs, _ = batch_data[0].to(self.device), batch_data[1]
                loss, acc = self.set_forward_loss(K_way, N_shot, N_query, imgs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            wandb.log({"protonet_train_loss": epoch_loss / len(train_loader),
                       "protonet_train_acc": epoch_acc / len(train_loader)})

            # 测试集评估
            test_acc, test_auc, tpr_at_5_fpr = self.eval_model_full(X_test, y_test, input_texts_test, epoch+1)
            wandb.log({"protonet_test_acc": test_acc, "protonet_test_auc": test_auc, "tpr_at_5_fpr": tpr_at_5_fpr})

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {epoch_acc/len(train_loader):.4f}")
            print(f"Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}, TPR@5% FPR: {tpr_at_5_fpr:.4f}")

    
    
    def eval_model_full(self, X_test, y_test, input_texts_test, epoch):
        self.feature_extractor.eval()
        with torch.no_grad():
            X_train_tensor = torch.tensor(self.train_dataset.X).to(self.device)
            y_train_tensor = torch.tensor(self.train_dataset.y).to(self.device)

            prototypes = []
            for class_label in torch.unique(y_train_tensor):
                class_samples = X_train_tensor[y_train_tensor == class_label]
                prototype = torch.mean(self.feature_extractor(class_samples), dim=0)
                prototypes.append(prototype)
            prototypes = torch.stack(prototypes)

            X_test_tensor = torch.tensor(X_test).to(self.device)
            test_z = self.feature_extractor(X_test_tensor)

            distances = torch.norm(test_z.unsqueeze(1) - prototypes.unsqueeze(0), dim=2)
            logits = -distances
            probabilities = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # 选择正类的概率
            
            pred_labels = torch.argmin(distances, dim=1)

            acc = torch.eq(pred_labels.cpu(), torch.tensor(y_test)).float().mean().item()
            # preds = distances[:, 0].cpu().numpy() - distances[:, 1].cpu().numpy()
            auc = roc_auc_score(y_test, probabilities)
            fpr, tpr, thresholds = roc_curve(y_test, probabilities)
            fpr_5_idx = np.where(fpr <= 0.05)[0][-1]
            tpr_at_5_fpr = tpr[fpr_5_idx]
            return acc, auc, tpr_at_5_fpr