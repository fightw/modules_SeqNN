import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import scipy.stats
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import Subset

# DNA序列到数字的映射
CODES = {
    "A": 0,
    "T": 1,
    "G": 2,
    "C": 3,
    "N": 4,
}

def n2id(n):
    """将碱基字符转换为对应的数字。"""
    return CODES[n.upper()]

class Seq2Tensor(torch.nn.Module):
    """使用one-hot编码对序列进行编码。"""
    def __init__(self):
        super().__init__()

    def forward(self, seq):
        seq = [n2id(x) for x in seq]

        code = torch.tensor(seq, dtype=torch.long)
        code = F.one_hot(code, num_classes=5).float()
        return code.transpose(0, 1)

class SeqDatasetProb(Dataset):
    """序列数据集。"""
    def __init__(self, ds, seqsize, shift=0.5, scale=0.5):
        self.ds = ds
        self.seqsize = seqsize
        self.totensor = Seq2Tensor()
        self.shift = shift
        self.scale = scale

    # def transform(self, x):
    #     assert isinstance(x, str)
    #     if len(x) > self.seqsize:
    #       x = x[:self.seqsize]  # 对长度超过的序列进行截断
    #     assert len(x) == self.seqsize
    #     return self.totensor(x)

    def transform(self, x):
        assert isinstance(x, str)
        if len(x) < self.seqsize:
            x = x + 'N' * (self.seqsize - len(x))  # 对长度不足的序列进行填充
        # if len(x) < self.seqsize:
        #     x = x.ljust(self.seqsize, ' ')  # 使用空白填充
        elif len(x) > self.seqsize:
            x = x[:self.seqsize]  # 对长度超过的序列进行截断
        assert len(x) == self.seqsize
        return self.totensor(x)

    def __getitem__(self, i):
        seq = self.transform(self.ds.seq.values[i])
        X = seq

        bin = self.ds.bin.values[i]
        MRL = self.ds.TE2.values[i]
        norm = scipy.stats.norm(loc=bin + self.shift, scale=self.scale)
        points = np.array([-np.inf, *range(1, 9, 1), np.inf])
        cumprobs = norm.cdf(points)
        probs = cumprobs[1:] - cumprobs[:-1]

        return X, probs, MRL

    def __len__(self):
        return len(self.ds.seq)

# def create_dataloaders(train_df, test_df, seqsize, batch_size=32, test_size=0.2):
#     """创建训练集和测试集的DataLoader"""
#     # train_df, test_df = train_test_split(dataset, test_size=test_size, stratify=dataset.bin)
#     train_dataset = SeqDatasetProb(train_df, seqsize)
#     test_dataset = SeqDatasetProb(test_df, seqsize)
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     return train_loader, test_loader

def create_dataloaders(dataset, seqsize, batch_size=32, num_test_samples=883):
    """
    创建训练集和测试集的DataLoader

    参数:
    - dataset: 输入的数据集，可以是一个 pandas DataFrame 或类似结构的数据对象。
    - seqsize: 每个序列的长度。
    - batch_size: 每个批次中的样本数量，默认为32。
    - num_test_samples: 测试集的大小，即要从数据集中取出作为测试集的样本数量。

    返回值:
    - train_loader: 训练集的 DataLoader 对象，用于按批次加载训练数据。
    - test_loader: 测试集的 DataLoader 对象，用于按批次加载测试数据。
    """
    # 确定测试集的起始索引
    start_index = len(dataset) - num_test_samples

    # 将数据集按照 num_test_samples 划分为训练集和测试集
    train_df = dataset.iloc[:start_index]
    test_df = dataset.iloc[start_index:]

    # 创建训练集和测试集的序列数据集对象
    train_dataset = SeqDatasetProb(train_df, seqsize)
    test_dataset = SeqDatasetProb(test_df, seqsize)

    # 创建 DataLoader 对象，用于批量加载数据
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#原始数据加载器
# def create_dataloaders(dataset, seqsize, batch_size=32, test_size=0.2):
#     """创建训练集和测试集的DataLoader"""
#     train_df, test_df = train_test_split(dataset, test_size=test_size)
#     train_dataset = SeqDatasetProb(train_df, seqsize)
#     test_dataset = SeqDatasetProb(test_df, seqsize)
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     return train_loader, test_loader

#十折交叉数据加载器
# def create_dataloaders(train_df, val_df, seqsize, batch_size=32):
#     """创建训练集和验证集的DataLoader"""
#     train_dataset = SeqDatasetProb(train_df, seqsize)
#     val_dataset = SeqDatasetProb(val_df, seqsize)
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
#     return train_loader, val_loader