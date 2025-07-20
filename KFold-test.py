# from sklearn.model_selection import KFold
# import torch
# from torch.utils.data import DataLoader
# import numpy as np
# from tqdm import tqdm
# from scipy.stats import pearsonr
# from sklearn.metrics import r2_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from data_process_SeqNN import *
# from SeqNN_Effblock import LegNet
# from train_module import Trainer
#
# from data_process_SeqNN import *
# import pandas as pd
# # 读取数据
# path = 'dataSet/PC3_5UTR_合成基因.csv'
# UTR_df = pd.read_csv(path)
# print(UTR_df)
#
# # 初始化模型
# model = LegNet(use_single_channel=False, use_reverse_channel=False)
# print(model)
#
# # 设定设备（GPU 或 CPU）
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 设置KFold交叉验证
# kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10折交叉验证
# fold_results = []
#
# # K折交叉验证
# for fold, (train_idx, val_idx) in enumerate(kf.split(UTR_df)):
#     print(f"\nTraining fold {fold + 1}...")
#
#     # 划分训练集和验证集
#     train_df = UTR_df.iloc[train_idx]
#     val_df = UTR_df.iloc[val_idx]
#
#     # 创建训练和验证的DataLoader
#     train_loader, val_loader = create_dataloaders(train_df, seqsize=100)
#
#     # 初始化 Trainer
#     trainer = Trainer(model, train_loader, val_loader, device, task_type='classification', epochs=50)
#
#     # 训练模型
#     train_loss_values, train_accuracy_values, val_loss_values, val_accuracy_values = trainer.train()
#
#     # 保存当前折的模型
#     torch.save(model.state_dict(), f'fold_{fold + 1}_best_trained_model.pth')
#
#     # 在验证集上评估模型
#     model.load_state_dict(torch.load(f'fold_{fold + 1}_best_trained_model.pth'))
#
#     # 用验证集进行预测
#     y_pred = []
#     y_true = []
#
#     for inputs, labels, TE in tqdm(val_loader, desc=f"Predicting fold {fold + 1}"):
#         inputs, labels, TE = inputs.to(device), labels.to(device), TE.float().to(device)
#         model_output = model(inputs)
#         yprobs, score = model_output
#         y_pred.append(score.detach().cpu().numpy().squeeze())
#         y_true.append(TE.cpu().numpy().squeeze())
#
#     # 计算 Pearson 相关系数和 R2 得分
#     y_pred = np.hstack(y_pred)
#     y_true = np.hstack(y_true)
#
#     r, p_value = pearsonr(y_pred, y_true)
#     r2 = r2_score(y_true, y_pred)
#
#     print(f"Fold {fold + 1} - Pearson Correlation: {r:.3f}, R2 Score: {r2:.3f}")
#
#     # 存储每个折的结果
#     fold_results.append((r, r2))
#
#     # 绘制当前折的散点图和拟合曲线
#     sns.jointplot(x=y_true, y=y_pred, scatter_kws={'s': 0.8}, kind='reg')
#     plt.xlabel('True Values')
#     plt.ylabel('Predictions')
#     plt.text(0.98, 0.95, f'R2 Score: {r2:.2f}\nPearson r: {r:.2f}',
#              transform=plt.gca().transAxes, ha='right', va='top', fontsize=10)
#     plt.title(f"Fold {fold + 1} - R2: {r2:.2f}, Pearson r: {r:.2f}")
#     plt.show()
#
# # 计算所有折的平均性能
# mean_r = np.mean([fold_result[0] for fold_result in fold_results])
# mean_r2 = np.mean([fold_result[1] for fold_result in fold_results])
#
# print(f"\nAverage Pearson Correlation over all folds: {mean_r:.3f}")
# print(f"Average R2 Score over all folds: {mean_r2:.3f}")
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from data_process_SeqNN import *
from SeqNN_Effblock import LegNet
from train_module import Trainer

import pandas as pd
# 读取数据
path = 'dataSet/traingene-5UTR-6720.csv'
UTR_df = pd.read_csv(path)
print(UTR_df)

# 设定设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置KFold交叉验证
kf = KFold(n_splits=10, shuffle=True)  # 10折交叉验证
fold_results = []

# K折交叉验证
for fold, (train_idx, val_idx) in enumerate(kf.split(UTR_df)):
    print(f"\nTraining fold {fold + 1}...")

    # 划分训练集和验证集
    train_df = UTR_df.iloc[train_idx]
    val_df = UTR_df.iloc[val_idx]

    # 创建训练和验证的DataLoader
    train_loader, val_loader = create_dataloaders(train_df, val_df, seqsize=100)

    # 在每一折开始时重新初始化模型
    model = LegNet(use_single_channel=False, use_reverse_channel=False)
    model.to(device)  # 将模型移动到相应设备

    # 初始化优化器和训练器
    trainer = Trainer(model, train_loader, val_loader, device, task_type='classification', epochs=50)

    # 训练模型
    train_loss_values, train_accuracy_values, val_loss_values, val_accuracy_values = trainer.train()

    # 保存当前折的模型
    torch.save(model.state_dict(), f'fold_{fold + 1}_best_trained_model.pth')

    # 在验证集上评估模型
    model.load_state_dict(torch.load(f'fold_{fold + 1}_best_trained_model.pth'))

    # 用验证集进行预测
    y_pred = []
    y_true = []

    for inputs, labels, TE in tqdm(val_loader, desc=f"Predicting fold {fold + 1}"):
        inputs, labels, TE = inputs.to(device), labels.to(device), TE.float().to(device)
        model_output = model(inputs)
        yprobs, score = model_output
        y_pred.append(score.detach().cpu().numpy().squeeze())
        y_true.append(TE.cpu().numpy().squeeze())

    # 计算 Pearson 相关系数和 R2 得分
    y_pred = np.hstack(y_pred)
    y_true = np.hstack(y_true)

    r, p_value = pearsonr(y_pred, y_true)
    r2 = r2_score(y_true, y_pred)

    print(f"Fold {fold + 1} - Pearson Correlation: {r:.3f}, R2 Score: {r2:.3f}")

    # 存储每个折的结果
    fold_results.append((r, r2))

    # 绘制当前折的散点图和拟合曲线
    sns.jointplot(x=y_true, y=y_pred, scatter_kws={'s': 0.8}, kind='reg')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.text(0.98, 0.95, f'R2 Score: {r2:.2f}\nPearson r: {r:.2f}',
             transform=plt.gca().transAxes, ha='right', va='top', fontsize=10)
    plt.title(f"Fold {fold + 1} - R2: {r2:.2f}, Pearson r: {r:.2f}")
    plt.show()

# 计算所有折的平均性能
mean_r = np.mean([fold_result[0] for fold_result in fold_results])
mean_r2 = np.mean([fold_result[1] for fold_result in fold_results])

print(f"\nAverage Pearson Correlation over all folds: {mean_r:.3f}")
print(f"Average R2 Score over all folds: {mean_r2:.3f}")