from data_process_SeqNN import *
import pandas as pd
from SeqNN_Effblock import LegNet
from train_module import Trainer
import torch
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 读取数据并创建数据加载器
path = 'dataSet/PC3.csv'
UTR_df = pd.read_csv(path)
print(UTR_df.columns)
print(UTR_df.index)

loaders = create_dataloaders(UTR_df, seqsize=100)

for fold, (train_loader, val_loader) in enumerate(loaders):
    print(f"Fold {fold+1}:")
    # 在这里进行训练和验证
    model = LegNet(use_single_channel=False, use_reverse_channel=False)
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, train_loader, val_loader, device, task_type='classification', epochs=50)
    train_loss_values, train_accuracy_values, test_loss_values, test_accuracy_values = trainer.train()
    torch.save(model.state_dict(), 'best_trained_model.pth')
    model.load_state_dict(torch.load('best_trained_model.pth'))

    # 进行预测
    y_pred = []
    y_true = []

    for inputs, labels, TE in tqdm(val_loader, desc="Predicting on Test Set"):
        inputs, labels, TE = inputs.to(device), labels.to(device), TE.float().to(device)
        model_output = model(inputs)
        yprobs, score = model_output
        y_pred.append(score.detach().cpu().numpy().squeeze())
        y_true.append(TE.cpu().numpy().squeeze())

    y_pred = np.hstack(y_pred)
    y_true = np.hstack(y_true)

    # 计算相关性和R2得分
    r, _ = pearsonr(y_pred, y_true)
    r2 = r2_score(y_true, y_pred)

    print('Pearson Correlation Coefficient on Test Set:', r)
    print('R2 Score on Test Set:', r2)

    # 绘制散点图和拟合曲线
    sns.jointplot(x=y_true, y=y_pred, scatter_kws={'s': 0.8}, kind='reg')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.text(0.98, 0.95, f'R2 Score: {r2:.2f}\nPearson r: {r:.2f}',
             transform=plt.gca().transAxes, ha='right', va='top',
             fontsize=10)
    plt.show()