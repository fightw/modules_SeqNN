### 读取5UTR，one-hot编码序列，并划分训练集和测试集
#%%
from data_process_SeqNN import *
import pandas as pd
from SeqNN_Effblock import LegNet
path = 'clstrcompare/50similarity/rondom5121-2.csv'
UTR_df = pd.read_csv(path)
print(UTR_df)
train_loader, test_loader = create_dataloaders(UTR_df, seqsize=100)


from SeqNN_Effblock import *
model = LegNet(use_single_channel=False, use_reverse_channel=False)
print(model)

from train_module import Trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = Trainer(model, train_loader, test_loader, device, task_type='classification',epochs=50)
train_loss_values, train_accuracy_values, test_loss_values, test_accuracy_values = trainer.train()

torch.save(model.state_dict(), 'best_trained_model.pth')
model.load_state_dict(torch.load('best_trained_model.pth'))


from tqdm import tqdm
from scipy.stats import pearsonr


from scipy.stats import spearmanr

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
# Calculate the R2 score for the test set and plot the R2 graph
# 将模型应用于测试集
y_pred = []
y_true = []

for inputs, labels, TE in tqdm(test_loader, desc="Predicting on Test Set"):
    # 将输入数据移动到同样的设备上
    inputs, labels, TE = inputs.to(device), labels.to(device), TE.float().to(device)
    model_output = model(inputs)
    yprobs, score = model_output
    # 将输出数据移回到CPU上，并转换为numpy数组
    y_pred.append(score.detach().cpu().numpy().squeeze())
    y_true.append(TE.cpu().numpy().squeeze())

# 将列表转换为NumPy数组
y_pred = np.hstack(y_pred)
y_true = np.hstack(y_true)

# 计算Pearson相关系数和R2得分
r, p_value = pearsonr(y_pred, y_true)

# r, p_value = spearmanr(y_pred, y_true)#获取相关性和P值

r2 = r2_score(y_true, y_pred)  # 计算R2得分

# r2 = r**2

print('Pearson Correlation Coefficient on Test Set:', r)
print('R2 Score on Test Set:', r2)

# 绘制散点图和拟合曲线
sns.jointplot(x=y_true, y=y_pred, scatter_kws={'s': 0.8}, kind='reg')
plt.xlabel('True Values')
plt.ylabel('Predictions')
# 设置标题位置和字体大小
plt.text(0.98, 0.95, f'R2 Score: {r2:.2f}\nPearson r: {r:.2f}',
         transform=plt.gca().transAxes, ha='right', va='top',
         fontsize=10)

# 设置标题字体大小
mpl.rcParams['axes.titlesize'] = 12

plt.show()