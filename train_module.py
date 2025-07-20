import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from plot_module import *
from Lion_optimizer import Lion
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class Trainer:
    def __init__(self, model, train_loader, test_loader, device, task_type='classification', epochs=2, lrate=0.0001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.task_type = task_type
        # 根据输入参数选择损失函数和优化器
        if task_type == 'classification':
            self.criterion = nn.KLDivLoss(reduction='batchmean')
            self.optimizer = Lion(model.parameters())  # Lion optimizer
        elif task_type == 'regression':
            self.criterion = nn.MSELoss()
            self.optimizer = AdamW(model.parameters(), lr=lrate)
        self.train_loss_values = []
        self.test_loss_values = []
        self.train_accuracy_values = []
        self.test_accuracy_values = []

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, 1)
        return torch.sum(preds == labels.argmax(dim=1)).item() / len(labels)

    def R2(self, outputs, bins):
        # 确保 outputs 和 bins 是同样的形状
        if outputs.shape != bins.shape:
            raise ValueError("输出和目标值的形状必须相同")
        # 计算实际值的平均值
        mean_bins = torch.mean(bins)
        # 计算总平方和（分母）
        total_variance = torch.sum((bins - mean_bins) ** 2)
        # 计算残差平方和（分子）
        residual_variance = torch.sum((bins - outputs) ** 2)
        # 计算 R2
        r2 = 1 - (residual_variance / (total_variance + 1e-8))
        return r2.item()  # 返回 R2 值

    def train_epoch(self):
        self.model.train()
        running_train_loss = 0.0
        running_train_accuracy = 0.0
        loop1 = tqdm(self.train_loader, desc="Training", leave=True)
        for inputs, labels, MRL in loop1:  # 使用tqdm显示进度条
            inputs, labels, MRL = inputs.to(self.device), labels.to(self.device), MRL.float().to(self.device)
            self.optimizer.zero_grad()
            model_outputs = self.model(inputs)
            yprobs, y = model_outputs
            if  self.task_type == 'classification':
                train_loss = self.criterion(yprobs, labels)
            elif self.task_type == 'regression':
                train_loss = self.criterion(y, MRL)
            train_loss.backward()
            self.optimizer.step()
            running_train_loss += train_loss.item()
            if self.task_type == 'classification':
                running_train_accuracy += self.accuracy(yprobs, labels)
            elif self.task_type == 'regression':
                running_train_accuracy += self.R2(y, MRL)
        return running_train_loss / len(self.train_loader), running_train_accuracy / len(self.train_loader)

    def test_epoch(self):
        self.model.eval()
        running_test_loss = 0.0
        running_test_accuracy = 0.0
        with torch.no_grad():
            loop2 = tqdm(self.test_loader, desc="Testing", leave=True)
            for inputs, labels, MRL in loop2:  # 使用tqdm显示进度条
                inputs, labels, MRL = inputs.to(self.device), labels.to(self.device), MRL.float().to(self.device)
                model_output = self.model(inputs)
                yprobs, score = model_output
                test_loss = self.criterion(yprobs, labels)
                running_test_loss += test_loss.item()
                if  self.task_type == 'classification':
                    running_test_accuracy += self.accuracy(yprobs, labels)
                elif self.task_type == 'regression':
                    running_test_accuracy += self.R2(score, MRL)
        return running_test_loss / len(self.test_loader), running_test_accuracy / len(self.test_loader)

    def train(self):
        early_stopping = EarlyStopping(patience=100, delta=0.001)
        for epoch in range(self.epochs):
            avg_train_loss, avg_train_accuracy = self.train_epoch()
            avg_test_loss, avg_test_accuracy = self.test_epoch()
            self.train_loss_values.append(avg_train_loss)
            self.test_loss_values.append(avg_test_loss)
            self.train_accuracy_values.append(avg_train_accuracy)
            self.test_accuracy_values.append(avg_test_accuracy)

            print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}')

            # 调用早停法
            early_stopping(avg_test_loss, self.model)

            if early_stopping.early_stop:
                print("早停法触发，终止训练")
                break

        print('Finished Training')
        # 保存最佳模型参数
        torch.save(self.model.state_dict(), 'best_trained_model.pth')
        LossPlotter.plot(self.train_loss_values, self.test_loss_values,
                         self.train_accuracy_values, self.test_accuracy_values,
                         epoch + 1)
        return self.train_loss_values, self.train_accuracy_values, self.test_loss_values, self.test_accuracy_values
