import matplotlib.pyplot as plt

class LossPlotter:
    @staticmethod
    def plot(train_loss_values, test_loss_values, train_accuracy_values, test_accuracy_values, epochs):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 绘制loss曲线
        ax1.plot(range(1, epochs + 1), train_loss_values, label='Train Loss', color='tab:blue')
        ax1.plot(range(1, epochs + 1), test_loss_values, label='Test Loss', color='tab:orange')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # 创建共享x轴的第二个y轴
        ax2 = ax1.twinx()
        # 绘制accuracy曲线
        ax2.plot(range(1, epochs + 1), train_accuracy_values, label='Train Accuracy', color='tab:green',
                 linestyle='dashed')
        ax2.plot(range(1, epochs + 1), test_accuracy_values, label='Test Accuracy', color='tab:red',
                 linestyle='dashed')
        ax2.set_ylabel('Accuracy', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        # 图例
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.title('Training and Test Loss/Accuracy Curve')
        plt.show()