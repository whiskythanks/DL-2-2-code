import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from MyDataset import *
from Unet_Plus import *

MyData = Train_Dataset()

train_size = int(0.8 * len(MyData))
test_size = len(MyData) - train_size
train_data, test_data = torch.utils.data.random_split(MyData, [train_size, test_size])

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)

# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=5)
test_dataloader = DataLoader(test_data, batch_size=5)

num_classes = 2
model = unetpluses(num_classes)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler1 = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("------------第 {} 轮训练开始------------".format(i + 1))
    # 训练步骤开始
    model.train()  # 这两个层，只对一部分层起作用，比如 dropout层；如果有这些特殊的层，才需要调用这个语句
    for data in train_dataloader:
        imgs, targets = data
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 优化器，梯度清零
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))  # 这里用到的 item()方法，有说法的，其实加不加都行，就是输出的形式不一样而已

# 测试步骤开始
    model.eval()  # 这两个层，只对一部分层起作用，比如 dropout层；如果有这些特殊的层，才需要调用这个语句
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 这样后面就没有梯度了，  测试的过程中，不需要更新参数，所以不需要梯度？
        for data in test_dataloader:  # 在测试集中，选取数据
            imgs, targets = data
            outputs = model(imgs)  # 分类的问题，是可以这样的，用一个output进行绘制
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()  # 为了查看总体数据上的 loss，创建的 total_test_loss，初始值是0

    print("整体测试集上的Loss: {}".format(total_test_loss))
