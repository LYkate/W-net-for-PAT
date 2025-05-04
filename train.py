from PATmodel import *
from data import *
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

data = SummaryWriter()

def train_net(net, device, train_input_root,train_lable_root, epoch=150, batch_size=20, lr=0.0001):
    # 加载训练集
    isbi_dataset_train = MyDataset(train_input_root,train_lable_root)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset_train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=30,
                                               pin_memory=True)

    # 定义RMSprop算法
    optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)#权重衰减, weight_decay=1e-6
    # 创建一个学习率调度器
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25, 100], gamma=0.1, last_epoch=-1)
    # 定义Loss算法
    criterion = nn.MSELoss()
    criterion = criterion.to(device=device)
    step=0
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epoch):
        # 训练模式
        net.train()
        lr = scheduler.get_last_lr()
        print("学习率：", lr)
        # 按照batch_size开始训练
        for signal, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            with  torch.no_grad():
                signal = signal.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(signal)
            # 计算loss
            loss = criterion(pred, label)
            print('Loss/train-', epoch, ':', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
            step=step+1
            data.add_scalar('Loss_Train',loss,step)
        scheduler.step()

#输入与标签图片所在的目录
train_input_root = 'database/input'
train_lable_root = 'database/label'

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载网络，图片单通道1，分类为255。
    net = PATModel(in_channels=1, out_channels=1)
    # 将网络拷贝到deivce中
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)  # 包装为并行风格模型
    net.to(device=device)
    # 指定训练集地址，开始训练
    train_net(net, device, train_input_root,train_lable_root)
