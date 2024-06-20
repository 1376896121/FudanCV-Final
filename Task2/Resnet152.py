import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet152, ResNet152_Weights
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter

# 数据增强策略，包括CutMix
class CutMix:
    def __init__(self, beta=1.0, prob=0.5):
        self.beta = beta
        self.prob = prob

    def __call__(self, images, labels):
        if np.random.rand() > self.prob:
            return images, labels, labels, 1.0
        
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        
        lam = np.random.beta(self.beta, self.beta)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
        
        labels_a, labels_b = labels, labels[indices]
        return images, labels_a, labels_b, lam

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

# 数据预处理和增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 尝试加载预训练的ResNet50模型，并捕获任何错误
try:
    weights = ResNet152_Weights.IMAGENET1K_V1
    model = resnet152(weights=weights)
except Exception as e:
    print(f"Error loading ResNet50 weights: {e}")
    # 退出程序或进行其他处理
    exit(1)

# 修改最后的全连接层以适应CIFAR-100的100个类别
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)

# 使用CUDA加速（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

cutmix = CutMix()

# TensorBoard可视化
writer = SummaryWriter()

# 训练模型
for epoch in range(300):  
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        
        inputs, labels_a, labels_b, lam = cutmix(inputs, labels)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:  # 每100个mini-batch打印一次
            writer.add_scalar('training_loss', running_loss / 100, epoch * len(trainloader) + i)
            running_loss = 0.0
    
    scheduler.step()

    # 验证模型并记录准确率和测试集损失
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0  # 初始化测试集上的损失
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  # 计算损失
            test_loss += loss.item() * images.size(0)  # 累加测试集上的损失
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    test_loss /= len(testset)  # 计算平均测试集上的损失
    print(f"Epoch {epoch + 1}:")
    print(f"  Accuracy on test set: {accuracy*100:.2f}%")
    print(f"  Test loss: {test_loss:.4f}")
    writer.add_scalar('test_accuracy', accuracy, epoch)
    writer.add_scalar('test_loss', test_loss, epoch)

    model.train()

print('Finished Training')

# 保存模型
torch.save(model.state_dict(), 'resnet152_cifar100_e-2.pth')

# 关闭TensorBoard写入器
writer.close()