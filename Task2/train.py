import os
import torch
import random
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR100
import timm
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed for reproducibility


def cutmix(data, target, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    
    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.7)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size(-1) * data.size(-2)))
    
    return data, target, shuffled_target, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



def train_one_epoch(model, loader, optimizer, criterion, device='cuda',alpha=1.0):
    model.train()
    total_loss = 0
    for i,(images, targets) in enumerate(loader):
        # if i % 50 == 0:
            # import pdb; pdb.set_trace()
        images, targets = images.to(device), targets.to(device)
        images, targets1, targets2, lam = cutmix(images, targets, alpha=alpha)
        
        outputs = model(images)
        loss = lam * criterion(outputs, targets1) + (1 - lam) * criterion(outputs, targets2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)
def validate(model, val_loader, criterion, device='cuda'):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss /= len(val_loader)
    accuracy = 100. * correct / total
    return val_loss, accuracy
# Training loop

def train(model, num_epochs, train_loader, optimizer, criterion, save_dir="model_checkpoints", alpha=1.0):
    best_val_accuracy = 0.
    writer = SummaryWriter(log_dir="/home/xcm/CV_final/logs/task2")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,alpha)
        val_loss, val_accuracy = validate(model, val_loader, criterion)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        writer.add_scalar("Loss/train",train_loss,epoch)
        writer.add_scalar("Loss/val",val_loss,epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_path = os.path.join(save_dir, 'best_model.tar')
            torch.save({
                'global_step': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
            }, save_path)
            print(f'Saved best model checkpoint at {epoch}-th epoch with validation accuracy {val_accuracy:.2f}')
    writer.close()


def arg_parse():
    parser = argparse.ArgumentParser(description="Vit training")
    parser.add_argument("--model_path",default="ckpts/pytorch_model.bin", type=str)
    parser.add_argument("--seed",default=42, type=int)
    parser.add_argument("--epochs",default=15, type=int,help="training epochs")
    parser.add_argument("--batch_size",default=32, type=int,help="training batch size")
    parser.add_argument("--lr",default=1e-4, type=int,help="training learning rate")
    parser.add_argument("--alpha",default=1.0,type=int, help="alpha for cutmix")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    seed = args.seed
    set_seed(seed)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=False, checkpoint_path=args.model_path,num_classesa=100)
    # model.load_state_dict(torch.load())
    model = model.to('cuda')
    total_params = sum(p.numel() for p in model.parameters())

    print(f'Total parameters: {total_params}')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(model, args.epochs, train_loader, optimizer, criterion, args.alpha)
    