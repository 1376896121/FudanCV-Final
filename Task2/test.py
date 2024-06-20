import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import timm
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description="Vit training")
    parser.add_argument("--model_path",default="CV_final/model_checkpoints/best_model.tar", type=str)
    parser.add_argument("--batch_size",default=32,type=int)
    args = parser.parse_args()
    return args

def test(model, test_loader, device='cuda'):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    return test_loss, accuracy


if __name__ == "__main__":
    args = arg_parse()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ViT input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the CIFAR-100 test dataset
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=False)
    checkpoints = torch.load(args.model_path)
    model.load_state_dict(checkpoints["model_state_dict"])
    model = model.to('cuda')

    test_loss, test_acc = test(model, test_loader)
    print(test_loss, test_acc)
