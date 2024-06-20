### Task1 自监督学习

SimCLR用于Resnet，CIFAR-100。
CIFAR-100学习到的表征在LCP协议下达到55.4\%。

##### To Run: 
bash run_all.sh

### Task2 CNN与ViT对比

CNN采用Resnet-152, ViT采用ViT-base, 参数量相近。
Resnet-152达到69.7\%, ViT达到82.63\%.

##### To Run: 
python Resnet152.py
python test.py
