# Fudan university Computer Vision Final Assignment
This repository include first and second tasks.
For more details, check our report pdf.

### Task1 自监督学习
### Self Supervised Learning
We Use SimCLR as SSL method, then test the performence under Linear Classification Protocol.

The network is Resnet-18, the dataset we use for SSL and LCP is CIFAR-100. 

However, the dataset can be chosen from CIFAR-10, CIFAR-100 and STL-10, and other resnet structures are supported. 

You can change the setting by modifying `run_all.sh`.

##### Result:

The best model under LCP achieve an accuracy of 55.4\%.

The best weight can be downloaded at https://pan.baidu.com/s/1ADv3W2GVLBNTJZ6Ip4DyRg?pwd=cquf

##### To Run: 
````
bash run_all.sh
````

### Task2 CNN与ViT对比
### Training Vision Transformer and CNN in CIFAR-100 to compare their performance
For ViT we use the checkpoint "vit_base_patch16_224.augreg2_in21k_ft_in1k", which can be downloaded in huggingface.

For CNN we use Resnet-152, ViT we use ViT-base, they have similar parameter amount (nearly 80M). 
Based on the fact that the two models have a similar number of parameters, we believe that this comparison is fair and reasonable.

ViT structure:

patch embedding + 11 transformer block + 1  linear layer(output the designated class numbers)

##### Results:

Resnet-152 achieve an accuracy of 69.7\%, ViT achieve an accuracy of 82.63\%.

Resnet-152 best weight can be downloaded at https://pan.baidu.com/s/1glSvK2vohn4KjIbH8eHkDw?pwd=ssoi 

ViT best weight can be downloaded at https://pan.baidu.com/s/13BUC16UUIGlaXj9wF9XrrQ?pwd=m3jn

##### To Run: 
````
python Resnet152.py
````

and

````
python train.py --model_path ckpts/pytorch_model.bin --seed 42 --epochs 15 --batch_size 32 --lr 1e-4 --alpha 1.0 
````

````
python test.py -model_path model_checkpoints/best_model.tar --batch_size 32
````

 
