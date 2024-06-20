# activate pip / conda environment first

Train SimCLR model
python main.py --dataset CIFAR100 --model_path save_CIFAR100_sr.5 --sliced_rate 0.5

Train linear model and run test
nohup python linear_evaluation.py --dataset CIFAR100 --model_path save_CIFAR100_sr.5 \
> save_CIFAR100_sr.5/output.log 2>&1 &



# python -m testing.logistic_regression \
#     with \
#     model_path=./logs/0