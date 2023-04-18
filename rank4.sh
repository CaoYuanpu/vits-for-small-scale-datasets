python3 -u vit_lora_split_reset.py  --arch vit_lora_split --dataset CIFAR100 --datapath ~/data/cifar-100-python --lr 0.004 --batch_size 256 --epochs 200 --lora_rank 4 --lora_reset 10 --gpu 3 --weight-decay 0 --warmup 1


python3 -u vit_lora_split_reset.py  --arch vit_lora_split --dataset CIFAR100 --datapath ~/data/cifar-100-python --lr 0.004 --batch_size 256 --epochs 200 --lora_rank 4 --lora_reset 250 --gpu 3 --weight-decay 0 --warmup 1