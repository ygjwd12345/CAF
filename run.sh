CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 run.py --data_root ./dataset --batch_size 10 --dataset ade --name LWF --task 100-50 --step 0 --lr 0.01 --epochs 60 --method att
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 run.py --data_root ./dataset --batch_size 6 --dataset ade --name LWF --task 100-50 --step 1 --lr 0.01 --epochs 60 --method LWF
#### for debug command
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 run.py --data_root ./dataset --batch_size 2 --dataset ade --name LWF --task 100-50 --step 0 --lr 0.01 --epochs 60 --method LWF --no_pretrained
### for small lambda

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset c --name MiB --task 50 --step 0 --lr 0.01 --epochs 60 --method MiB
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 7 --dataset ade --name MiB --task 50 --step 1 --lr 0.001 --epochs 60 --method MiB
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 7 --dataset ade --name MiB --task 50 --step 2 --lr 0.001 --epochs 60 --method MiB

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name MiB --task 19-1 --step 0 --lr 0.01 --epochs 30 --method MiB
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 7 --dataset voc --name MiB --task 19-1 --step 1 --lr 0.001 --epochs 30 --method MiB
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name MiB --task 19-1 --step 1 --lr 0.001 --epochs 30 --method MiB

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name MiB --task 19-1 --step 0 --lr 0.01 --epochs 30 --method MiB --overlap
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 7 --dataset voc --name MiB --task 19-1 --step 1 --lr 0.001 --epochs 30 --method MiB --overlap


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 8 --dataset voc --name att --task 19-1 --step 0 --lr 0.01 --epochs 30 --method att --overlap
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 19-1 --step 1 --lr 0.001 --epochs 30 --method att --overlap

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 8 --dataset ade --name att --task 100-50 --step 0 --lr 0.01 --epochs 60 --method att
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 8 --dataset ade --name att --task 100-10 --step 0 --lr 0.01 --epochs 60 --method att
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 8 --dataset ade --name att --task 50 --step 0 --lr 0.01 --epochs 60 --method att
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset ade --name att --task 50 --step 1 --lr 0.01 --epochs 60 --method att
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset ade --name att --task 50 --step 2 --lr 0.01 --epochs 60 --method att

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 8 --dataset voc --name att --task 15-5 --step 0 --lr 0.01 --epochs 30 --method att
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5 --step 1 --lr 0.001 --epochs 30 --method att
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5 --step 1 --lr 0.001 --epochs 30 --method att --overlap

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5s --step 0 --lr 0.01 --epochs 30 --method att
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5s --step 1 --lr 0.001 --epochs 30 --method att
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5s --step 2 --lr 0.001 --epochs 30 --method att
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5s --step 3 --lr 0.001 --epochs 30 --method att
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5s --step 4 --lr 0.001 --epochs 30 --method att
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5s --step 5 --lr 0.001 --epochs 30 --method att

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5 --step 1 --lr 0.001 --epochs 30 --method att

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset ade --name att --task 100-50 --step 1 --lr 0.01 --epochs 60 --method att

CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 19-1 --step 1 --lr 0.001 --epochs 30 --method att
