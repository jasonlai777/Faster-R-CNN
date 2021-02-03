CUDA_VISIBLE_DEVICES=1 python3 -u trainval_net.py --dataset pascal_voc --net res101 --disp_interval 10 --bs 2 --nw 1 --lr 0.001 --lr_decay_step 5 --cuda
