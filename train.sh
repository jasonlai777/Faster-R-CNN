CUDA_VISIBLE_DEVICES=2 python3 -u trainval_net.py --dataset pascal_voc --net res101 --disp_interval 100 --bs 8 --nw 1 --lr 0.001 --lr_decay_step 5 --cuda
