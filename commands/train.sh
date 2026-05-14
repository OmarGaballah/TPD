# VITONHD_phase1_crossattn_captions
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--logdir train_logs/VITONHD/ \
--pretrained_model checkpoints/release/TPD_240epochs.ckpt \
--base configs/train/train_VITONHD.yaml \
--scale_lr False \
--name VITONHD_phase1_crossattn_captions