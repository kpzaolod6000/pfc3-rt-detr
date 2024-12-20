train : CUDA_VISIBLE_DEVICES=0 torchrun --master_port=9909 --nproc_per_node=1 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml  --use-amp --seed=0 &> log.txt 2>&1 &
tunning: CUDA_VISIBLE_DEVICES=0 torchrun --master_port=9909 --nproc_per_node=1 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -t rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --use-amp --seed=0 &> log.txt 2>&1 &
test: CUDA_VISIBLE_DEVICES=0 torchrun --master_port=9909 --nproc_per_node=1 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r checkpoint --test-only
inference: python rtdetrv2_torch.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r output/rtdetrv2_r18vd_120e_coco/checkpoint0007.pth -f carfoogy.png -d cuda


BiFPN
train : CUDA_VISIBLE_DEVICES=0 torchrun --master_port=9909 --nproc_per_node=1 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml  --use-amp --seed=0 &> log.txt 2>&1 &
tunning: CUDA_VISIBLE_DEVICES=0 torchrun --master_port=9909 --nproc_per_node=1 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -t rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --use-amp --seed=0 &> log_BIFPN.txt 2>&1 &
test: CUDA_VISIBLE_DEVICES=0 torchrun --master_port=9909 --nproc_per_node=1 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --test-only
inference: python rtdetrv2_torch.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r output/rtdetrv2_r18vd_120e_coco/checkpoint0007.pth -f carfoogy.png -d cuda


commands with tensorRT
python references/deploy/rtdetrv2_tensorrt.py --trt-file=rtdetr_r18_static_fp16_model_1.trt --im-file=gun1.png
python tools/export_onnx.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r 
best.pth --check