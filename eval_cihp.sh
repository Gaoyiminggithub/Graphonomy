python ./exp/test/eval_show_pascal2cihp.py \
 --batch 1 --gpus 1 --classes 20 \
 --gt_path './data/datasets/CIHP_4w/Category_ids' \
 --txt_file './data/datasets/CIHP_4w/lists/val_id.txt' \
 --loadmodel './data/pretrained_model/inference.pth'
