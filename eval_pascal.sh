python ./exp/test/eval_show_pascal.py \
 --batch 1 --gpus 1 --classes 7 \
 --gt_path './data/datasets/pascal/SegmentationPart/' \
 --txt_file './data/datasets/pascal/list/val_id.txt' \
 --loadmodel './cihp2pascal.pth'
