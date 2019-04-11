# Graphonomy: Universal Human Parsing via Graph Transfer Learning

This repository contains the code for the paper:

[**Graphonomy: Universal Human Parsing via Graph Transfer Learning**](https://arxiv.org/abs/1904.04536)
,Ke Gong, Yiming Gao, Xiaodan Liang, Xiaohui Shen, Meng Wang, Liang Lin.

# Environment and installation
+ Pytorch = 0.4.0
+ torchvision
+ scipy
+ tensorboardX
+ numpy
+ opencv-python
+ matplotlib
+ networkx

   you can install above package by using `pip install -r requirements.txt`

# Getting Started
### Data Preparation
+ You need to download the human parsing dataset, prepare the images and store in `/data/datasets/dataset_name/`.
We recommend to symlink the path to the dataets to `/data/dataset/` as follows

```
# symlink the Pascal-Person-Part dataset for example
ln -s /path_to_Pascal_Person_Part/* data/datasets/pascal/
```
+ The file structure should look like:
```
/Graphonomy
  /data
    /datasets
      /pascal
        /JPEGImages
        /list
        /SegmentationPart
      /CIHP_4w
        /Images
        /lists
        ...  
```

### Inference
We provide a simply script to get the visualization result on the CIHP dataset using [trained](https://drive.google.com/file/d/1O9YD4kHgs3w2DUcWxtHiEFyWjCBeS_Vc/view?usp=sharing)
 models as follows :
```shell
# Example of inference
python exp/inference/inference.py  \
--loadmodel /path_to_inference_model \
--img_path ./img/messi.jpg \
--output_path ./img/ \
--output_name /output_file_name
``` 

### Training
#### Transfer learning
1. Download the Pascal pretrained model(avaliable soon).
2. Run the `sh train_transfer_cihp.sh`.
3. The results and models are saved in exp/transfer/run/.
4. Evaluation and visualization script is eval_cihp.sh. You only need to change the attribute of `--loadmodel` before you run it.

#### Universal training
1. Download the [pretrained](https://drive.google.com/file/d/18WiffKnxaJo50sCC9zroNyHjcnTxGCbk/view?usp=sharing) model and store in /data/pretrained_model/.
2. Run the `sh train_universal.sh`.
3. The results and models are saved in exp/universal/run/.

### Testing 
If you want to evaluate the performance of a pre-trained model on PASCAL-Person-Part or CIHP val/test set, 
simply run the script: `sh eval_cihp/pascal.sh`.
Specify the specific model. And we provide the final model that you can download and store it in /data/pretrained_model/.

### Models
**Pascal-Person-Part trained model**

|Model|Google Cloud|Baidu Yun|
|--------|--------------|-----------|
|Graphonomy(CIHP)| [Download](https://drive.google.com/file/d/1cwEhlYEzC7jIShENNLnbmcBR0SNlZDE6/view?usp=sharing)| Avaliable soon|

**CIHP trained model**

|Model|Google Cloud|Baidu Yun|
|--------|--------------|-----------|
|Graphonomy(PASCAL)| [Download](https://drive.google.com/file/d/1O9YD4kHgs3w2DUcWxtHiEFyWjCBeS_Vc/view?usp=sharing)| Avaliable soon|

**Universal trained model**

|Model|Google Cloud|Baidu Yun|
|--------|--------------|-----------|
|Universal|Avaliable soon|Avaliable soon|

### Todo:
- [ ] release pretrained and trained models
- [ ] update universal eval code&script

# Citation

```
@inproceedings{Gong2019Graphonomy,
author = {Ke Gong and Yiming Gao and Xiaodan Liang and Xiaohui Shen and Meng Wang and Liang Lin},
title = {Graphonomy: Universal Human Parsing via Graph Transfer Learning},
booktitle = {CVPR},
year = {2019},
}

```

# Contact
if you have any questions about this repo, please feel free to contact 
[gaoym9@mail2.sysu.edu.cn](mailto:gaoym9@mail2.sysu.edu.cn).

