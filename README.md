# FATMEF
This is a PyTorch/GPU implementation of the manuscript: Frequency Domain Attention Enhanced Multi-Exposure Image Fusion with FATMEF:
# pre-trained Swin-Transformer
Information about Swin-Transformer can be found in "https://github.com/microsoft/Swin-Transformer".The model Swin-B trained by ImageNet-22K with a resolution of 224 can be found.

You need to place the pre trained model "swin_base_patch4_window7_224_22k.pth" in the checkpoint folder

# Dataset
We use [MS-COCO](https://docs.voxel51.com/user_guide/dataset_zoo/datasets.html#dataset-zoo-coco-2017) for traning.

# For training
Then, you can train with
```
python train.py --samplelist path_SCIE --root path_COCO --batch_size 32 --miniset true --minirate 1 --epoch 20  --w_crloss 0.2 --w_ssimloss 0 --w_mseloss 0 --w_l1loss 0 --w_tvloss 20 --img_size 256 --crloss True
```
test with
```
python finalfusion.py --model_path path_model --ue_path under-exposed/ --oe_path over-exposed/ --save_path ./MEFB_clif_/ --resize True --method T
```
