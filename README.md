# FATMEF
First, you need to prepare the Swintranformer.
Information about Swintranformer can be found in "https://github.com/microsoft/Swin-Transformer"

You need to place the pre trained model "swin_base_patch4_window7_224_22k.pth" in the checkpoint folder

Then, you can train with

python train.py --samplelist path_SCIE --root path_COCO --batch_size 1 --miniset true --minirate 0.1 --epoch 20  --w_crloss 0.2 --w_ssimloss 0 --w_mseloss 0 --w_l1loss 0 --w_tvloss 20 --img_size 256 --crloss True

test with

python finalfusion.py --model_path path_model --ue_path under-exposed/ --oe_path over-exposed/ --save_path ./MEFB_clif_/ --resize True --method T
