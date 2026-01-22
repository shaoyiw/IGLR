# IGLR

This is the official implementation of the paper "Interaction-Guided Attention and Local Region Refinement for Interactive Image Segmentation".

### <p align="center"> Interaction-Guided Attention and Local Region Refinement for Interactive Image Segmentation
<br>

<div align="center">
  Jianwu&nbsp;Long</a> <b>&middot;</b>
  Shaoyi&nbsp;Wang</a> <b>&middot;</b>
  Yuanqin&nbsp;Liu</a>
</div>
</br>

<div align=center><img src="assets/IGLR.png" /></div>

### Abstract

Interactive image segmentation aims to generate accurate object masks with minimal user interaction. Existing Transformer-based methods often rely on high-resolution inputs to preserve object boundaries, resulting in heavy computational overhead during both training and inference. In addition, most approaches rely on the direct concatenation of user interactions and feedback with low-level image features. Consequently, the high-level semantic information in the feedback is largely overlooked, which weakens interaction guidance. To address these limitations, we propose IGLR, an efficient interactive image segmentation framework that integrates Interaction-Guided Attention and Local Region Refinement. Interaction-Guided Attention injects user interactions and feedback directly into the self-attention mechanism, enabling more effective exploitation of interaction cues. To achieve accurate boundaries with reduced computational cost, IGLR adopts a lightweight two-stage design that first produces a coarse prediction over interaction-guided regions of interest and then refines boundary transition regions for fine-grained segmentation.Experimental results demonstrate that IGLR achieves superior performance compared with existing state-of-the-art methods on four benchmark datasets, namely GrabCut, Berkeley, SBD, and DAVIS.

### Preparations

PyTorch 1.13.0, mmcv_full 1.2.7, CUDA 11.6.

```
pip3 install -r requirements.txt
```

### Download

The datasets for training and validation can be downloaded by following: [RITM Github](https://github.com/saic-vul/ritm_interactive_segmentation)

The pre-trained models are coming soon.

### Evaluation

Before evaluation, please download the datasets and models and configure the path in configs.yml.

The following script will start validation with the default hyperparameters:

```
python scripts/evaluate_model.py NoBRS \
--gpu=0 \
--checkpoint=/home/ubuntu/liu/ISLR/experiments/ISLRformer/segformerB3_S2_cclvs/000_segformerB3_S2_cclvs_num_4/checkpoints/last_checkpoint.pth \
--eval-mode=cvpr \
--datasets=GrabCut,Berkeley,DAVIS,SBD
```

### Training

Before training, please download the pre-trained weights (click to download: [Segformer](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia)).

Use the following code to train a base model on coco+lvis dataset:

```
python train.py ./models/ISLRformer/segformerB3_S2_cclvs.py \
--batch-size=24 \
--ngpus=2
```

## Acknowledgement
Here, we thank so much for these great works:  [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) and [FocalClick](https://github.com/XavierCHEN34/ClickSEG)
