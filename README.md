# Let Two Graphs Talk: Self-Supervised Dual-Graph Reconstruction for Multimodal Recommendation

## Introduction

This is the Pytorch implementation for our paper: Let Two Graphs Talk: Self-Supervised Dual-Graph Reconstruction for Multimodal Recommendation

## Overview

The structure of our model is available for viewing in the following:

<p align="center">
   <img src="src/fig/DuGRec.png" width="900">
</p>



### Enviroment Requirement

Python==3.8,

Pytorch==2.0.0,

Install all requirements with ``pip install -r requirements.txt``.

## Download data

Put the Baby, Clothing, and Sports datasets and other required data into the folder ``CPPRec/data`` by downloading from this link [Google Drive](https://drive.google.com/drive/folders/1BxObpWApHbGx9jCQGc8z52cV3t9_NE0f?usp=sharing).

## 3. Training on a local server using PyCharm.

Run DugRec by ``python main.py`` with the default dataset as Baby. Specific dataset selection can be modified in `main.py`.

## 4. Training on a local server using Git Bash.

Run DugRec by ``train.\`` with the default dataset is Baby. Specific dataset selection can be modified in `train.py`.



## 5. Modify specific parameters.

You may specify other parameters in CMD or config with `configs/model/*.yaml` and `configs/dataset/*.yaml`. 



## Acknowledgement


The structure of this code is  based on [MMRec](https://github.com/enoche/MMRec). Thank for their work.

