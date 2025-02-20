# AI for Human Burn Classification

## Project Overview
This project implements multiple deep learning models, including ResNeXt, VGG16, AlexNet, and BuRnGANeXt50, for burn classification.

## Installation
```bash
pip install -r requirements.txt
```

## Dataset
The dataset should be organized as:
```
data/
    train/
        class_1/
        class_2/
        class_3/
    test/
        class_1/
        class_2/
        class_3/
```

## Training the Model
```bash
python train.py
```

## Evaluating the Model
```bash
python evaluate.py
```
