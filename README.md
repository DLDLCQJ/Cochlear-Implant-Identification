# Cochlear-Implant-Identification
Neural Prediction of Spoken Language Improvements in Children with Cochlear Implants

## Objective 
This study aims to compare the accuracy of traditional machine learning (ML) to deep transfer learning (DTL) algorithms to predict post-CI spoken language development of children with bilateral SNHL using a binary classification model of high versus low language improvers.


## Machine Learning and Deep Learning Models
### 1. Deep Transfer Learning Models
The standard transfer learning strategy involves using pre-trained CNN models on ImageNet as the backbone of the model, followed by fine-tuning the top layers. 

Examples include: 
- AlexNet
- VGG19
- ResNet
- Inception
- GoogleNet
- MobileNet
- DenseNet

### 2. Machine Learning Models
- Linear Regression (LR)
- Support Vector Machine (SVM)
- Random Forest (RF)
- Decision Tree (DT)
- K-Nearest Neighbor (KNN)
- eXtreme Gradient Boosting (XGBoost)
  
### 3. Reduction Dimensionality Techniques:
- Principal component analysis (PCA)
- Gaussian random projection (GRP)
- Recursive feature elimination (RFE)
- Univariate feature selection (UFS)

## Hardware
- Platform: MAC Mini 2023
- GPU: A100 GPU RAM 40.0 GB

## Requirements
- PyTorch (1.9 or greater)
- NumPy (1.19 or greater)
- tqdm (4.31 or greater)
- nibabel (3.2 or greater)
- matplotlib (3.3 or greater)
- scikit-learn (0.23 or greater)
- scipy (1.5.4 or greater)
- SimpleITK (2.3.1 or greater)
- timm (1.0.3 or greater)
- transformers (4.41.2 or greater)

You can install the required dependencies using pip:

```sh
pip install -r requirements.txt
```

## Citations
<!-- Add your citations here -->
<details>

