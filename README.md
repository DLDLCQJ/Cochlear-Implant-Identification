# Cochlear-Impalnt-Identification
Neural Prediction of Spoken Language Improvements in Children with Cochlear Implants

# Objective 
This study aims to construct neural predictive models to forecast post-cochlear implant (CI) language improvements in local hearing-impaired children and to evaluate whether these models are language-specific or language-universal based on CI candidates learning English.

# Machine Learning and Deep Learning Models
## 1.Transfer Learning Models
The standard transfer learning strategy involves using pre-trained CNN models on ImageNet as the backbone of the model, followed by fine-tuning the top layers. Such as: AlexNet, VGG19, ResNet, Inception, GoogleNet, MobileNet, and DenseNet.
## 2. Machine Learning Models
Linear regression (LR), Support Vector Machine (SVM), Random Forest (RF), Decision Tree (DT), K-Nearest Neighbor (KNN), and eXtreme Gradient Boosting (XGBoost)

# Hardware
Platform: MAC Mini 2023
GPU: A100 GPU RAM 40.0 GB

# Requirements
PyTorch (1.9 or greater).
NumPy (1.19 or greater).
tqdm (4.31 or greater).
nibabel (3.2 or greater).
matplotlib (3.3 or greater).
scikit-learn (0.23 or greater).
scipy (1.5.4 or greater).
SimpleITK (2.3.1 or greater).
timm (1.0.3 or greater).
transformers (4.41.2 or greater).


# Model Performance and Generalization

## The classification performance of the Transfer Learning models and Machine Learning models in the Chicago English group.![image](https://github.com/DLDLCQJ/Cochlear-Impalnt-Identification/assets/145650040/79a91245-f797-4641-8d73-d71b56a7afdb)


## The performance of the Transfer Learning method within and across datasets using the MobileNet model![image](https://github.com/DLDLCQJ/Cochlear-Impalnt-Identification/assets/145650040/a6f69049-43ed-4ccc-bca0-27c4627131bb)

## Classification performance for machine learning models and transfer learning models (A) and the generalization of the model within datasets and across datasets (B).![image](https://github.com/DLDLCQJ/Cochlear-Impalnt-Identification/assets/145650040/3251c627-2591-48cf-bc8a-6454fede9eb6)

