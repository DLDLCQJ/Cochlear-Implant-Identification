# Cochlear-Implant-Identification
Neural Prediction of Spoken Language Improvements in Children with Cochlear Implants

## Objective 
This study aims to construct neural predictive models to forecast post-cochlear implant (CI) language improvements in local hearing-impaired children and to evaluate whether these models are language-specific or language-universal based on CI candidates learning English.

## Machine Learning and Deep Learning Models
### 1. Transfer Learning Models
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

## Models Comparisons
![Figure2](https://github.com/DLDLCQJ/Cochlear-Implant-Identification/assets/145650040/1405a323-0ba7-4155-9e1a-635adff244ec)
Figure: Classification performance for machine learning models and transfer learning models (A) and the generalization of the model within datasets and across datasets (B).

</details>
<details>
<summary> Table 2. The classification performance of the Transfer Learning models and Machine Learning models in the Chicago English group.</summary>

| Types       | Models          | Accuracy (95% CI)         | Sensitivity (95% CI)      | Specificity (95% CI)      | AUC (95% CI)            |
|-------------|------------------|---------------------------|---------------------------|---------------------------|-------------------------|
| Slice-based | VGG19_bn         | 81.17 (80.11-82.22)       | 86.19 (84.80-87.57)       | 75.73 (73.55-77.90)       | 0.810 (0.799-0.820)     |
| Slice-based | ResNet-50d       | 88.02 (86.92-89.11)       | 88.16 (85.98-90.34)       | 87.86 (86.21-89.51)       | 0.880 (0.869-0.891)     |
| Slice-based | DenseNet_169     | 89.09 (88.06-90.12)       | 92.11 (91.47-92.74)       | 85.83 (83.64-88.02)       | 0.890 (0.879-0.900)     |
| Slice-based | AlexNet          | 79.95 (78.61-81.30)       | 84.13 (82.67-85.58)       | 75.44 (72.53-78.35)       | 0.800 (0.786-0.813)     |
| Slice-based | Inception_V3     | 83.64 (81.75-85.53)       | 85.65 (77.40-93.90)       | 81.46 (73.24-89.67)       | 0.836 (0.817-0.854)     |
| Slice-based | GoogleNet        | 87.13 (85.54-88.72)       | 92.38 (90.53-94.22)       | 81.46 (79.07-83.84)       | 0.869 (0.853-0.885)     |
| Slice-based | MobileNet        | 89.74 (89.39-90.10)       | 87.09 (86.17-88.00)       | 92.20 (90.98-93.42)       | 0.896 (0.893-0.900)     |
| Voxel-based | LR               | 58.74 (47.71-69.77)       | 52.89 (41.88-63.91)       | 63.51 (31.67-95.34)       | 0.582 (0.432-0.732)     |
| Voxel-based | DT               | 55.30 (37.53-73.07)       | 74.65 (53.49-95.81)       | 38.43 (9.40-67.46)        | 0.565 (0.477-0.654)     |
| Voxel-based | SVM              | 49.73 (40.55-58.91)       | 36.67 (8.26-65.08)        | 63.40 (34.43-92.37)       | 0.500 (0.414-0.586)     |
| Voxel-based | KNN              | 50.37 (43.68-57.06)       | 53.25 (28.96-77.55)       | 47.54 (22.54-72.54)       | 0.504 (0.431-0.577)     |
| Voxel-based | RF               | 48.45 (31.79-65.11)       | 36.38 (15.65-57.12)       | 66.13 (35.02-97.25)       | 0.512 (0.364-0.661)     |
| Voxel-based | XGBoost          | 53.25 (42.39-64.12)       | 53.86 (42.30-65.43)       | 53.07 (34.47-71.66)       | 0.5347 (41.35-65.58)    |

</details>
<details>
<summary> Table 3. The performance of the Transfer Learning method within and across datasets using the MobileNet model.</summary>

| Datasets                           | Accuracy (95% CI)       | Sensitivity (95% CI)     | Specificity (95% CI)     | AUC (95% CI)            |
|------------------------------------|-------------------------|--------------------------|--------------------------|-------------------------|
| Single Dataset                     |                         |                          |                          |                         |
| Chicago_English                    | 89.74 (89.39-90.10)     | 87.09 (86.17-88.00)      | 92.20 (90.98-93.42)      | 0.896 (0.893-0.900)     |
| Melbourne_English                  | 91.03 (90.60-91.46)     | 91.67 (90.63-92.70)      | 90.41 (89.09-91.72)      | 0.910 (0.906-0.915)     |
| Chicago_Spanish                    | 85.41 (70.96-99.85)     | 89.02 (87.69-90.35)      | 82.33 (54.97-99.96)      | 0.857 (0.724-0.990)     |
| Across Center                      |                         |                          |                          |                         |
| Melbourne_English (independent)    | 50.95 (49.14-52.75)     | 62.90 (3.74-100)         | 39.28 (0-95.66)          | 0.511 (0.489-0.533)     |
| Across Language                    |                         |                          |                          |                         |
| Chicago_Spanish (independent)      | 50.27 (46.78-53.76)     | 36.89 (0-93.88)          | 63.95 (6.43-100)         | 0.499 (0.467-0.532)     |
| Across Center & Language           |                         |                          |                          |                         |
| Hong Kong_Chinese (independent)    | 50.75 (47.62-53.87)     | 36.67 (0-96.18)          | 63.26 (3.46-100)         | 0.500 (0.496-0.504)     |
| Combined Dataset                   |                         |                          |                          |                         |
| Chicago+Melbourne                  | 87.38 (87.12-87.64)     | 85.36 (84.02-86.70)      | 89.57 (88.04-91.11)      | 0.874 (0.871-0.876)     |
| Chicago+Melbourne+HK               | 87.94 (87.28-88.59)     | 88.33 (87.18-89.48)      | 87.56 (86.12-89.00)      | 0.879 (0.873-0.886)     |

</details>
<details>

## Citations
<!-- Add your citations here -->
<details>

