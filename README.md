# Investigation-of-the-internal-behaviour-of-a-CNN
Understanding, why a CNN makes decisions

# Motivation & Expactation
Most Fashion-MNIST projects stop after reporting accuracy values. While metrics such as Accuracy, Loss and Confusion Matrices indicate model performance, they do not explain why a model arrives at a specific prediction.

This project investigates the internal behavior of a CNN trained on Fashion-MNIST using Explainable AI techniques.

# Research Questions

1.) Which image regions does the CNN focus on?

2.) Why do certain classes generate more errors than others?

3.) How are clothing categories represented inside the learned feature space?


# Test environment
## Hardware
  * CPU: Intel Core i3-12100F (4 cores / 8 threads)
  * GPU: Nvidia RTX 3050 - 6GB VRAM
  * RAM: 16 GB (dual-channel)
## Software
  * OS: Ubuntu 22.04.05 LTS (64bit)
  * Visual Studio Code
  * Python 3.11.15
  * TensorFlow incl. CUDA
  * Scikit-Learn
  * Pandas
  * Seaborn & Matplotlib

# Dataset

## Fashion-MNIST
  * 60,000 training images
  * 10,000 test images
  * 10 classes
  * Grayscale images
  * Resolution: 28 × 28 pixels

## Classes
  * T-Shirt/top
  * Trouser
  * Pullover
  * Dress
  * Coat
  * Sandal
  * Shirt
  * Sneaker
  * Bag
  * Ankle Boot

## CNN architecture


  * Input (28x28x1)
  * Conv2D(32)
  * MaxPooling
  * Conv2D(64)
  * MaxPooling
  * Flatten
  * Dense(128)
  * Dropout(0.3)
  * Dense(10 Softmax)


## Explainability Pipeline

1.) Grad-CAM Visualizations

2.) Failure Analysis

3.) Confusion Matrix Diagnostics

4.) t-SNE Feature Space Visualization


# Results & Evaluation
## 1. Grad-CAM Analysis

**What does the CNN actually focus on?**

<img width="1200" height="600" alt="gradcam_examples" src="https://github.com/user-attachments/assets/a2c4bd36-4c98-4b90-ad82-3a42baaba0c5" />

Grad-CAM visualizations highlight image regions that contribute most strongly to a prediction.

**Observations**
**Trouser**

The CNN focuses almost exclusively on the vertical leg structures.

**Ankle Boot**

The strongest activations appear around the sole and lower boot contour.

**Shirt**

Attention becomes noticeably more diffuse and distributed across the garment.

This suggests that the network lacks a unique feature prototype for shirts.

**Key Insight**

Classes with highly localized attention patterns tend to achieve substantially higher classification accuracy

## 2. Failure Analysis
**Which mistakes does the CNN make?**

<img width="1200" height="600" alt="failure_analysis_grid" src="https://github.com/user-attachments/assets/a0386975-5561-41bd-bcf4-a1d26c6c0799" />

Most classification errors occur between visually similar upper-body garments.

Recurring error patterns include:

    * Shirt → T-shirt/top
    * Shirt → Pullover
    * Shirt → Coat
    * Coat → Pullover
    * Dress → T-shirt/top

Interestingly, many of these samples are difficult even for human observers due to the low image resolution.
**Key Insight**

The CNN does not make random mistakes. Most errors occur between categories with genuine visual overlap.

## 3. Confusion Matrix Analysis
**Where are the error hotspots?**

<img width="1000" height="800" alt="failure_confusion_matrix" src="https://github.com/user-attachments/assets/5a7ca221-9c34-4c24-baa9-faef8e24ac75" />

**Strongest Classes**

| Class      | correct predictions |
|------------|---------------------|
| Bag        | 989                 |
| Trouser    | 984                 |
| Sneaker    | 983                 |
| Sandal     | 970                 |
| Ankle Boot | 959                 |

These classes possess highly distinctive shapes and therefore form strong class boundaries.

**Most difficult Class**

**Shirt**

Only 655 of 1000 shirt images were classified correctly.

Largest confusions:
| predicted class | count |
|-----------------|-------|
| T-shirt/top     | 152   |
| Pullover        | 92    |
| Coat            | 62    |

**Key Insights**
The majority of remaining errors originate from a small group of highly similar garment categories.


## 4. Feature Space Visualization
**How does the CNN represent clothing internally?**

<img width="1000" height="800" alt="tsne_feature_space" src="https://github.com/user-attachments/assets/50ff5824-15d0-414f-a278-57cb887a6cb4" />

To investigate internal representations, feature vectors from the final dense layer were projected into two dimensions using:

    * PCA (50 components)
    * t-SNE
    
**Well-separated clusters**
Several classes form compact and isolated clusters:

    * Trouser
    * Bag
    * Sneaker
    * Sandal
    * Ankle Boot

These categories also achieve the highest classification accuracy.

**Overlapping Classes**

Significant overlap occurs between:

    * Shirt
    * Pullover
    * Coat
    * Dress

This feature-space overlap directly mirrors the confusion matrix.

**Key Insight**

Classes with overlapping latent representations produce the majority of model errors.

## Connecting all findings

All explainability methods point toward the same conclusion:

| method           | observation                                     |
|------------------|-------------------------------------------------|
| Grad-CAM         | Diffuse attention for Shirts                    |
| Failure Analysis | Most errors involve upper-body garments         |
| t-SNE            | Strong overlap between Shirt, Coat and Pullover |

The results consistently indicate that remaining errors are caused primarily by overlapping feature representations rather than optimization failures.

# Comprehensive Interpretation & Insights
**1. The CNN learned meaningful visual representations**

Grad-CAM demonstrates that the model focuses on semantically relevant image regions such as shoe soles, trouser legs and bag outlines.

**2. Visual similarity drives classification difficulty**

Most errors occur between garment categories sharing similar silhouettes and structure.

**3. t-SNE explains the confusion matrix**

Classes with compact feature clusters achieve high classification accuracy, while overlapping clusters generate most misclassifications.

**4. Explainability reveals more than accuracy alone**

Traditional metrics only measure performance.

Explainability techniques reveal:

    * why a prediction was made,
    * where mistakes originate,
    * how classes are internally represented.

# Conclusion
The combination of Grad-CAM, Failure Analysis, Confusion Matrix diagnostics and t-SNE visualization provides a comprehensive view of the CNN's decision-making process.
While the model achieves strong classification performance, Explainable AI reveals that most remaining errors originate from overlapping feature representations among visually similar garment classes.
The results suggest that the primary limitation lies in the intrinsic ambiguity of the Fashion-MNIST dataset rather than insufficient model quality. Understanding why a model fails is often more valuable than improving accuracy by another percentage point.

# Future Work
    * UMAP vs. t-SNE comparison
    * Explainability analysis of Transfer Learning models
    * Comparison of multiple CNN architectures
    * Class-specific Grad-CAM statistics
    * Feature-space evolution across training epochs
    * Explainability analysis on custom datasets
