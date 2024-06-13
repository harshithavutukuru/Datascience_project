# Datascience_project
Brain Tumor detection using CNN


# Project Goals
The objective of our project, CerebroVision, is to leverage Convolutional Neural Networks (CNN), a powerful form of deep learning, to improve the accuracy of early brain tumor detection and diagnosis. By using advanced algorithms to analyze MRI scans, our system aims to enhance the clarity and speed of diagnosis, potentially saving lives through earlier intervention.

Some of the specific goals that we have focused on throughout our project implementation are as given below -

High Diagnostic Accuracy: Achieve a high level of accuracy in detecting the presence of brain tumors and classifying their types. To determine the most effective model based on ROC curves and AUC metrics.

Early Detection: Identify tumors at early stages when they might be missed by traditional diagnostic methods.

Reduce False Negatives and Positives: Minimize false negatives to ensure patients receive timely treatment and reduce false positives to avoid unnecessary stress and procedures.

# DATASET DETAILS:
The dataset used in this project is the "Brain Tumor Dataset" from Kaggle, available at [this link](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset/data?select=Brain+Tumor+Data+Set). It includes 4,602 MRI images categorized into two classes: brain tumor, consisting of 2513 images and no tumor, consisting of 2087 images. For simplicity, we have reclassified these into two classes: brain tumor and healthy. Each image is provided in JPEG format with varying dimensions.

# Development tools

1. Python: Programming language used.
2. TensorFlow and Keras: For building and training CNN models.
3. Scikit-learn: Used for RES and Random Forest implementation and metrics calculation.
4. Matplotlib and Seaborn: For plotting ROC curves and other visualizations.
5. Weights & Biases (wandb): Integrated for tracking experiments and model performance

# Methodology

This project aims to evaluate and compare the performance of multiple deep learning models in classifying brain tumors from MRI images. The models used include ResNet50, VGG19, and a custom model (referred to as HALNet1, HALNet2, SMVNet, and JAPNet for different iterations and setups).

# Data Loading and Preprocessing

1. The MRI images are sourced from a specified directory using TensorFlow's image_dataset_from_directory function. This function automatically loads images and labels based on folder structure, significantly simplifying data handling.
2. Validation Split: The dataset is split into training and validation subsets with 20% of the data reserved for validation. This split is controlled by the validation_split parameter, ensuring that the model can be evaluated on unseen data. The train- 
   test split, loss and optimizers are the same across all the models used in this project.
3. Image Resizing and Normalization: Each image is resized to 224x224 pixels to match the input size expected by the pretrained models used later. Additionally, pixel values are rescaled to a range of 0 to 1 (from the original 0 to 255) to facilitate       model training, as this normalization helps in speeding up the convergence and improves training dynamics.
4. Augmentation: To increase the dataset's diversity, we applied image augmentation techniques including rotation and flipping in models like HALNet2, SMVnet and JAPNet.

# Models Setup and Configuration

1. Pretrained Models (ResNet50 and VGG19): 
    - Both models are loaded with pretrained weights from ImageNet, excluding their top layers (include_top=False), to leverage learned features while allowing customization for the binary classification task.
    - The output of these pretrained models is pooled using average pooling to reduce the dimensionality, preparing the output for the final classification layer.
2. Custom Models (HALNet1, HALNet2, SMVNet, JAPNet): 
    - A simple convolutional neural network is built from scratch, starting with convolutional layers followed by max pooling layers, which help in extracting key features from the images while reducing their spatial dimensions.
3. The network includes batch normalization to standardize activations from the previous layer, improving training stability.
4. Dropout is included to prevent overfitting by randomly omitting a subset of features during training.
5. Output Layer:
   - For all models, a dense output layer with a sigmoid activation function is used. This setup is typical for binary classification, where the output is a probability indicating the presence of a tumor.


