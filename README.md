# BotanicVision: CNN-Driven Rice Leaf Disease Detection Using DenseNet201

Welcome to BotanicVision, a state-of-the-art project aimed at identifying and classifying rice leaf diseases using convolutional neural networks (CNNs). This project leverages the DenseNet201 architecture for efficient and accurate disease detection to assist farmers and researchers in improving crop health management.
Table of Contents

    Introduction
    Features
    Technologies Used
    Dataset
    Installation
    Usage
    Model Training and Evaluation
    Results
    Future Scope
    Contributing
    License

Introduction

Rice is a staple food for billions, and its yield is often affected by various diseases. Early and precise detection of these diseases can mitigate losses and enhance productivity. BotanicVision uses DenseNet201, a robust CNN architecture, to classify rice leaf diseases with high accuracy, providing actionable insights to improve agricultural outcomes.
Features

    Deep Learning-Based Disease Detection: Utilizes DenseNet201 for feature extraction and classification.
    Pretrained Weights: Incorporates transfer learning for faster and efficient model training.
    Scalable Architecture: Can be adapted for other crops and diseases with minimal changes.
    User-Friendly Deployment: Designed to be integrated into mobile or web applications.

Technologies Used

    Programming Language: Python
    Deep Learning Framework: TensorFlow/Keras
    Model Architecture: DenseNet201
    Visualization: Matplotlib, Seaborn
    Data Handling: Pandas, NumPy

Dataset

The dataset consists of labeled images of rice leaves categorized into multiple disease classes, including:

    Healthy
    Bacterial Leaf Blight
    Brown Spot
    Leaf Smut

Usage
Training the Model

Run the following command to train the model:

python train.py  

Evaluating the Model

Evaluate the trained model using:

python evaluate.py  

Making Predictions

Predict the class of a new image:

python predict.py --image path_to_image.jpg  

Model Training and Evaluation

    Preprocessing: Images are resized to 224x224 and normalized.
    Augmentation: Random flipping, rotation, and zooming are applied to enhance generalization.
    Optimizer: Adam optimizer is used with a learning rate scheduler.
    Loss Function: Categorical cross-entropy.
    Metrics: Accuracy, precision, recall, and F1-score.

Results

The model achieved the following performance on the test set:

    Accuracy: 98%
    Precision: 97%
    Recall: 96%
    F1-Score: 96.5%

Confusion matrix and sample predictions are provided in the results/ directory.
Future Scope

    Expanding to detect other crop diseases.
    Integrating with IoT devices for real-time field analysis.
    Deploying as a mobile or web application for broader accessibility.
