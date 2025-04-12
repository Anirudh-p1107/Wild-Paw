

# WildPaw: Animal Footprint Classification with CNN

<p align="center">
  <img src="sound.jpg" alt="Animal Footprint Classification" height="50%" width="80%">
</p>

## Overview
   "WildPaw" is a deep learning project that leverages Convolutional Neural Networks (CNNs) to classify animals based on their footprints. Built in Google Colab, WildPaw supports wildlife monitoring and ecological research through robust footprint image analysis. It allows classification of animals such as lions, deers, horses, wolves, and more through their footprint patterns.

## Features
- **Footprint Classification:** Recognizing animal species using footprint images.
- **Image Preprocessing:** Handles resizing, rotation, and normalization of input images.
- **CNN Architecture:** Custom CNN with rotation invariance to handle footprints in any orientation.
- **Training Pipeline:** Includes data augmentation, early stopping, and model checkpointing.
- **Prediction:** Identifies animal species from new image files with confidence scores.
- **Environment:** Fully implemented in Google Colab with Google Drive integration for dataset storage.


## How It Works
- **Image Input:** The user provides a grayscale image of an animal footprint.
  
 *Class Distribution*

| Class     | Count | Class     | Count |
|-----------|-------|-----------|-------|
| Bear      | 50    | Deer      | 50    |
| Bobcat    | 50    | Horse     | 50    |
| Fox       | 50    | Lion      | 50    |
| Mouse     | 50    | Wolf      | 50    |
| Racoon    | 50    | Squirrel  | 50    |

- **Preprocessing:** Images are resized to 128x128, normalized, and augmented with random rotations and flips.
- **Feature Extraction:** A custom CNN learns to identify key features of each footprint.
- **Classification:** The model outputs a predicted animal class based on footprint features.
- **Prediction Output:** The system provides both the predicted label and the confidence score.

## Model Training
- Uses a CNN with multiple convolutional + batch norm + dropout layers.
- Implements learning rate scheduling and model checkpointing.
- Saves the best-performing model (`best_footprint.pth`).

   <p align="center">
  <img src="training_history.png" alt="Plotting graph" height="50%" width="80%">
</p>

  ## Classification Report

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **Bear**     | 0.83      | 0.81   | 0.82     | 31      |
| **Bobcat**   | 0.76      | 0.91   | 0.83     | 34      |
| **Deer**     | 0.88      | 0.74   | 0.81     | 31      |
| **Fox**      | 0.76      | 0.78   | 0.77     | 32      |
| **Horse**    | 0.97      | 0.97   | 0.97     | 33      |
| **Lion**     | 0.94      | 0.80   | 0.87     | 41      |
| **Mouse**    | 0.90      | 1.00   | 0.95     | 36      |
| **Racoon**   | 0.78      | 0.86   | 0.82     | 37      |
| **Squirrel** | 0.78      | 0.78   | 0.78     | 23      |
| **Wolf**     | 0.89      | 0.78   | 0.83     | 32      |


**Overall Accuracy:** 85%  
**Macro Avg:** Precision = 0.85, Recall = 0.84, F1 = 0.84  
**Weighted Avg:** Precision = 0.85, Recall = 0.85, F1 = 0.85  

 ## Tools Used
- **Google Colab:** Free online place to run the project.
- **Google Drive:** Stores the audio files.
- **OpenCV:** Helps turn sounds into pictures.
- **PyTorch:** Deep learning framework powering the CNN.

 ## Dataset Access
You can access the dataset from the following Google Drive link:
[Dataset Link](https://drive.google.com/drive/folders/1W6hzpkutT4BEexB5zqV0EGFg9v99afKO?usp=drive_link)

 ## Results
- **Training:** Achieved strong generalization within 50 epochs.
- **Testing:** Accurately classifies footprints like “Lion (90%)” or “Mouse (95%)”.
- **Performance:** Works best with balanced class samples and clear prints.

 ## Uses
- **Wildlife Monitoring:** Tracks animals in protected habitats.
- **Environmental Studies:** Monitors biodiversity changes through track analysis.
- **Smart Farming:** Identifies animal intrusions on farmland by their prints.
- **Research & Education:** Enhances ecological learning using AI.
- **Rescue Teams:** Helps locate endangered species through trace evidence.

 ## Contributing
Feel free to submit issues or pull requests for improvements!
