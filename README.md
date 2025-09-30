# Building-a-Digit-Recognizer-from-Scratch

## Project Description

This project focuses on building and comparing different deep learning models for classifying images from two popular datasets: Fashion-MNIST and CIFAR-100. The goal is to explore the impact of increasing model complexity, from a simple Artificial Neural Network (ANN) to more sophisticated Convolutional Neural Networks (CNNs), on performance and efficiency across datasets of varying complexity. The project covers data preprocessing, model building (ANN, Basic CNN, Deeper CNN), training with Early Stopping and Model Checkpointing, model evaluation (loss, accuracy, training history, confusion matrices/classification reports), and prediction analysis.

## Tech Stack

*   **Python:** The primary programming language used for the analysis and model building.
*   **pandas:** Used for data handling and analysis (though less prominent in this image-focused project).
*   **NumPy:** Used for numerical operations and data handling, especially with image data.
*   **TensorFlow / Keras:** The main library used for building, training, and evaluating deep learning models.
*   **Matplotlib:** Used for creating static visualizations, such as the image grids and performance comparison plots.
*   **Seaborn:** Used for creating informative statistical graphics, specifically for the confusion matrices.
*   **Plotly:** Used for creating interactive visualizations of training history.
*   **scikit-learn:** Used for generating classification reports and confusion matrices.
*   **Google Colab:** The environment where the notebook was developed and executed.

## How to Run

1.  **Open the notebook in Google Colab:** Upload or open the notebook file (.ipynb) in your Google Colab environment.
2.  **Install dependencies:** Ensure you have the necessary libraries installed. If running in a new Colab environment, you can install them using pip (refer to the `requirements.txt` file generated earlier).
3.  **Run all cells:** Execute all the code cells in the notebook sequentially from top to bottom. This will:
    *   Load and preprocess both Fashion-MNIST and CIFAR-100 datasets.
    *   Define and compile the ANN, Basic CNN, and Deeper CNN models for both datasets.
    *   Train each model with Early Stopping and Model Checkpointing.
    *   Evaluate the performance of each model on their respective test sets.
    *   Visualize training history and analyze predictions.

## Screenshot

**1. Model Performance Comparision (CIFAR-100)**

<img width="565" height="245" alt="image" src="https://github.com/user-attachments/assets/3be6d860-f10a-4d96-8210-ecd7b751c392" />

**2. Deeper CNN Predictions (CIFAR-100) (Mix of Correct and Incorrect)**

<img width="663" height="341" alt="image" src="https://github.com/user-attachments/assets/b8ed2059-4a3e-4a09-b8c6-59385dee0eee" />
