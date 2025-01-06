# Fashion Classification App

## Project Overview

This project is a web-based application that classifies fashion items into predefined categories using pre-trained deep learning models. Users can upload an image, and the app will predict the item category and display the prediction probabilities. The project demonstrates the use of CNN and VGG16 models for image classification.

## Features

**Models:**
- **Convolutional Neural Network (CNN):**
    - A lightweight model trained specifically for this project.
- **VGG16:**
    - A robust pre-trained model with fine-tuning for fashion item classification.

**Functionality:**
- Upload an image of a fashion item (e.g., T-shirt, trousers, shoes). On a white background more preferable
- Predict the class of the item from 10 predefined categories.
- Display prediction probabilities for each category.
- Visualize prediction probabilities as a bar chart.

## Technologies Used

- **Languages**: Python
- **Libraries**:
    - **TensorFlow/Keras** for deep learning models.
    - **Matplotlib** for data visualization.
    - **NumPy** for numerical operations.
    - **Pillow** for image preprocessing.
    - **Streamlit** For building an interactive web application to visualize data and make predictions
- **Tools**:
    - **Google Colab** for model training and experimentation.
    - **Git & GitHub** for version control.

## Installation

To set up the environment and run the project, follow these steps:

1. Clone the repository:
```
git clone https://github.com/jamleston/fashion-app
cd fashion-app
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Run the application:
```
streamlit run app.py
```

4. Or you can also see my project through this link:
**https://m3p3knpxfhcu6qh7jeenkh.streamlit.app/**

## Usage

- Upload a fashion item image (JPEG, PNG).
- Select a model (CNN or VGG16) from the sidebar.
- View the predicted class and class probabilities.
- Analyze the prediction probabilities via the bar chart.

## Repository Structure
```
├── venv/                         # Virtual environment directory
├── models.ipynb                  # Jupyter notebook for model training and experimentation
├── app.py                        # Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
```

## Developed by
- [Valeriia Alieksieienko](https://github.com/jamleston)