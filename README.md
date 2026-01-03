# Eye Disease Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning project designed to classify various eye diseases from retinal fundus images. This repository explores **Supervised Learning** baselines and implements **Semi-Supervised Learning** techniques to improve performance on limited labeled data. It also features a **Streamlit** web application for easy model demonstration.

## 📂 Repository Structure

| Directory / File | Description |
| :--- | :--- |
| **`BaseModel/`** | Contains the initial supervised baseline models, preprocessing scripts, and architecture definitions. Swin and Efficient Net architecture is experimented. |
| **`SemiSupervisedLearning/`** | Experiments utilizing semi-supervised learning. Used mean teacher framework. |
| **`streamlit_app.py`** | A Python script to launch the web interface for real-time image classification. |

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* pip

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/BeratMert29/EyeDiseaseClassification.git](https://github.com/BeratMert29/EyeDiseaseClassification.git)
    cd EyeDiseaseClassification
    ```

2.  **Install dependencies**
    *(Note: Ensure you have the necessary libraries installed. If a requirements.txt is available, use that. Otherwise, install the core packages below.)*
    ```bash
    pip install numpy pandas matplotlib scikit-learn streamlit
    # pip install torch torchvision
    ```

## 💻 Usage

### Running the Web Application
To start the graphical interface and test the model with your own images:

```bash
streamlit run streamlit_app.py

The app will open in your browser (usually at http://localhost:8501).
Upload a retinal image to see the predicted disease class and confidence score.
```

Training the Models

To reproduce training results or improve the models:

    Base Model: Navigate to BaseModel/ and run the provided notebooks/scripts to train the supervised baseline.

    Semi-Supervised: Navigate to SemiSupervisedLearning/ to run experiments that utilize unlabeled data.

📊 Dataset & Classes

This model is designed to classify conditions such as:

    Normal

    Glaucoma

    Cataract

    Diabetic Retinopathy
