# Credit Risk Predictor

This project predicts whether a borrower will default on a credit loan using three machine learning models: **Logistic Regression**, **Random Forest**, and **XGBoost**.

The application uses **SHAP (SHapley Additive exPlanations)** to provide transparent, interpretable reasons behind each prediction (specifically for the XGBoost model).

---

## 🚀 Features
- **3-Model Prediction:** Compares results from Logistic Regression, Random Forest, and XGBoost.
- **Top 3 Reasons:** Explains the prediction using SHAP values (XGBoost).
- **Clean Streamlit UI:** Easy-to-use interface for entering borrower details.
- **Synthetic Dataset Generation:** Includes a script to generate training data.
- **Complete ML Pipeline:** Scripts to train and save models.

---

## 📊 Model Comparison

The project trains and compares three models:

1.  **Logistic Regression** (Baseline)
2.  **Random Forest**
3.  **XGBoost** (Primary Model)

The Streamlit UI displays the default probability from all three models, but uses XGBoost for the primary decision and explanation.

---

## 📁 Project Structure

```
credit-risk-xai/
│── app.py                # Streamlit app
│── train_models.py       # Train LR, RF, and XGBoost models
│── create_synthetic.py   # Generate synthetic dataset
│── model/                # Saved model pipelines (.joblib)
│── data/                 # Dataset directory
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
```

---

## 🔧 Setup & Installation

Follow the instructions below for your operating system.

### 1. Windows (using WSL Terminal)

Prerequisites: Ensure you have WSL (Windows Subsystem for Linux) installed with a Linux distribution (e.g., Ubuntu).

1.  **Open your WSL terminal.**

2.  **Clone the repository:**
    ```bash
    git clone git@github.com:MohdAinul/credit-risk-xai.git
    cd credit-risk-xai
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Generate the dataset:**
    ```bash
    python create_synthetic.py
    ```

6.  **Train the models:**
    ```bash
    python train_models.py
    ```

7.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

### 2. Mac

1.  **Open your Terminal.**

2.  **Clone the repository:**
    ```bash
    git clone git@github.com:MohdAinul/credit-risk-xai.git
    cd credit-risk-xai
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Generate the dataset:**
    ```bash
    python create_synthetic.py
    ```

6.  **Train the models:**
    ```bash
    python train_models.py
    ```

7.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

---

## 📌 Example Output

- **Default Probability:** 42.5%
- **Result:** Borrower is unlikely to default.
- **Top Reasons:** PAY_0, BILL_AMT1, LIMIT_BAL

---

## 🧠 Tech Used
- Python
- Streamlit
- Scikit-learn
- XGBoost
- SHAP
- Joblib

---

## ⭐ If you like this project, please star the repo!
