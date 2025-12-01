# Credit Risk Prediction using Explainable AI (EBM)

This project predicts whether a borrower will default on a credit loan using an Explainable Boosting Machine (EBM).  
The model provides transparent, interpretable reasons behind each prediction.

---

## 🚀 Features
- Explainable AI model (EBM)
- Top 3 reasons for every prediction
- Clean Streamlit UI
- Synthetic dataset generation
- Complete ML training pipeline

---

## 📁 Project Structure

credit-risk-xai/
│── app.py # Streamlit app
│── train_ebm.py # Train EBM model
│── create_synthetic.py # Generate dataset
│── model/ebm_pipeline.joblib
│── data/credit.csv
│── requirements.txt
│── README.md

---

## 🔧 Installation

git clone git@github.com:MohdAinul/credit-risk-xai.git
cd credit-risk-xai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

yaml
Copy code

---

## ▶️ Run App

streamlit run app.py

yaml
Copy code

---

## 📌 Example Output

- Default Probability: 0.42  
- Result: ❌ Likely to Default  
- Top Reasons: PAY_0, BILL_AMT1, LIMIT_BAL  

---

## 🧠 Tech Used
- Python  
- InterpretML  
- Streamlit  
- Scikit-learn  
- Joblib  

---

## 📜 Report
The complete project report is included in the repository.

---

## ⭐ If you like this project, please star the repo!
