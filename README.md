# FNB DataQuest 2026 — Interpretable Credit Risk Modelling

An interactive Streamlit dashboard for exploring a loan book dataset and an
interpretable credit-risk model (logistic regression with WoE/IV binning),
built for the FNB DataQuest 2026 challenge.

**🔗 Live demo:** [fnb-dataquest-credit-risk-2026.streamlit.app](https://fnb-dataquest-credit-risk-2026-nzk8mpoezxhocqrj97uxcy.streamlit.app/)

## Project structure

```
.
├── app.py            # Streamlit dashboard (EDA + model results viewer)
├── model.py          # Data cleaning, feature engineering, and model training
├── loan_book.csv     # Loan book dataset used for EDA and modelling
└── requirements.txt  # Python dependencies
```

## Setup

```bash
git clone https://github.com/Xola1917/FNB-DataQuest-Credit-Risk-2026.git
cd FNB-DataQuest-Credit-Risk-2026
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. **Train the model** (generates `model_outputs.pkl`, used by the dashboard):
   ```bash
   python model.py
   ```
2. **Launch the dashboard:**
   ```bash
   streamlit run app.py
   ```
   This opens the app in your browser, typically at `http://localhost:8501`.

## About

Built by Xola Mtintweni as part of the FNB DataQuest 2026 challenge.
