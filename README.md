# Machine Learning Detection of Insider Threats Using Behavioral Log Analysis

## Project Purpose

This project builds a machine learning pipeline for detecting possible insider threats from behavioral and system activity logs. The goal is to identify unusual user behavior such as after-hours access, spikes in file activity, removable device use, heavy email activity, and abnormal web browsing patterns.

The project is designed as an undergraduate data science course project. It emphasizes clear preprocessing, interpretable feature engineering, standard machine learning models, and practical evaluation metrics for rare security events.

## Research Question

Can machine learning models detect insider threats by learning anomalous patterns in user activity logs such as logon/logoff events, file access, device usage, email, and web browsing?

## Dataset Source

The intended dataset is the CERT Insider Threat Dataset from Carnegie Mellon University's CERT Division.

Expected CSV files include:

- `logon.csv`
- `device.csv`
- `file.csv`
- `email.csv`
- `http.csv`
- ground truth

Place these files in the local `data/` folder:

```text
data/
  logon.csv
  device.csv
  file.csv
  email.csv
  http.csv
  decoy_file.csv
  answers.csv or answers.tar.bz2
```

The notebook also includes a small synthetic demo fallback so the workflow can run even when the real CERT files are not available. Any final report should use the real CERT data if the assignment requires empirical results.

## Project Structure

```text
.
├── data/
├── notebooks/
│   └── insider_threat_detection.ipynb
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── evaluation.py
├── README.md
└── requirements.txt
```

## Installation
First, open Terminal and move into the project folder:

```bash
cd path/to/Insider-Threat-Detection-Project

Create and activate a virtual environment:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```
## Running the Notebook

PLEASE ALLOW ROUGHLY 245 SECONDS FOR ALL CELLS TO FINISH RUNNING

Start JupyterLab:

```bash
jupyter lab
```

Open:

```text
notebooks/insider_threat_detection.ipynb
```

Run the notebook from top to bottom. If CERT CSV files are in `data/`, the notebook will load them. It can also read CSV answer files extracted under `data/answers/` or stored inside an `answers.tar.bz2` archive in `data/`. If no CSV files are found, it will generate a small synthetic CERT-like dataset for demonstration.

Large CERT releases can be too big to load fully on a laptop. The notebook uses sampled loading by default:

```python
USE_SAMPLED_LOADING = True
SAMPLE_FRACTION = None
MAX_ROWS_PER_LOG = 250_000
ANSWER_RELEASE = "r6.2"
```

This keeps the notebook manageable by reading a capped number of rows from each large normal log and adding malicious rows from the answer key. If you have enough memory and time, you can set `USE_SAMPLED_LOADING = False` in the notebook.

## Expected Outputs

The notebook produces:

- Dataset loading and missing-value checks
- Exploratory data analysis tables
- Visualizations for:
  - Class distribution
  - Login activity over time
  - Event counts by user
  - Correlation heatmap
  - Feature distributions
- Aggregated user/time-window behavioral features
- Trained supervised and anomaly detection models
- Model comparison table
- Confusion matrix and ROC curves
- Final discussion of best model, useful features, limitations, and future improvements

## Models Used

Supervised models:

- Logistic Regression
- Random Forest
- Support Vector Machine

Anomaly detection models:

- Isolation Forest
- One-Class SVM

## Metrics Used

The project reports:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix
- False negative rate

Recall and false negative rate are emphasized because insider threats are rare and missed threats can be costly. A model with high accuracy may still be weak if it misses most malicious users or malicious time windows.

## Limitations

- Insider threat data is highly imbalanced, so accuracy alone can be misleading.
- CERT is synthetic, which means results may not fully generalize to real organizations.
- Labels may be incomplete or provided as malicious intervals rather than event-level labels.
- Aggregating by user and day can hide short bursts of activity; smaller windows may detect faster attacks but can be noisier.
- Behavioral features may reflect policy violations or unusual work patterns, not necessarily malicious intent.
- Real-world deployment would require privacy review, access controls, monitoring policies, and human analyst validation.

