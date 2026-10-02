# Prediction of UAV Battery Endurance and Low-Energy Risk Using Machine Learning Algorithms

## 📌 Project Overview

This project develops a Machine Learning-based UAV battery monitoring system for estimating **Battery State of Health (SOH)** and identifying the current **battery operating state**.

UAV batteries are affected by factors such as charge-discharge cycles, current, temperature, voltage, internal resistance, payload and flight conditions. These factors can influence battery performance, endurance and mission reliability.

The proposed system uses Machine Learning models to analyse these parameters and provide battery health and state predictions through an interactive **Streamlit dashboard**.

---

## 🎯 Aim

The aim of this project is to develop a Machine Learning-based UAV battery monitoring system that predicts battery State of Health (SOH), classifies battery operating states and provides visual insights to support UAV battery monitoring and low-energy awareness.

---

## 📊 Dataset

The project uses the:

**UAV Battery SOC and SOH Monitoring Dataset**

Dataset characteristics:

- 15,000 records
- 37 columns
- 3 categorical features
- Electrical, thermal, environmental and flight-related measurements
- No missing values
- No duplicate records

### Important Features

- Battery Type
- Operation Mode
- Flight Phase
- Cycle Count
- Voltage
- Current
- Temperature
- Charge Capacity
- Discharge Capacity
- Flight Duration
- Payload Weight
- Altitude
- Flight Speed
- Humidity
- Internal Resistance
- Power
- C-rate
- Cell Voltage
- Cell Voltage Imbalance
- Depth of Discharge
- SOC
- SOH

---

## 🤖 Machine Learning Modules

### 1. SOH Prediction

State of Health is predicted as a continuous value using:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression (SVR)

Evaluation metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### 2. Battery State Classification

The battery is classified into:

- Healthy_High_SOC
- Moderate_State
- Aging_State
- Low_SOC
- Degraded

Classification models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 📈 Results

### SOH Regression

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1.7481 | 2.2024 | 0.9228 |
| Decision Tree | 2.4957 | 3.1726 | 0.8397 |
| Random Forest | 1.7737 | 2.2388 | 0.9202 |
| Gradient Boosting | 1.7487 | 2.2003 | 0.9229 |
| SVR | 1.9547 | 2.4419 | 0.9050 |

### Battery State Classification

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 89.73% | 69.34% | 83.67% | 70.88% |
| Decision Tree | 92.47% | 71.06% | 74.87% | 71.44% |
| Random Forest | 95.37% | 74.36% | 78.26% | 75.74% |
| Gradient Boosting | 95.90% | 82.99% | 74.91% | 76.47% |

The classification experiment uses leakage-controlled features by excluding variables such as SOC, SOH, capacity fade and resistance growth.

---

## 📊 Project Outputs

The training pipeline automatically generates:

- SOC distribution
- SOH distribution
- Battery state distribution
- SOH vs Cycle Count
- Capacity Fade vs Cycle Count
- Temperature vs Current
- Payload vs Flight Duration
- Power vs Flight Duration
- SOH model comparison
- Classification model comparison
- Confusion Matrix
- Classification Report

---

## 🖥️ Streamlit Dashboard

The project includes an interactive dashboard displaying:

- SOC
- Actual SOH
- Predicted SOH
- Voltage
- Current
- Temperature
- Cycle Count
- Flight Duration
- Predicted Battery State
- SOH vs Cycle Count
- Battery State Distribution

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/saisarveshgopinath/Prediction-of-UAV-Battery-Endurance-and-Low-Energy-Risk.git