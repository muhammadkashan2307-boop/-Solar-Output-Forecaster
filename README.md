# 🧠 DeepGrid AI: Multi-Dataset Neural Solar & Smog Forecaster

**DeepGrid AI** is an end-to-end Deep Learning and Smart Microgrid analytics web application. Built with a custom **3-Layer Multi-Layer Perceptron (MLP) Deep Neural Network**, the system fuses three distinct datasets to accurately predict daily solar photovoltaic (PV) energy generation (kWh), quantify power losses caused by severe urban smog (AQI), and calculate real-time financial savings in PKR.

---

## 🎯 Problem Statement
Traditional solar calculators rely on basic linear equations that only look at panel size and sunlight hours. However, in cities like **Lahore, Pakistan**, solar arrays experience severe non-linear efficiency drops due to:
1. **Winter Smog & High AQI (40–450+):** Thick particulate matter scatters and blocks incoming solar photons.
2. **Summer Thermal Derating (>35°C):** Extreme ambient heat reduces solar cell voltage efficiency unless offset by wind cooling.

**DeepGrid AI** solves this by training a Deep Neural Network on multi-variable atmospheric, hardware, and grid telemetry data to model real-world solar physics.

---

## 🗄️ Multi-Dataset Data Fusion Architecture
The project merges **1,500+ records** across three domain-specific datasets into a single master training pipeline:

1. **Dataset 1: Solar Array & Irradiance Telemetry (GitHub Synced)**
   * Features: `Sunlight_Hours`, `Solar_Irradiance_Wm2`, `Panel_Capacity_kW`
   * Syncs live reference telemetry from open-source GitHub solar generation logs.
2. **Dataset 2: Atmospheric, Smog (AQI) & Thermal Derating Data**
   * Features: `Ambient_Temp_C`, `Smog_AQI`, `Cloud_Cover_Pct`, `Wind_Speed_kmh`, `Humidity_Pct`
   * Captures photon attenuation from smog and thermal efficiency penalties.
3. **Dataset 3: Household Grid Load & LESCO Tariff Data**
   * Features: `Home_Load_kWh`, `Grid_Tariff_PKR`, `Battery_Health_Pct`
   * Powers the financial engine for net-metering export/import calculations and monthly PKR savings.

---

## 🧠 Deep Neural Network Architecture
Instead of basic linear regression, this project implements a custom **Feedforward Deep Neural Network (MLP)** from scratch using pure vectorized NumPy:
* **Input Layer:** 9 Z-score normalized features (`StandardScaler`)
* **Hidden Layer 1:** 128 Neurons (`ReLU` Activation + He Weight Initialization)
* **Hidden Layer 2:** 64 Neurons (`ReLU` Activation)
* **Hidden Layer 3:** 32 Neurons (`ReLU` Activation)
* **Output Layer:** 1 Linear Neuron (Predicted Daily Solar Output in `kWh`)
* **Optimizer & Training:** Trained over **200 Epochs** using Backpropagation and the **Adam Optimizer** ($\beta_1 = 0.9, \beta_2 = 0.999, \text{lr} = 0.01$) minimizing Mean Squared Error (MSE).

---

## ✨ Key Features
* **⚡ Real-Time Neural Inference:** Adjust 9 interactive sidebar sliders to instantly predict daily solar generation (`kWh`).
* **🌫️ Smog Loss Quantifier:** Automatically runs a counterfactual neural simulation comparing current AQI against clean air (`AQI = 40`) to display exact `kWh` lost to smog.
* **💰 Smart Grid Financial Engine:** Calculates daily net-metering balance (Grid Export vs. Grid Import) and estimated monthly savings in **PKR**.
* **📈 Live Training Diagnostics:** Visualizes the 200-epoch **Neural Network Loss Curve (MSE)**, Actual vs. Predicted test performance, and an interactive **Smog Degradation Curve**.
* **☁️ Google Drive & Cloud Export:** Automatically backs up fused datasets when run in Google Colab and lets users download custom neural prediction reports as a `.csv` file.

---

## 🛠️ Tech Stack
* **Frontend & Cloud Deployment:** Streamlit Community Cloud
* **Deep Learning & Math Engine:** Pure Vectorized NumPy (Custom MLP Backpropagation & Adam Optimizer)
* **Data Engineering:** Pandas
* **Cloud Storage Integration:** Google Colab Drive API & CSV Export
