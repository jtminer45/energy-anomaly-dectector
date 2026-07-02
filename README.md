# Energy Anomaly Detector

A deep learning system using PyTorch autoencoders to detect anomalies in real hourly energy consumption data. Built as a portfolio project for MSc Data Science applications and the Anthropic Fellows Program.

## Project Overview
This project applies unsupervised deep learning to identify unusual patterns in energy consumption data from AEP (American Electric Power) — one of the largest electric utilities in the US. The system learns what "normal" consumption looks like and flags readings that deviate significantly.

## Real World Applications
- Equipment failure detection
- Energy theft identification
- Grid instability warnings
- Demand forecasting validation

## Dataset
AEP Hourly Energy Consumption — sourced from Kaggle.
Download: kaggle.com/datasets/robikscube/hourly-energy-consumption
Place `AEP_hourly.csv` in the `data/` folder to reproduce this project.

- 121,273 hourly readings
- Date range: 2004 to 2018
- Single feature: energy consumption in Megawatts (MW)

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Time series EDA — hourly and monthly consumption patterns, distribution analysis |

## Key EDA Findings
- Energy consumption follows a clear U-shaped seasonal pattern — high in winter and summer, low in spring and autumn
- Daily consumption spikes at 6am (morning activity) and 6pm (evening return home)
- Consumption is flat between 12pm and 5pm — consistent with commercial/office hours
- Most readings fall between 5,000-10,000 MW with rare outliers above 15,000 MW

## Stack
- Python, Pandas, NumPy
- PyTorch
- Matplotlib, Seaborn
- Jupyter Notebooks

## What is an Autoencoder?
An autoencoder is a neural network trained to compress and reconstruct normal data. When it encounters an anomalous reading it struggles to reconstruct it accurately — producing high reconstruction error. That error is the anomaly signal.

## Next Steps
- Data cleaning and feature engineering
- PyTorch autoencoder architecture
- Anomaly detection and threshold tuning
- Visualisation of detected anomalies

## Goal
Demonstrate deep learning engineering capability relevant to AI research roles including the Anthropic Fellows Program.
