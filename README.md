# Latency-Aware-and-Priority-Based-Task-Scheduling-for-Real-Time-Traffic-Monitoring-
 Latency-Aware and Priority-Based Task Scheduling for Real-Time Traffic Monitoring Using Edge Computing
# 🚦 Video-Driven Edge–Cloud Traffic Analytics System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Edge Computing](https://img.shields.io/badge/Domain-Edge%20Computing-green)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-orange)
![Computer Vision](https://img.shields.io/badge/CV-YOLOv8-red)
![Simulation](https://img.shields.io/badge/Simulation-SimPy-purple)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview

This project implements a **video-driven edge computing framework** for **real-time traffic analytics**, where traffic video streams are analyzed to generate computational tasks that are dynamically scheduled between **edge nodes and a cloud server**.

The system integrates **computer vision**, **machine learning–based congestion detection**, and **discrete-event simulation** to study **latency-aware task scheduling and offloading** in edge–cloud environments.

---

## 🎯 Objectives

- Extract traffic workloads directly from real-world video
- Detect vehicles and estimate their speeds
- Predict traffic congestion using a trained ML model
- Dynamically prioritize tasks at the edge
- Offload tasks to the cloud when deadlines cannot be met
- Analyze latency and execution behavior

---

## 🏗️ High-Level Architecture

Traffic Video
↓
YOLOv8 Vehicle Detection
↓
Vehicle Tracking & Speed Estimation
↓
Congestion Prediction (ML)
↓
Task Generation
↓
Priority-Based Edge Scheduling
↓
Cloud Offloading (if required)
↓
Performance Analytics


---

## 🧠 Core Components

### 🔹 Video Analytics
- **YOLOv8 (pretrained)** for vehicle detection
- Detects cars, buses, trucks, and motorcycles
- Lightweight tracking for vehicle identity
- Speed estimation using pixel displacement and FPS

### 🔹 Congestion Detection
- **Random Forest classifier**
- Trained locally using synthetic traffic data
- Input features:
  - Vehicle count
  - Average speed
- Output classes:
  - LOW
  - MEDIUM
  - HIGH
- Model saved as `congestion_model.joblib`

### 🔹 Task Generation
Each detected vehicle produces a task with:
- Priority (urgent / normal)
- Compute workload (MI)
- Deadline constraints

Urgent tasks are generated when:
- Speed exceeds threshold
- Congestion level is HIGH

### 🔹 Edge–Cloud Simulation
- Implemented using **SimPy (Discrete Event Simulation)**
- Edge node:
  - Priority queue
  - Limited compute capacity
- Cloud node:
  - Higher compute capacity
- Tasks are offloaded when deadlines cannot be met at the edge

---

## 📊 Performance Metrics

The system evaluates:
- Total processed tasks
- Edge vs cloud execution count
- Deadline misses
- Task latency distribution

Outputs include:
- CSV logs
- Latency comparison plots
- Optional congestion-annotated video

---

## 🛠️ Tech Stack

| Category | Tools |
|-------|------|
| Language | Python 3.10+ |
| Computer Vision | OpenCV, YOLOv8 |
| Machine Learning | Scikit-learn (Random Forest) |
| Simulation | SimPy |
| Model Persistence | Joblib |
| Visualization | Matplotlib, Seaborn |

---

## 📂 Project Outputs

After execution, the following files are generated:

congestion_model.joblib # Trained congestion model
simulation_results.csv # Task-level analytics
latency_plot.png # Edge vs cloud latency
predicted_congestion_video.mp4 # Annotated output video (optional)


---

## ▶️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install opencv-python simpy ultralytics scikit-learn joblib matplotlib seaborn
