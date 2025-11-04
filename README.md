# Road-Accident
# UK Road Accident Severity Prediction 🚗

This project analyzes road accident data from the UK to build a machine learning model that can predict the severity of an accident (Slight, Serious, or Fatal) based on contributing factors like weather, time, speed limit, and road conditions.

## 🎯 Problem Statement & Goal

The goal of this project is to identify the key factors that contribute to severe accidents. By building a classification model, we aim to:
1.  Predict the severity of an accident given a set of circumstances.
2.  Understand which features (e.g., `Speed_limit`, `Weather_Conditions`) are the most important predictors.
3.  Provide a basis for potential road safety interventions.

## ⚙️ Project Workflow

My process for this project followed these main steps:

1.  **Data Loading & Cleaning:** Loaded the data into a `pandas` DataFrame. Checked for null values and corrected data types (e.g., `Accident Date` to datetime).
2.  **Feature Engineering:**
    * Encoded the target variable `Accident_Severity` into a numeric format: `Accident_Severity_Encoded` (1: Slight, 2: Serious, 3: Fatal).
    * Used existing one-hot encoded features for modeling (e.g., `JD_...`, `LC_...`, `Weather_...`).
3.  **Exploratory Data Analysis (EDA):**
    * Analyzed the distribution of the target variable (Accident Severity), revealing a significant **class imbalance**.
    * Plotted accidents by hour of the day, day of the week, and weather conditions to identify trends.
4.  **Model Preprocessing:**
    * Identified and removed "leaky" columns (`Number_of_Casualties`) and non-numeric columns (`Accident Date`) from the feature set.
    * Used a `ColumnTransformer` and `StandardScaler` to scale continuous features (`Speed_limit`, `Latitude`, `Longitude`, etc.) while leaving the (0/1) encoded features untouched.
5.  **Modeling & Evaluation:**
    * Built a `Pipeline` to streamline the preprocessing and modeling steps, preventing data leakage.
    * Trained and evaluated a baseline **Logistic Regression** model.
    * Trained and evaluated a **Random Forest Classifier** to handle non-linearity and improve performance, using `class_weight='balanced'` to combat imbalance.
    * Evaluated models using `accuracy_score`, `classification_report`, and `confusion_matrix`.

## 📊 Key Findings & Results

### Exploratory Data Analysis (EDA)

* **Class Imbalance:** The dataset is heavily imbalanced. The vast majority of accidents are "Slight," making it challenging to predict the rare "Serious" and "Fatal" classes.
    * `Slight (1)`: ~84%
    * `Serious (2)`: ~14%
    * `Fatal (3)`: ~2%
* **Accidents by Hour:** Accident frequency peaks during morning and evening commute times (around 8:00 AM and 5:00 PM).
* **Accidents by Day:** Accidents are most frequent on Fridays and drop significantly on Sundays.
