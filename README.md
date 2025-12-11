# Structural Health Monitoring: Sensor Decorrelation Analysis

In the context of Structural Health Monitoring (SHM), measuring the aging of a structure (bridges, dams, tunnels) is often complicated by environmental factors. 

This project aims to **decorrelate environmental effects** (Temperature and Solar Irradiance) from the raw measurements of a displacement sensor. By removing these reversible thermal effects, we can isolate the irreversible behavior of the structure and assess its true health.

**Objective:**
1.  Load and clean the raw dataset (`dataset.dat`).
2.  Analyze correlations between environmental variables and displacement.
3.  Train a regression model to predict the thermal response.
4.  Compute the **decorrelated signal** (residuals) to reveal the structure's intrinsic behavior.

---

## 🛠️ Installation & Requirements

This project requires **Python 3.8+** and the following data science libraries.

### 1. Clone or Download
Ensure you have the following files in your working directory:
*   `solution.ipynb` (The main Jupyter Notebook)
*   `dataset.dat` (The raw data source)
*   `README.md` (This file)

### 2. Install Dependencies
You can install the necessary libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## 🚀 How to Run

The solution is provided as a **Jupyter Notebook**, which combines code, visualization, and markdown analysis.

1.  Open your terminal or command prompt.
2.  Navigate to the project folder.
3.  Launch Jupyter:
    ```bash
    jupyter notebook
    ```
4.  Open **`solution.ipynb`** and run all cells ("Run All").

**Output:**
*   The notebook displays the correlation matrix, time-series plots, and model performance.
*   A CSV file named **`resultats_decorreles.csv`** will be generated containing the cleaned and processed data.

---

## 🧠 Methodology

### 1. Data Cleaning
*   **Parsing:** The input file `.dat` is read using a flexible separator engine.
*   **Formatting:** The `TIMESTAMP` column is converted to Datetime objects and set as the index.
*   **Error Handling:** String values like `"NAN"` are coerced to numeric `NaN` and rows containing missing values are dropped to ensure data quality.

### 2. Exploratory Data Analysis (EDA)
*   **Correlation Matrix:** Revealed a strong negative correlation between **Ensoleillement (Sun)** and **Deplacement (-0.89)**, as well as **Temperature (-0.74)**.
*   **Visualization:** Confirmed that displacement peaks align perfectly with temperature/sun peaks, validating the hypothesis of a thermal effect.

### 3. Modeling
A **Multiple Linear Regression** model was chosen for its interpretability and robustness given the high linearity of the data.
*   **Formula:** $Displacement = \alpha \cdot Temp + \beta \cdot Sun + Intercept$
*   **Performance:** The model achieved an **$R^2$ score of ~0.86**, meaning 86% of the sensor's movement is explained solely by weather conditions.

### 4. Decorrelation
The decorrelated value is calculated as the residual of the model:
$$Decorrelated = Measured - Predicted_{environment}$$

---

## 📊 Results & Conclusion

*   **Raw Data (Blue curve):** Shows high amplitude oscillations (ranging from -8 to +4) due to daily thermal cycles.
*   **Decorrelated Data (Green curve):** The signal is flattened (amplitude close to 0).
*   **Structural Health:** The decorrelated signal is stable over time. There is **no significant drift**, suggesting the structure is healthy and behaving elastically.

### Future Improvements
To further improve the model (targeting $R^2 > 0.90$):
1.  **Thermal Inertia (Lag):** Concrete structures take time to heat up. Adding lagged variables (e.g., `Temp_T-1h`) would capture this delay.
2.  **Seasonality (HST Model):** For longer datasets (years), adding sinusoidal components would account for seasonal variations (Winter vs. Summer behavior).

---

*For any questions regarding this analysis, please feel free to contact me.*
