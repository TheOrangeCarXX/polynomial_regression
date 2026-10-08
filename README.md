# Polynomial Regression — ML Assignment 1

**Roll Number:** BT2024103  
**Course:** Machine Learning (Sem 5)

---

## Overview

This repository contains the solution for ML Assignment 1, which involves building polynomial regression models to predict a continuous target variable `y` from two personalised datasets (`var1` and `var2`).

| Problem | Description | Features | Max Degree |
|---|---|---|---|
| `var1` | Power Plant Steam Turbine Optimization | 6 (`x1`–`x6`) | 10 |
| `var2` | Subterranean Thermal Reservoir Mapping | 3 (`x1`, `x2`, `x3`) | 20 |

---

## Approach

- **Polynomial feature expansion** using `sklearn.preprocessing.PolynomialFeatures`
- **Ridge regression** with automatic regularization parameter selection via `RidgeCV`
- **5-fold cross-validation** to select the optimal polynomial degree
- **StandardScaler** applied inside the pipeline to prevent data leakage

### Selected Degrees

| Dataset | Best Degree | Ridge α | Train MSE | Train R² |
|---|---|---|---|---|
| `var1` | 5 | 24.4205 | 0.20328 | 0.97912 |
| `var2` | 11 | 1.2068 | 0.16019 | 0.99602 |

---

## Repository Structure

```
.
├── ml_assignment_solution.py   # Full training + CV + inference pipeline
├── BT2024103_pred_var1.csv     # Predictions for var1 test set
├── BT2024103_pred_var2.csv     # Predictions for var2 test set
├── report_BT2024103.tex        # LaTeX source for the report
├── figures/
│   ├── var1_degree_vs_error.png
│   ├── var1_residuals.png
│   ├── var2_degree_vs_error.png
│   └── var2_residuals.png
└── README.md
```

---

## How to Run

1. Place the dataset files in a folder matching the expected path in the script, or update the `BASE` / `DATA` variables in `ml_assignment_solution.py`.
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run the solution script:
   ```bash
   python ml_assignment_solution.py
   ```
   This will:
   - Run 5-fold CV across all candidate degrees
   - Train final Ridge models
   - Save prediction CSVs (`BT2024103_pred_var1.csv`, `BT2024103_pred_var2.csv`)
   - Save all figures to `figures/`
