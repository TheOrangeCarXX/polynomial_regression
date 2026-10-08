# Polynomial regression - ML Assignment 1

Yashwanth Reddy Pidela, roll number BT2024103.

This repository contains polynomial Ridge regression models for the two personalized datasets. Var1 has six inputs and permits total degrees up to 10. Var2 has three inputs and permits total degrees up to 20. Each problem has 1,000 labeled training rows and 1,000 test rows with hidden targets.

## Method and results

For each problem, a shuffled split with seed 42 reserves 200 rows as a holdout and uses 800 rows for model selection. Five-fold cross-validation with seed 43 searches every allowed degree and 50 logarithmically spaced alpha values from 0.0001 to 10,000. Alpha is the Ridge penalty strength. Every fold fits polynomial expansion and scaling using only its 640 training rows, then scores its 160 validation rows.

The search chooses the degree and alpha together by the lowest mean validation MSE. MSE is the mean squared prediction error. R2 compares that error with the target's variation about its mean; larger values are better. R2 is recorded alongside MSE but does not decide the selected settings.

The selected pair fits the 800 development rows and is assessed once on the 200 untouched holdout rows. The pair then fits all 1,000 labeled rows to generate the submission predictions. The holdout result belongs to the 800-row fit. The final training result belongs to the 1,000-row refit.

| Dataset | Selected degree | Alpha | Selection CV MSE | Holdout MSE | Holdout R2 | Final train MSE | Final train R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Var1 | 5 | 24.42053095 | 0.528587 | 0.445705 | 0.962392 | 0.203276 | 0.979125 |
| Var2 | 12 | 1.75751062 | 0.262967 | 0.254243 | 0.994152 | 0.160039 | 0.996019 |

These are the lowest-CV-MSE settings among the evaluated configurations. The selection score can be optimistic because it guided the search. The holdout gives a separate assessment. Hidden test targets are unavailable, so no test MSE or test R2 is claimed.

`PolynomialFeatures(include_bias=False)` creates the nonconstant terms, `StandardScaler` scales them, and `Ridge` fits their coefficients with a separate unpenalized intercept. For example, Var1 at degree 5 has 461 polynomial columns plus the intercept. The final model is a scikit-learn pipeline containing all three fitted steps.

During selection, the script reuses one singular value decomposition per degree and fold to evaluate all 50 Ridge alphas. This computes the Ridge solutions while keeping all preprocessing inside the fold. It avoids repeating the same decomposition for each alpha. The selected settings and the extreme-degree calculations were checked against separate scikit-learn `Ridge(solver="svd")` pipeline fits.

## Files

```text
.
|-- ml_assignment_solution.py       Training, selection, figures, and inference
|-- requirements.txt                Pinned Python dependencies
|-- .gitignore                      Excludes input data and local environment files
|-- README.md
|-- CHANGES_AND_CHECKS.md           Summary of changes and executed checks
|-- ML_Assignment_1_Report.pdf      Six-page report
|-- BT2024103_pred_var1.csv         One y column, 1,000 rows
|-- BT2024103_pred_var2.csv         One y column, 1,000 rows
`-- figures/
    |-- var1_degree_vs_error.png
    |-- var1_residuals.png
    |-- var2_degree_vs_error.png
    `-- var2_residuals.png
```

## Input data

Download the personalized BT2024103 data using the link in the assignment brief. Extract it and put the four CSV files together in a folder named `data`, or point `--data-dir` to the folder that already contains them. The program does not require any particular outer ZIP folder structure.

```text
data/
|-- BT2024103_train_var1.csv   Columns x1,x2,x3,x4,x5,x6,y
|-- BT2024103_test_var1.csv    Columns x1,x2,x3,x4,x5,x6
|-- BT2024103_train_var2.csv   Columns x1,x2,x3,y
`-- BT2024103_test_var2.csv    Columns x1,x2,x3
```

The input datasets are excluded from this repository.

## Run the full workflow

The provided run used Python 3.12.14. Use Python 3.12 and install the pinned dependencies in a virtual environment. Run these commands from the repository folder:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Then install and run:

```bash
python -m pip install -r requirements.txt
python ml_assignment_solution.py --data-dir data
```

The training command searches all 1,500 degree/alpha pairs across the two problems, totaling 7,500 fold evaluations. It saves the predictions and four figures.

To use a different output folder:

```bash
python ml_assignment_solution.py --data-dir "path/to/BT2024103" --output-dir run_outputs
```

The script uses two BLAS threads by default. `--threads` can change this limit. Numerical-library differences may cause small floating-point differences across machines.

The selection figures use a logarithmic MSE axis so the differences among higher degrees remain visible. The residual figures describe the final model's training errors. They are training diagnostics; the holdout metrics provide the independent performance assessment.

## References

The assignment brief specifies the input counts, degree limits, and submission format. The implementation uses the official scikit-learn APIs for [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html), [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), and [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html).
