"""
ML Assignment 1 - Polynomial Regression
Student Roll Number: BT2024103
Problem 1 (var1): Power Plant Steam Turbine Optimization (6 features, degree up to 10)
Problem 2 (var2): Subterranean Thermal Reservoir Mapping (3 features, degree up to 20)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import os

# Paths
BASE = r'c:\Users\yashw\Downloads\Academics\Sem 5\ML\Ass1'
DATA = os.path.join(BASE, 'BT2024103', 'BT2024103')
ROLL = 'BT2024103'

train1 = pd.read_csv(os.path.join(DATA, f'{ROLL}_train_var1.csv'))
test1  = pd.read_csv(os.path.join(DATA, f'{ROLL}_test_var1.csv'))
train2 = pd.read_csv(os.path.join(DATA, f'{ROLL}_train_var2.csv'))
test2  = pd.read_csv(os.path.join(DATA, f'{ROLL}_test_var2.csv'))

print("Dataset shapes:")
print(f"  Train var1: {train1.shape}, Test var1: {test1.shape}")
print(f"  Train var2: {train2.shape}, Test var2: {test2.shape}")


def find_best_degree(X_train, y_train, max_degree, alphas=None, cv=5, label=""):
    if alphas is None:
        alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    results = []

    for degree in range(1, max_degree + 1):
        pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=True)),
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=alphas, cv=kf))
        ])
        scores = cross_val_score(pipe, X_train, y_train,
                                 scoring='neg_mean_squared_error',
                                 cv=kf, n_jobs=-1)
        mean_mse  = -scores.mean()
        std_mse   = scores.std()
        r2_scores = cross_val_score(pipe, X_train, y_train,
                                    scoring='r2', cv=kf, n_jobs=-1)
        mean_r2 = r2_scores.mean()

        n_features = PolynomialFeatures(degree=degree).fit(X_train).n_output_features_
        results.append({
            'degree': degree,
            'cv_mse': mean_mse,
            'cv_mse_std': std_mse,
            'cv_r2': mean_r2,
            'n_features': n_features
        })
        print(f"  [{label}] degree={degree:2d} | CV MSE={mean_mse:.6f} +/- {std_mse:.6f} "
              f"| CV R2={mean_r2:.6f} | #features={n_features}")

        if n_features > 50000:
            print(f"  Stopping at degree {degree}: too many features ({n_features})")
            break

    df_res = pd.DataFrame(results)
    best_row = df_res.loc[df_res['cv_mse'].idxmin()]
    best_degree = int(best_row['degree'])
    print(f"\n  >>> Best degree for {label}: {best_degree} "
          f"(CV MSE={best_row['cv_mse']:.6f}, CV R2={best_row['cv_r2']:.6f})")
    return best_degree, df_res


def train_final_model(X_train, y_train, degree, alphas=None, cv=5):
    if alphas is None:
        alphas = np.logspace(-4, 4, 50)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=True)),
        ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=alphas, cv=kf))
    ])
    pipe.fit(X_train, y_train)
    best_alpha = pipe.named_steps['ridge'].alpha_
    print(f"  Final model alpha (Ridge): {best_alpha:.6f}")
    return pipe, best_alpha


# PROBLEM 1: VAR1
print("\n" + "="*60)
print("PROBLEM 1 - VAR1: Power Plant Steam Turbine Optimization")
print("="*60)

feat_cols_1 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
X_train1 = train1[feat_cols_1].values
y_train1 = train1['y'].values
X_test1  = test1[feat_cols_1].values

print(f"\nSearching best degree (max=10) for var1...")
best_deg1, df_deg1 = find_best_degree(X_train1, y_train1, max_degree=10, label="var1")

print(f"\nTraining final model for var1 at degree {best_deg1}...")
model1, alpha1 = train_final_model(X_train1, y_train1, best_deg1)

y_pred_train1 = model1.predict(X_train1)
train_mse1 = mean_squared_error(y_train1, y_pred_train1)
train_r2_1 = r2_score(y_train1, y_pred_train1)
print(f"  Train MSE={train_mse1:.6f}, Train R2={train_r2_1:.6f}")

y_pred_test1 = model1.predict(X_test1)
print(f"  Test predictions - min={y_pred_test1.min():.4f}, max={y_pred_test1.max():.4f}, "
      f"mean={y_pred_test1.mean():.4f}")


# PROBLEM 2: VAR2
print("\n" + "="*60)
print("PROBLEM 2 - VAR2: Subterranean Thermal Reservoir Mapping")
print("="*60)

feat_cols_2 = ['x1', 'x2', 'x3']
X_train2 = train2[feat_cols_2].values
y_train2 = train2['y'].values
X_test2  = test2[feat_cols_2].values

print(f"\nSearching best degree (max=20) for var2...")
best_deg2, df_deg2 = find_best_degree(X_train2, y_train2, max_degree=20, label="var2")

print(f"\nTraining final model for var2 at degree {best_deg2}...")
model2, alpha2 = train_final_model(X_train2, y_train2, best_deg2)

y_pred_train2 = model2.predict(X_train2)
train_mse2 = mean_squared_error(y_train2, y_pred_train2)
train_r2_2 = r2_score(y_train2, y_pred_train2)
print(f"  Train MSE={train_mse2:.6f}, Train R2={train_r2_2:.6f}")

y_pred_test2 = model2.predict(X_test2)
print(f"  Test predictions - min={y_pred_test2.min():.4f}, max={y_pred_test2.max():.4f}, "
      f"mean={y_pred_test2.mean():.4f}")


# SAVE PREDICTION FILES
pred1_path = os.path.join(BASE, f'{ROLL}_pred_var1.csv')
pred2_path = os.path.join(BASE, f'{ROLL}_pred_var2.csv')

pd.DataFrame({'y': y_pred_test1}).to_csv(pred1_path, index=False)
pd.DataFrame({'y': y_pred_test2}).to_csv(pred2_path, index=False)
print(f"\nSaved: {pred1_path}")
print(f"Saved: {pred2_path}")


# PLOTS
fig = plt.figure(figsize=(18, 14))
fig.suptitle('ML Assignment 1 - Polynomial Regression Results (BT2024103)',
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Row 0: Degree selection curves
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(df_deg1['degree'], df_deg1['cv_mse'], 'o-', color='steelblue', lw=2)
ax0.fill_between(df_deg1['degree'],
                 df_deg1['cv_mse'] - df_deg1['cv_mse_std'],
                 df_deg1['cv_mse'] + df_deg1['cv_mse_std'],
                 alpha=0.2, color='steelblue')
ax0.axvline(best_deg1, color='red', ls='--', label=f'Best degree={best_deg1}')
ax0.set_xlabel('Polynomial Degree')
ax0.set_ylabel('CV MSE')
ax0.set_title('Var1: Degree Selection (CV MSE)')
ax0.legend(fontsize=9)
ax0.grid(alpha=0.3)

ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(df_deg2['degree'], df_deg2['cv_mse'], 'o-', color='darkorange', lw=2)
ax1.fill_between(df_deg2['degree'],
                 df_deg2['cv_mse'] - df_deg2['cv_mse_std'],
                 df_deg2['cv_mse'] + df_deg2['cv_mse_std'],
                 alpha=0.2, color='darkorange')
ax1.axvline(best_deg2, color='red', ls='--', label=f'Best degree={best_deg2}')
ax1.set_xlabel('Polynomial Degree')
ax1.set_ylabel('CV MSE')
ax1.set_title('Var2: Degree Selection (CV MSE)')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(df_deg1['degree'], df_deg1['cv_r2'], 's--', color='steelblue', label='Var1', lw=2)
ax2.plot(df_deg2['degree'], df_deg2['cv_r2'], 's--', color='darkorange', label='Var2', lw=2)
ax2.axvline(best_deg1, color='steelblue', ls=':', alpha=0.7)
ax2.axvline(best_deg2, color='darkorange', ls=':', alpha=0.7)
ax2.set_xlabel('Polynomial Degree')
ax2.set_ylabel('CV R2')
ax2.set_title('R2 vs Degree (Both Problems)')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Row 1: Actual vs Predicted
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(y_train1, y_pred_train1, alpha=0.4, s=10, color='steelblue')
lims = [min(y_train1.min(), y_pred_train1.min()),
        max(y_train1.max(), y_pred_train1.max())]
ax3.plot(lims, lims, 'r--', lw=1.5)
ax3.set_xlabel('Actual y')
ax3.set_ylabel('Predicted y')
ax3.set_title(f'Var1: Actual vs Predicted\nTrain MSE={train_mse1:.4f}, R2={train_r2_1:.4f}')
ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(y_train2, y_pred_train2, alpha=0.4, s=10, color='darkorange')
lims2 = [min(y_train2.min(), y_pred_train2.min()),
         max(y_train2.max(), y_pred_train2.max())]
ax4.plot(lims2, lims2, 'r--', lw=1.5)
ax4.set_xlabel('Actual y')
ax4.set_ylabel('Predicted y')
ax4.set_title(f'Var2: Actual vs Predicted\nTrain MSE={train_mse2:.4f}, R2={train_r2_2:.4f}')
ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(gs[1, 2])
ax5.bar(df_deg1['degree'], np.log10(df_deg1['n_features'] + 1),
        color='steelblue', alpha=0.6, label='Var1')
ax5.bar(df_deg2['degree'], np.log10(df_deg2['n_features'] + 1),
        color='darkorange', alpha=0.6, label='Var2')
ax5.set_xlabel('Polynomial Degree')
ax5.set_ylabel('log10(# Features)')
ax5.set_title('Feature Space Size vs Degree')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)

# Row 2: Residuals
residuals1 = y_train1 - y_pred_train1
residuals2 = y_train2 - y_pred_train2

ax6 = fig.add_subplot(gs[2, 0])
ax6.scatter(y_pred_train1, residuals1, alpha=0.4, s=10, color='steelblue')
ax6.axhline(0, color='red', ls='--', lw=1.5)
ax6.set_xlabel('Predicted y')
ax6.set_ylabel('Residual')
ax6.set_title('Var1: Residuals vs Predicted')
ax6.grid(alpha=0.3)

ax7 = fig.add_subplot(gs[2, 1])
ax7.scatter(y_pred_train2, residuals2, alpha=0.4, s=10, color='darkorange')
ax7.axhline(0, color='red', ls='--', lw=1.5)
ax7.set_xlabel('Predicted y')
ax7.set_ylabel('Residual')
ax7.set_title('Var2: Residuals vs Predicted')
ax7.grid(alpha=0.3)

ax8 = fig.add_subplot(gs[2, 2])
ax8.hist(residuals1, bins=40, color='steelblue', alpha=0.5, label='Var1', density=True)
ax8.hist(residuals2, bins=40, color='darkorange', alpha=0.5, label='Var2', density=True)
ax8.axvline(0, color='black', lw=1.5, ls='--')
ax8.set_xlabel('Residual')
ax8.set_ylabel('Density')
ax8.set_title('Residual Distributions')
ax8.legend(fontsize=9)
ax8.grid(alpha=0.3)

plot_path = os.path.join(BASE, 'ml_assignment_plots.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved plot: {plot_path}")
plt.close()

# SUMMARY
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"\nVar1 (Steam Turbine Optimization):")
print(f"  Best Polynomial Degree : {best_deg1}")
print(f"  Ridge Alpha            : {alpha1:.6f}")
print(f"  Train MSE              : {train_mse1:.6f}")
print(f"  Train R2               : {train_r2_1:.6f}")
n_feat1 = PolynomialFeatures(degree=best_deg1).fit(X_train1).n_output_features_
print(f"  # Polynomial Features  : {n_feat1}")

print(f"\nVar2 (Thermal Reservoir Mapping):")
print(f"  Best Polynomial Degree : {best_deg2}")
print(f"  Ridge Alpha            : {alpha2:.6f}")
print(f"  Train MSE              : {train_mse2:.6f}")
print(f"  Train R2               : {train_r2_2:.6f}")
n_feat2 = PolynomialFeatures(degree=best_deg2).fit(X_train2).n_output_features_
print(f"  # Polynomial Features  : {n_feat2}")

print("\nDone! All files saved.")
