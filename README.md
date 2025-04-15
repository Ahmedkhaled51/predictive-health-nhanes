# NHANES Analysis Summary

This README provides a structured summary of findings, model diagnostics, and recommendations based on the NHANES dataset analysis.

---

## 1. Key Regression Model Results

### **BMI Model (Ordinary Least Squares - OLS)**

- **Variables Used:** `height`, `weight`
- **R-squared:** ~1.0 (indicates a perfect fit, likely due to multicollinearity)

**Coefficients:**
- `height`: -0.3013 (significant, *p* = 0.000)
- `weight`: 0.3561 (significant, *p* = 0.000)

**Issues Identified:**
- **Multicollinearity:** High condition number (> 4900). Strong correlation between height and weight.
- **Heteroscedasticity:** 
  - Breusch-Pagan test *p* ≈ 0 (rejects homoscedasticity).
  - Residual plots show non-constant variance.

---

### **Weighted Least Squares (WLS) Model**

- **Transformation:** `log(bmi) ~ log(height) + log(weight)`
- **R-squared:** ~0.999

**Diagnostics:**
- Heteroscedasticity still present (Breusch-Pagan *p* ≈ 7.6e-23)
- Transformation improves linearity but doesn’t fully resolve variance issues.

---

### **Blood Pressure Model (Logistic Regression)**

- **Variables Used:** `bpsystol`, `bpdiast`
- **Outcome:** High blood pressure (HBP)
- **Results:** Both variables significantly associated with HBP.

---

## 2. Model Diagnostics

### **Assumptions Check**

- **Normality:**
  - QQ plots show deviation (e.g., residual skewness = 0.684)
- **Linearity:**
  - Log transformations improved model form
- **Homoscedasticity:**
  - Breusch-Pagan consistently rejects equal variance
  - **Recommendation:** Use robust standard errors or alternative models (e.g., GLS)
- **Independence:**
  - Durbin-Watson ≈ 2 (suggests no autocorrelation)

### **Influential Observations**

- **Cook’s Distance:** Most points within acceptable range
- **Leverage:** No high-leverage points found (`3*(k+1)/n` threshold)

---

## 3. Data Cleaning & Preprocessing

- **Missing Values:**
  - High: `lead`, `tgresult`, `fhtatk` (~50%) → consider imputation or exclusion
  - Low: `hlthstat`, `heartatk`, `diabetes` (~0.02%)

- **Categorical Variables:**
  - Encoded `sex`, `race`, `psu`
  - Check for typos (e.g., `orace` vs. `race`)

- **Outlier Handling:**
  - Filtered using 10th–90th percentile
  - Confirm data retention after filtering

---

## 4. Recommendations

### **Addressing Multicollinearity**

- **Option 1:** Drop a predictor (e.g., `weight`)
- **Option 2:** Apply ridge regression

---

### **Fixing Heteroscedasticity**

- **Use Robust Standard Errors:**

```python
bmi_model_robust = sm.OLS.from_formula('bmi ~ height + weight', data=nhanes).fit(cov_type='HC3')
```

- **Try Other Transformations:** e.g., `sqrt(bmi)`, `1/bmi`

---

### **Predictive Modeling for Diabetes**

- **Logistic Regression:**

```python
diabetes_model = smf.logit('diabetes ~ age + bmi + sex + race', data=nhanes).fit()
```

- **Feature Importance:** Evaluate coefficients or use permutation importance

---

### **Descriptive Analysis**

```python
print(nhanes[['weight', 'height', 'age']].mean())
sns.histplot(nhanes['region'], discrete=True, shrink=0.8)
```

---

### **Code Improvements**

- **Fix Typos:** e.g., `depnedant variabels` → `dependent variables`
- **Verify Column Names:** (e.g., `hct` vs. `HCT`)
- **Weight Calculation:**

```python
waight = 1 / (bmi_model.resid**2 + 1e-6)  # Prevent division by zero
```

---

## 5. Next Steps

1. **Complete Data Cleaning:** Handle missing and inconsistent values
2. **Re-run Models:** With robust estimators or transformed variables
3. **Generate Visuals:** Residual plots, histograms, correlation heatmaps
4. **Report Results:** Emphasize strong predictors and relationships

---
