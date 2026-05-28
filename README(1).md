# 🏦 Bayesian Logistic Regression for Credit Card Default Prediction

> *Can we predict who will default — and how sure are we about that prediction?*

**Dataset:** Default of Credit Card Clients · UCI Machine Learning Repository  
**Source:** Taiwan credit card data, October 2005 · 30,000 clients  
**Task:** Binary classification — predict whether a client will default next month  
**Method:** Bayesian Logistic Regression via MCMC (NUTS sampler in PyMC)

---

## 📌 Table of Contents

1. [Project Motivation](#-project-motivation)
2. [Dataset Overview](#-dataset-overview)
3. [Exploratory Data Analysis](#-exploratory-data-analysis)
4. [Preprocessing](#-preprocessing)
5. [Frequentist Baselines](#-frequentist-baselines)
6. [Bayesian Model](#-bayesian-model)
7. [Posterior Analysis](#-posterior-analysis)
8. [Model Comparison & ROC Curves](#-model-comparison--roc-curves)
9. [Calibration Analysis](#-calibration-analysis)
10. [Posterior Predictive Checks](#-posterior-predictive-checks)
11. [Prior Sensitivity Analysis](#-prior-sensitivity-analysis)
12. [Decision Threshold Policy](#-decision-threshold-policy)
13. [Key Takeaways](#-key-takeaways)
14. [Tech Stack](#-tech-stack)

---

## 💡 Project Motivation

Most credit scoring models give one number — but that number alone does not tell us how sure the model is.

A standard logistic regression might say *"this client has a 28% chance of default."* A Bayesian model gives the same answer, and also tells you whether that 28% is a confident estimate or a rough guess. In lending, that difference matters a lot.

This project builds a full Bayesian Logistic Regression pipeline to:

- **Match** the performance of standard and tree-based models
- **Measure uncertainty** on every prediction using the posterior standard deviation
- **Check model fit** through posterior predictive checks
- **Turn** posterior probabilities into a clear, tiered credit decision policy

---

## 📊 Dataset Overview

**Reference:** Yeh, I. C., & Lien, C. H. (2009). *Expert Systems with Applications*, 36(2), 2473–2480.  
**Link:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The dataset has **30,000 clients** and 23 features in three groups.

### Feature Groups

| Group | Variables | Description |
|-------|-----------|-------------|
| Demographics | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Client background |
| Credit | `LIMIT_BAL` | Credit limit (NT$) |
| Repayment history | `PAY_1` – `PAY_6` | Monthly payment status, Sep → Apr 2005 |
| Bill amounts | `BILL_AMT1` – `BILL_AMT6` | Monthly bill balances |
| Payment amounts | `PAY_AMT1` – `PAY_AMT6` | Actual payments made each month |

**Repayment status codes:**

| Code | Meaning |
|:----:|---------|
| `-2` | No activity (dormant account) |
| `-1` | Paid in full |
| `0`  | Carried a balance, no delay |
| `1–9` | Months of late payment |

> ⚠️ The original dataset labels September's column `PAY_0`. We rename it `PAY_1` to avoid confusion.

**Target:** `default` — 1 if the client defaulted next month, 0 if not. Overall default rate: **22.1%**

---

## 🔍 Exploratory Data Analysis

### Signal Check: Two Key Claims

Before building any model, we checked two simple claims to confirm the data has real, usable signal.

**Claim 1 — Late payment leads to default**

| PAY_1 | Meaning | Default Rate |
|:-----:|---------|:------------:|
| -2 | No activity | 13.5% |
| 0  | Carried balance, on time | **12.8% ← Lowest** |
| 2  | 2 months late | ~70% |
| 8+ | Very late | >75% |
| — | **Overall** | **22.1%** |

✅ **Confirmed:** One missed payment more than triples default risk. `PAY_1` is the strongest predictor in the whole dataset.

**Claim 2 — Clients with no activity are low risk**

Clients who made no purchases in all 6 months defaulted at only **13.5%** — well below the 22.1% average.

✅ **Confirmed — with a note:** The lowest-risk group is actually `PAY_1 = 0` (clients who carried a balance but paid on time, 12.8%). Both groups are low risk, but for different reasons.

### Age & Credit Limit Patterns

- **Age:** Young clients (20–25) default more often. The 40–50 age group is the most stable.
- **Credit limit:** Clients with a low credit limit (below 50K NT) default at about 3× the rate of clients with a high limit (above 500K NT). This is because banks only give large credit limits to clients who are already known to be reliable.

---

## ⚙️ Preprocessing

A few values in the data were not explained in the original paper:

- `EDUCATION` had codes `0, 5, 6` → merged into category `4` (*others*)
- `MARRIAGE` had code `0` → merged into category `3` (*others*)
- All features were scaled using `StandardScaler` so that MCMC can run well and coefficients can be compared directly

**Final split:** 24,000 training · 6,000 test · 27 features · same default rate in both sets

---

## 📏 Frequentist Baselines

We built two standard models before the Bayesian model:

| Model | Purpose |
|-------|---------|
| **Logistic Regression** | Direct comparison — same model type, but fitted the standard way |
| **XGBoost** | Upper bound — shows the best AUC a non-linear model can reach |

---

## 🔮 Bayesian Model

### Model Specification

$$y_i \sim \text{Bernoulli}(p_i)$$

$$\text{logit}(p_i) = \alpha + \mathbf{x}_i^\top \boldsymbol{\beta}$$

$$\alpha \sim \mathcal{N}(0,\ 2.5), \qquad \beta_j \sim \mathcal{N}(0,\ 1)$$

### Why These Priors

All features are scaled. A coefficient of ±2 already means the odds of default change by ~7×, which is a very large effect. `N(0, 1)` is a **weakly informative prior** — it keeps the model away from unrealistic values without pushing estimates toward zero.

### Sampling Setup

- **Sampler:** NUTS via PyMC
- **Setup:** 4 chains × 1,000 draws + 1,000 tuning steps, `target_accept = 0.9`
- **Runtime:** About 10–20 minutes on a normal laptop for the full dataset

---

## 📐 Posterior Analysis

### Convergence Check

| Diagnostic | Threshold | Result |
|:----------|:---------:|:------:|
| R̂ (Gelman-Rubin) | < 1.01 | ✅ All parameters |
| ESS Bulk | > 400 | ✅ All parameters |
| Trace plots | Stable, well-mixed | ✅ Confirmed |

All 4 chains reached the same posterior — the results are reliable.

### Posterior Coefficient Forest Plot

Each bar shows a coefficient's posterior mean and 94% HDI (Highest Density Interval):

- 🔴 **Red** — risk factors: higher value → more likely to default
- 🟢 **Green** — protective factors: higher value → less likely to default

**Main findings:**
- `PAY_1` has the largest positive coefficient by far
- `LIMIT_BAL` has a strong negative coefficient — a higher credit limit reduces risk
- Earlier months (`PAY_2`–`PAY_6`) show smaller but consistent effects
- Demographic features (age, sex, education) have small and uncertain effects

---

## 📈 Model Comparison & ROC Curves

### Performance Table

| Model | AUC ↑ | Brier Score ↓ | ECE ↓ |
|:------|:-----:|:-------------:|:-----:|
| Logistic Regression | 0.7099 | 0.1466 | 0.0492 |
| XGBoost | **0.7771** | **0.1355** | **0.0088** |
| **Bayesian LR** | 0.7101 | 0.1466 | 0.0490 |

- **AUC** — how well the model ranks high-risk clients above low-risk ones (higher = better)
- **Brier Score** — average error on predicted probabilities (lower = better)
- **ECE** — gap between what the model predicts and what actually happens (lower = better)

**Key point:** Bayesian LR and standard Logistic Regression score almost the same on all three metrics. The Bayesian approach costs nothing in performance, but gives us one extra thing: a measure of uncertainty for every single prediction. XGBoost scores higher on AUC due to its non-linear structure, but cannot give any uncertainty estimate.

---

## 📐 Calibration Analysis

AUC measures ranking. Calibration measures **probability accuracy**: when the model says "30% default risk," about 30% of those clients should actually default. In lending, a model that is not well calibrated will misprice risk.

The **reliability diagram** plots predicted probability against the actual default rate per group. A perfect model follows the diagonal line exactly.

XGBoost has the best ECE (0.0088), but gives no uncertainty per prediction. Bayesian LR (ECE 0.0490) is close in calibration and also provides a posterior standard deviation for every client.

---

## 🔄 Posterior Predictive Checks

A posterior predictive check (PPC) asks: *if we generate new data from the fitted model, does it look like the real data?*

**Check 1 — Overall default rate**  
We generated 4,000 fake datasets from the posterior. The default rates in these fake datasets should be centered around the real rate of ~22%. ✅ Passed.

**Check 2 — Default rate by PAY_1 group**

| Group | Observed Rate | PPC Result |
|-------|:-------------:|:----------:|
| On-time (PAY_1 ≤ 0) | ~13% | ✅ Matched |
| Mild delay (1–2 months) | ~40–50% | ✅ Matched |
| Severe delay (3+ months) | ~70%+ | ✅ Matched |

The model learned the structure of risk across groups, not just the overall average.

---

## 🔁 Prior Sensitivity Analysis

A good Bayesian model should give the same results no matter which reasonable prior we choose — the data should do most of the work.

We re-fitted the model with three different priors and compared the results:

| Prior SD | Effect |
|:--------:|--------|
| 0.5 | Tighter, more conservative |
| **1.0** | **Used in our final model** |
| 2.5 | Broader, more open |

**Result:** The posterior means are almost the same across all three. With 24,000 training samples, the data takes over — our results are not driven by our prior choice.

---

## 🎯 Decision Threshold Policy

A credit model's final job is to make a **decision**, not just give a number. We turn the posterior into a four-tier policy:

| Tier | Condition | Action |
|:-----|:----------|:-------|
| ✅ **Auto-approve** | `P(default) < 16.7%` | Approve, standard terms |
| 🔍 **Manual review** | `16.7% ≤ P ≤ 30%` | A person reviews the case |
| ❌ **Auto-reject** | `P(default) > 30%` | Decline or offer a secured product |
| ⚠️ **Uncertainty flag** | `posterior std > 0.05` | Send to review no matter what the mean says |

### Why the Uncertainty Tier Matters

This fourth tier is only possible with Bayesian inference. A client with `P(default) = 14%` but `std = 0.08` would be auto-approved by any standard model — but the posterior tells us the model is not confident about this client. Sending such cases to a human reviewer is a risk control step that no point-estimate model can offer.

---

## 🏆 Key Takeaways

| # | Finding |
|---|---------|
| 1 | **PAY_1 is the strongest predictor** — being 2 months late alone pushes default risk to ~70% |
| 2 | **Bayesian LR matches standard LR** on all metrics (AUC 0.7101 vs 0.7099) — no loss in performance |
| 3 | **Uncertainty is useful** — posterior std enables a fourth decision tier that no point-estimate model can provide |
| 4 | **PPC confirms model fit** — simulated data matches real data both overall and within risk groups |
| 5 | **Results are stable across priors** — 24K samples mean the data drives the conclusions, not the prior |
| 6 | **Credit limit is protective** — banks pre-screen before giving high limits, so high-limit clients are already reliable |

---

## 🛠 Tech Stack

| Library | Role |
|---------|------|
| `PyMC` | Probabilistic programming, NUTS sampler |
| `ArviZ` | Posterior diagnostics, trace plots, HDI |
| `scikit-learn` | Preprocessing, baselines, calibration |
| `XGBoost` | Non-linear benchmark |
| `pandas` / `numpy` | Data handling |
| `matplotlib` / `seaborn` | Charts |

---

## 📚 References

- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020
- UCI ML Repository: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Gelman et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman & Hall/CRC.
- PyMC: https://www.pymc.io · ArviZ: https://python.arviz.org

---

*Built with PyMC · Week 10 Final Project*
