# Galaxy Morphology Unlearning Dashboard Guide

## 1. What this file is

This document explains how to open the unlearning dashboard in the project app, what machine unlearning is in our context, why it is core to this project, and how to interpret results.

---

## 2. How to launch the web app (dashboard)

1. Open command prompt / terminal in your project folder.
2. Activate your virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
3. Install dependencies if you haven’t already:
   - `pip install -r requirements.txt`
4. Start the app:
   - `streamlit run app.py`
5. In your browser, Streamlit opens a local address (usually `http://localhost:8501`).

You should see the app interface with controls and result panels.

---

## 3. Where to find Unlearning in the app

The dashboard has a section called “Unlearning Pipeline” or similar. It contains a button:

- `Run Unlearning Pipeline`

The app workflow is:

1. Load current model data via `get_data_loaders()`.
2. Inject mislabels into the training set via `inject_mislabels()`.
3. Split data into `forget` and `retain` sets for unlearning evaluation.
4. Evaluate baseline model on test/forget/retain sets.
5. Perform methods: 
   - `Gradient Ascent Unlearning`
   - `Fisher Forgetting`
6. Show results including test/forget/retain and time.

---

## 4. What unlearning means in this project

### Machine unlearning core idea

Machine unlearning means modifying a model so it *forgets specific examples* while keeping good accuracy on the rest. In our project we use it to simulate removing wrongly labeled galaxies and measuring the effect.

### Specific unlearning behavior implemented:

- **Inject mislabels**: Some training entries are deliberately swapped (12% by default).
- **Forget set**: images with synthetic mislabels.
- **Retain set**: correctly labeled images.

Then the methods aim to lower accuracy on the forget set (bad data should be forgotten) while keeping high retain/test accuracy.

---

## 5. Why unlearning is core to the project

- The project name explicitly references machine unlearning.
- It’s a research experiment in: "can a model unlearn a subset of data fast, instead of full retrain?"
- Evaluation compares:
  - Baseline (no unlearning)
  - Gradient Ascent unlearning
  - Fisher Forgetting
  - Full retrain (ground truth)

This is the key novelty and contribution of your paper.

---

## 6. What output and metrics to look at

Main metrics (in `unlearning_results.csv` and dashboard):

- `test_accuracy`: accuracy on clean test set
- `forget_accuracy`: accuracy on mislabel forgetting set (lower is better for unlearning)
- `retain_accuracy`: accuracy on correctly labeled retain set (higher is better)
- `computation_time_seconds`: speed / efficiency
- `test_loss`, `forget_loss`, `retain_loss`: cross-entropy losses

Interpretation:

- Good unlearning method: low forget acc + high retain/test acc.
- Bad unlearning: high forget acc (still remembers bad labels) or low retain acc (unlearning destroyed good knowledge).

---

## 7. Easy terms

- **Baseline**: model before any unlearning intervention.
- **Forget set**: intentionally corrupted data examples to forget.
- **Retain set**: clean examples to keep.
- **Gradient Ascent Unlearning**: model update approach that reduces confidence on forget set.
- **Fisher Forgetting**: uses Fisher information to reduce dependence on forget parameters.
- **Full Retrain**: retraining from scratch without corrupted data (ideal reference).

---

## 8. Quick check before sharing / paper

1. Run full training and confirm baseline accuracy in `training_log.csv`.
2. Run unlearning pipeline in app and/or `main.py`.
3. Export `unlearning_results.csv` and include the table in your paper.
4. Emphasize the ‘forget vs retain tradeoff’ in conclusions.
