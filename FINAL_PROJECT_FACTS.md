# Final Project Facts

This file is the factual, low-theory summary of the final project state.

## 1. Project Identity

- Project title: `ResNet34-Based Galaxy Morphology Classification with Machine Unlearning`
- Dataset: `Galaxy Zoo 2`
- Final backbone: `ResNet34`
- Final classes:
  - `Smooth`
  - `Featured/Disk`
  - `Artifact`

## 2. Dataset Facts

- Image folder used by loader: `data/images_training_rev1/`
- CSV used by loader: `data/training_solutions_rev1.csv`
- Total images matched to CSV labels: `61,578`

Mapped class counts in the current dataset:
- `Smooth: 26,693`
- `Featured/Disk: 34,826`
- `Artifact: 59`

## 3. Final Training Setup In Code

From the current final training pipeline:
- Script: `train.py`
- Optimizer: `AdamW`
- Loss: `FocalLoss` by default
- Input size: `224 x 224`
- Data split: `70 / 15 / 15`
- Random seed: `42`
- Mixed precision on CUDA: `enabled`
- Early stopping: `enabled`
- Best checkpoint path: `best_model.pth`

Important note:
- The current code defaults and the saved successful run are not identical in every numeric detail.
- The saved run values below come from the actual CSV logs already present in the repo.

## 4. Saved Training Run Facts

From `training_log.csv`:
- Epochs completed in the saved run: `19`
- Best validation accuracy: `85.5797%`
- Best balanced accuracy across the run: `87.5616%`
- Best macro-F1 across the run: `76.4143%`

Best validation-accuracy row:
- Epoch: `19`
- Train loss: `0.000818`
- Val loss: `0.005620`
- Val accuracy: `85.5797%`
- Val balanced accuracy: `79.3356%`
- Val macro-F1: `75.1485%`
- Learning rate at that epoch: `0.000030`

## 5. Saved Unlearning Run Facts

From `unlearning_results.csv`:

### Gradient Ascent
- Test accuracy: `70.8022%`
- Forget accuracy: `11.0128%`
- Retain accuracy: `76.3869%`
- Time: `103.5737 s`
- GPU memory: `262.50 MB`

### Fisher Forgetting
- Test accuracy: `41.7127%`
- Forget accuracy: `28.3586%`
- Retain accuracy: `43.4358%`
- Time: `213.7627 s`
- GPU memory: `425.87 MB`

### Full Retrain
- Test accuracy: `76.1719%`
- Forget accuracy: `8.3586%`
- Retain accuracy: `99.6418%`
- Time: `2649.4344 s`
- GPU memory: `589.10 MB`

## 6. Method Files

- Training pipeline: `train.py`
- Standalone unlearning pipeline: `unlearn.py`
- Model definition: `model.py`
- Data pipeline: `data_loader.py`
- App/dashboard: `app.py`

## 7. Saved Models Present In This Repo

- `best_model.pth`
- `best_model_backup_before_retrain.pth`
- `ga_unlearned_model.pth`
- `ff_unlearned_model.pth`
- `retrained_model.pth`

## 8. Plots Present In This Repo

Current generated plots:
- `training_curves.png`
- `accuracy_comparison.png`
- `forget_accuracy.png`
- `computation_time.png`
- `confusion_matrix_before.png`
- `confusion_matrix_after.png`
- `confusion_matrix_before_after.png`
- `project_pipeline_flowchart.png`
- `cnn_morphology_pipeline.png`
- `unlearning_benefits_flowchart.png`
- `unlearning_methods_flowchart.png`

## 9. Practical Demo Path

If you only need to present the project:
1. Keep the saved checkpoint and CSV files
2. Run `streamlit run app.py`
3. Use the dashboard tabs and saved result plots

## 10. Remaining Numerical Standardization Tasks

These should be unified next before the final paper/repo lock:
- choose one official mislabel ratio for the final release: `10%` or `12%`
- choose one official training configuration summary for the paper
- remove or archive any legacy result references that still point to the old ~33% run
- make the paper dataset counts match the actual final dataset counts in this repo

Until those are standardized, this file should be used as the most reliable factual snapshot of the current final project state.
