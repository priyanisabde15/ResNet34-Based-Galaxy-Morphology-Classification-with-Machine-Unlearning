# Project Summary

## Final Project Identity

- Project name: `ResNet34-Based Galaxy Morphology Classification with Machine Unlearning`
- Domain: astronomical image classification + machine unlearning
- Dataset: `Galaxy Zoo 2`
- Final backbone: `ResNet34`
- Final classes: `Smooth`, `Featured/Disk`, `Artifact`

## Final Project Outcome

This project trains a ResNet34-based classifier on real Galaxy Zoo 2 images and then studies how machine unlearning methods behave when a subset of training samples is intentionally mislabeled.

## Final Components

- `data_loader.py`: real Galaxy Zoo 2 loading and split logic
- `model.py`: ResNet model definition and checkpoint handling
- `train.py`: final training script for the classifier
- `unlearn.py`: unlearning pipeline from a saved checkpoint
- `evaluate.py`: metric calculation
- `visualize.py`: plots and charts
- `app.py`: Streamlit dashboard

## Final Saved Training Results

From `training_log.csv`:
- Best validation accuracy: `85.58%`
- Best balanced accuracy: `87.56%`
- Best macro-F1: `76.41%`
- Saved run length: `19 epochs`

## Final Saved Unlearning Results

From `unlearning_results.csv`:
- Gradient Ascent: test `70.80%`, forget `11.01%`, retain `76.39%`, time `103.57 s`
- Fisher Forgetting: test `41.71%`, forget `28.36%`, retain `43.44%`, time `213.76 s`
- Full Retrain: test `76.17%`, forget `8.36%`, retain `99.64%`, time `2649.43 s`

## Final Repo Guidance

This repository should be treated as a `ResNet34` project repo with saved outputs and a working app. The main factual reference inside the repo is `FINAL_PROJECT_FACTS.md`.
