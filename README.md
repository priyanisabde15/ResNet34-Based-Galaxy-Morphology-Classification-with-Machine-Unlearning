# ResNet34-Based Galaxy Morphology Classification with Machine Unlearning

This repository contains the final project version of our Galaxy Zoo 2 classifier and machine unlearning pipeline. It is centered on a ResNet34 backbone, real Galaxy Zoo 2 data, saved checkpoints, logged experiments, and a Streamlit dashboard for classification and result visualization.

## Final Repo Scope

This project is about two linked tasks:
- galaxy morphology classification using a pretrained ResNet34
- machine unlearning to reduce the effect of intentionally mislabeled training samples

The three output classes are:
- `Smooth`
- `Featured/Disk`
- `Artifact`

## What Is True In This Repo

- Backbone used in the final setup: `ResNet34`
- Dataset used: real `Galaxy Zoo 2`
- Image count matched to CSV labels: `61,578`
- Final project classes: `Smooth`, `Featured/Disk`, `Artifact`
- Training and unlearning were run on a consumer GPU setup based on `RTX 3050 Laptop GPU (4 GB VRAM)`
- Saved final classifier checkpoint: `best_model.pth`
- Saved unlearning outputs: `ga_unlearned_model.pth`, `ff_unlearned_model.pth`, `retrained_model.pth`

## Latest Saved Results In This Workspace

### Training

From `training_log.csv`:
- Best validation accuracy: `85.58%`
- Best balanced accuracy: `87.56%`
- Best macro-F1: `76.41%`
- Epochs completed in the saved run: `19`

### Unlearning

From `unlearning_results.csv`:

| Method | Test Accuracy | Forget Accuracy | Retain Accuracy | Time (s) |
|---|---:|---:|---:|---:|
| Gradient Ascent | 70.80 | 11.01 | 76.39 | 103.57 |
| Fisher Forgetting | 41.71 | 28.36 | 43.44 | 213.76 |
| Full Retrain | 76.17 | 8.36 | 99.64 | 2649.43 |

## Important File Roles

- `train.py`: final training pipeline used for the ResNet34 classifier
- `unlearn.py`: standalone unlearning pipeline that runs from a saved checkpoint
- `model.py`: ResNet model definition and checkpoint utilities
- `data_loader.py`: Galaxy Zoo 2 loading, transforms, and mislabel injection
- `evaluate.py`: evaluation metrics and comparison logic
- `visualize.py`: plots and result visualizations
- `app.py`: Streamlit dashboard
- `FINAL_PROJECT_FACTS.md`: one-file factual summary of the final project state

## Recommended Usage

### 1. Run the app using the saved model

```bash
streamlit run app.py
```

### 2. Train again only if you want a new experiment

```bash
python train.py --device auto --backbone resnet34
```

### 3. Run unlearning from the saved checkpoint

```bash
python unlearn.py --device auto --checkpoint best_model.pth
```

## Notes Before Final Commit

- This repo is now cleaned around the final `ResNet34` project version.
- The most reliable result files are `training_log.csv` and `unlearning_results.csv`.
- Final plots in `plots/` are included so the downloaded ZIP shows the saved visual outputs directly.
- `FINAL_PROJECT_FACTS.md` should be treated as the current source-of-truth summary for the project.

## License

See `LICENSE`.
