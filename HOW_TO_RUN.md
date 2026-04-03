# How To Run The Final Project

This guide reflects the cleaned final project state.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Ensure the real dataset exists

Expected structure:

```text
data/
??? images_training_rev1/
??? training_solutions_rev1.csv
```

The final project uses the real Galaxy Zoo 2 dataset. Synthetic fallback should not be treated as part of the final workflow.

## 3. Run the app directly using saved artifacts

If you only need the demo/application flow:

```bash
streamlit run app.py
```

This uses the saved model and saved CSV outputs already present in the project folder.

## 4. Train a fresh classifier only if needed

```bash
python train.py --device auto --epochs 50 --batch-size 16 --lr 2e-4 --backbone resnet34 --patience 12
```

Use this when you want a new experiment, not when you only want to demo the project.

## 5. Run unlearning from the saved checkpoint

```bash
python unlearn.py --device auto --checkpoint best_model.pth
```

## 6. Files you should look at after a run

- `best_model.pth`
- `training_log.csv`
- `unlearning_results.csv`
- `plots/`

## Final note

For the exact saved run details used in this repo, refer to `FINAL_PROJECT_FACTS.md`.
