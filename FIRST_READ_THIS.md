# FIRST READ THIS

This is the first file you should read before approaching the project.

It explains the exact basic order to follow:
1. Add your Kaggle API key
2. Download and prepare the dataset
3. Train the model
4. Run unlearning
5. Launch the Streamlit app

## Step 1: Get Your Kaggle API Key

1. Log in to your Kaggle account.
2. Open `Account`.
3. Scroll to the `API` section.
4. Click `Create New Token`.
5. Kaggle will download a file called `kaggle.json`.

Place that file in this project folder.

If you are creating it manually, the file should look like this:

```json
{
  "username": "YOUR_KAGGLE_USERNAME",
  "key": "YOUR_KAGGLE_API_KEY"
}
```

## Step 2: Download the Dataset

After adding `kaggle.json`, download and set up the Galaxy Zoo 2 dataset.

Use:

```bash
python setup_dataset.py
```

If needed, you can also use:

```bash
python download_dataset.py
```

Important:
- `data_loader.py` loads and processes the dataset for training
- the actual download/setup step is handled by `setup_dataset.py` or `download_dataset.py`

After setup, your folder should contain:

```text
data/
??? images_training_rev1/
??? training_solutions_rev1.csv
```

## Step 3: Train the Model

Run:

```bash
python train.py --device auto --epochs 30 --batch-size 16 --lr 3e-4 --backbone resnet34 --patience 12
```

This will train the ResNet34 classifier and save the best checkpoint.

Main output files:
- `best_model.pth`
- `training_log.csv`

## Step 4: Run Unlearning

After training is complete, run:

```bash
python unlearn.py --device auto --checkpoint best_model.pth
```

This will run:
- Gradient Ascent
- Fisher Forgetting
- Full Retrain

Main output file:
- `unlearning_results.csv`

## Step 5: Run the Streamlit App

After training and unlearning are complete, launch the app:

```bash
streamlit run app.py
```

## Recommended Order

Always follow this order:

1. `FIRST_READ_THIS.md`
2. `setup_dataset.py`
3. `train.py`
4. `unlearn.py`
5. `app.py`

## Final Note

If someone downloads this project as a ZIP, this is the first file they should read before running anything.
