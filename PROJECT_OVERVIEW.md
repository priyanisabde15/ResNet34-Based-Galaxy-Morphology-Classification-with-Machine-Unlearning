# Galaxy Morphology Classification with Machine Unlearning
### A Deep Learning Research Project using ResNet34 + Galaxy Zoo 2

---

## Latest Results

| Method | Test Accuracy | Forget Accuracy | Retain Accuracy | Time |
|---|---|---|---|---|
| Gradient Ascent | 70.80% | 11.01% | 76.39% | 103.6s |
| Fisher Forgetting | 41.71% | 28.36% | 43.44% | 213.8s |
| Full Retrain | 76.17% | 8.36% | 99.64% | 2649.4s |

> Latest training run reached `85.58%` best validation accuracy, with `87.56%` best balanced accuracy and `76.41%` best macro-F1 using the current ResNet34 setup.

---

## What This Project Is

This project applies **machine unlearning** to the problem of galaxy morphology classification. A ResNet34 convolutional neural network is trained on real galaxy images from the Galaxy Zoo 2 citizen science dataset. The model learns to classify galaxies into three types: Smooth (elliptical), Featured/Disk (spiral), and Artifact (stars/noise).

The twist: 12% of the training data is intentionally mislabeled to simulate real-world label noise. Machine unlearning algorithms are then applied to make the model forget those corrupted samples — without retraining from scratch.

This is directly relevant to modern AI privacy and data governance requirements. Laws like GDPR's "right to be forgotten" require AI systems to be able to remove the influence of specific data points from trained models. This project demonstrates three practical methods for doing exactly that.

---

## The Dataset — Galaxy Zoo 2

Galaxy Zoo 2 is a real citizen science project run by Oxford University and the Zooniverse platform. Over 300,000 volunteers manually classified 304,122 galaxy images from the Sloan Digital Sky Survey (SDSS). Each galaxy was classified by multiple volunteers, and the results were aggregated into probability scores.

The dataset contains:
- 61,578 training images (JPG, 424×424 pixels)
- `training_solutions_rev1.csv` — probability scores for each galaxy across 37 morphological questions

For this project, we use the top-level classification:
- `Class1.1` — probability the galaxy is smooth/elliptical
- `Class1.2` — probability the galaxy has features/disk (spiral)
- `Class1.3` — probability it is a star or artifact

The class with the highest probability becomes the label. This gives us a 3-class classification problem.

---

## Galaxy Classes Explained

**Class 0 — Smooth (Elliptical)**
Round, featureless galaxies. No spiral arms, no disk structure. They look like glowing blobs of light. These are older galaxies where star formation has mostly stopped. Example: Messier 87 (the galaxy with the famous black hole photo).

**Class 1 — Featured/Disk (Spiral)**
Galaxies with visible structure — spiral arms, bars, rings, clumps. These are actively forming stars. The Milky Way itself is a Featured/Disk galaxy. These are the most visually interesting and the hardest to classify correctly.

**Class 2 — Artifact**
Not a galaxy at all. These are foreground stars, imaging noise, cosmic rays, or objects that ended up in the survey by mistake. The model needs to learn to reject these.

---

## The Mislabeling Problem

In real-world datasets, label noise is unavoidable. Galaxy Zoo 2 itself has noise — volunteers sometimes disagree, and the probability scores reflect that uncertainty. To simulate this, the project intentionally mislabels 12% of training samples by randomly assigning them to a different class.

For example, a spiral galaxy (Featured) might get labeled as Smooth. The model trains on this corrupted data and learns the wrong association. The goal of machine unlearning is to fix this without throwing away the entire trained model and starting over.

---

## Machine Unlearning — The Core Research Element

Machine unlearning answers the question: *given a trained model, how do you make it forget specific training examples?*

The naive answer is to retrain from scratch without those examples. But with 61,578 images and the current fine-tuning setup, that takes substantial compute time. The research question is whether we can achieve the same result faster.

### Method 1 — Gradient Ascent Unlearning

**What it does:**
Normal training minimizes loss — it adjusts weights to make the model more correct. Gradient Ascent does the opposite on the forget set. It maximizes loss on the mislabeled samples, forcing the model to become wrong on those specific examples. This effectively erases the patterns learned from corrupted data.

**The math:**
Normal gradient descent: `θ = θ - α ∇L(θ, x_forget)`
Gradient ascent: `θ = θ + α ∇L(θ, x_forget)`

The only change is the sign. Instead of subtracting the gradient (learning), we add it (unlearning).

**In the code (`unlearn.py`):**
```python
# KEY: Negate loss for gradient ASCENT (maximize loss)
ascent_loss = -loss
scaler.scale(ascent_loss).backward()
```

**Result in latest run:** Test `70.80%`, Forget `11.01%`, Retain `76.39%`, Time `103.6s`.

---

### Method 2 — Fisher Forgetting

**What it does:**
The Fisher Information Matrix (FIM) measures how sensitive the model's predictions are to each weight. Weights with high Fisher information are important for the retain set (clean data). Weights with low Fisher information are less critical.

Fisher Forgetting adds noise to the model's weights — but the noise is calibrated: weights that are important for clean data get less noise, weights that were mainly used for the corrupted data get more noise. This selectively scrambles the memory of the mislabeled samples.

**The math:**
For each weight `θ_i`, add noise: `θ_i = θ_i + N(0, σ²/F_i)`
Where `F_i` is the Fisher information for that weight and `σ` is the noise scale.

**In the code (`unlearn.py`):**
```python
# Noise inversely proportional to Fisher information
# High Fisher = important for retain set = less noise
fisher_info = fisher_dict[name] + 1e-8
noise = torch.randn_like(param) * noise_scale / torch.sqrt(fisher_info)
param.add_(noise)
```

**Result in latest run:** Test `41.71%`, Forget `28.36%`, Retain `43.44%`, Time `213.8s`.

---

### Method 3 — Full Retrain (Gold Standard)

**What it does:**
Remove all mislabeled samples from the training set and retrain the model from scratch. This is the ground truth — it guarantees the model has never seen the corrupted data. All other methods are compared against this.

**Why it matters:**
Full retrain is the gold standard but expensive. In the latest workspace run it took `2649.4s`. The research value of Gradient Ascent and Fisher Forgetting is that they attempt to approach retraining quality at lower cost.

**Result in latest run:** Test `76.17%`, Forget `8.36%`, Retain `99.64%`.

---

## How We Know Unlearning Actually Worked

Three metrics prove unlearning happened:

**1. Forget Accuracy drops**
In the latest run:
- Gradient Ascent forget accuracy: `11.01%`
- Fisher Forgetting forget accuracy: `28.36%`
- Full Retrain forget accuracy: `8.36%`

**2. Retain Accuracy stays stable**
The model's performance on clean, correctly-labeled data should not degrade significantly. If retain accuracy crashes, the unlearning was too aggressive and damaged the model.

**3. Comparison to Full Retrain**
The closer a method's metrics are to Full Retrain, the better. Full Retrain is the theoretical upper bound — it's what perfect unlearning looks like.

The ideal unlearning result:
- Forget accuracy: as low as possible (model forgot the bad data)
- Retain accuracy: as high as possible (model kept the good data)
- Time: as fast as possible (faster than full retrain)

---

## The Streamlit Web App

The web app (`app.py`) has three tabs:

### Tab 1 — Galaxy Classifier
Upload any galaxy image (JPG or PNG) and the trained ResNet34 model classifies it in real time. The app shows:
- The predicted class (Smooth, Featured/Disk, or Artifact)
- Confidence percentage for the prediction
- A bar chart showing confidence scores for all three classes
- Color-coded result card (blue for Smooth, green for Featured, red for Artifact)

When you upload a galaxy image, the app runs it through the same preprocessing pipeline used during training (resize to 224×224, normalize with ImageNet mean/std), then passes it through the model. The output is a softmax probability distribution across the 3 classes.

### Tab 2 — Unlearning Dashboard
This is where you see the unlearning in action. The dashboard:
- Explains what each of the 3 unlearning methods does
- Shows a "Run Unlearning Pipeline" button that executes all 3 methods live
- Displays a comparison table: test accuracy, forget accuracy, retain accuracy, and computation time for each method
- Shows a grouped bar chart comparing all methods side by side
- Shows a computation time chart so you can see the speed tradeoff

The key thing to look for: after running unlearning, the forget accuracy column should be lower than the baseline (No Unlearning row). That's the proof that the model forgot the mislabeled data.

### Tab 3 — Training Metrics
Shows the training history from `training_log.csv`:
- Train loss vs validation loss over the latest run
- Best validation accuracy achieved
- Best balanced accuracy and best macro-F1
- Total epochs trained

If validation loss starts rising while training loss keeps falling, that's overfitting — the model is memorizing training data instead of generalizing.

---

## What Happens When You Upload a Galaxy Image

1. Image is loaded and converted to RGB
2. Resized to 224×224 pixels
3. Normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. Converted to a PyTorch tensor and passed through ResNet34
5. The final fully-connected layer outputs 3 values (one per class)
6. Softmax converts these to probabilities that sum to 1.0
7. The class with the highest probability is the prediction
8. All three probabilities are displayed as confidence bars

The model loaded in the app is the corrupted model (trained on data with 12% mislabels). This is intentional — it demonstrates the problem that unlearning is solving. After running the unlearning pipeline in Tab 2, you can reload the app with a different model to see how predictions change.

---

## What the Model Has Done on the Training Dataset

During training (`main.py`), the model:

1. Loaded 61,578 galaxy images from Galaxy Zoo 2
2. Split them 70/15/15 into train/validation/test sets
3. Applied data augmentation (random flips, rotations, color jitter) to the training set
4. Trained ResNet34 with transfer learning and mixed precision (FP16)
5. Saved the best model based on validation loss
6. Then intentionally mislabeled 12% of training samples
7. Saved this as the corrupted model
8. Applied all 3 unlearning methods to the corrupted model
9. Evaluated each method on test, forget, and retain sets
10. Saved unlearning results to `unlearning_results.csv` and generated plots in `plots/`

The model uses transfer learning — ResNet34 pretrained on ImageNet is fine-tuned on galaxy images. The ImageNet weights give the model a head start on recognizing edges, textures, and shapes, which transfers well to galaxy morphology.

---

## Project File Structure

```
project/
├── main.py              — runs the full pipeline end to end
├── model.py             — ResNet34 architecture definition
├── train.py             — training loop with mixed precision
├── unlearn.py           — all 3 unlearning methods
├── evaluate.py          — accuracy, forget/retain metrics
├── data_loader.py       — Galaxy Zoo 2 data loading + mislabeling
├── visualize.py         — generates all plots
├── app.py               — Streamlit web interface
├── config.py            — all hyperparameters in one place
├── setup_dataset.py     — one-click Kaggle download
├── data/
│   ├── images_training_rev1/   — 61,578 galaxy images
│   └── training_solutions_rev1.csv  — labels
├── plots/               — generated visualizations
└── unlearning_results.csv — latest comparison table
```

---

## Why This Is Research-Worthy

1. Galaxy Zoo 2 is a real, peer-reviewed scientific dataset used in published astronomy papers
2. Machine unlearning is an active research area with direct applications to GDPR compliance
3. Applying unlearning to astronomical image classification is a novel combination — most unlearning papers use CIFAR-10 or MNIST
4. The project implements and compares 3 distinct unlearning approaches with quantitative metrics
5. The interactive web app makes the research accessible and demonstrable at conferences

Related published work includes Cao & Yang (2015) "Towards Making Systems Forget with Machine Unlearning", Golatkar et al. (2020) "Eternal Sunshine of the Spotless Net", and Bourtoule et al. (2021) "Machine Unlearning" — all of which this project builds upon.
