# Machine Unlearning In This Project

This file explains machine unlearning only in the context of the final project repository.

## Project Context

The classifier is trained on Galaxy Zoo 2 galaxy images using a ResNet34 backbone. After training, a subset of samples is intentionally mislabeled. Those mislabeled samples are treated as the forget set. The remaining clean samples act as the retain set.

## Why Unlearning Matters Here

Galaxy morphology datasets can contain ambiguous or noisy labels. If the model learns from wrong labels, its decision boundaries can shift in the wrong direction. Machine unlearning gives us a way to reduce the influence of those bad samples without fully restarting every time.

## Methods Implemented In This Repo

### 1. Gradient Ascent
The model is updated to increase loss on the forget set. This pushes the model away from what it learned from the corrupted samples.

Saved result in this repo:
- Test accuracy: 70.80%
- Forget accuracy: 11.01%
- Retain accuracy: 76.39%
- Time: 103.57 s

### 2. Fisher Forgetting
Noise is injected into parameters using Fisher-based importance estimates. In this project, it was the weakest of the three saved methods.

Saved result in this repo:
- Test accuracy: 41.71%
- Forget accuracy: 28.36%
- Retain accuracy: 43.44%
- Time: 213.76 s

### 3. Full Retrain
The forget-set samples are removed and the model is retrained from scratch on the retain set. This is the strongest baseline but also the slowest.

Saved result in this repo:
- Test accuracy: 76.17%
- Forget accuracy: 8.36%
- Retain accuracy: 99.64%
- Time: 2649.43 s

## Final Takeaway

In this repo, machine unlearning is not presented as theory only. It is implemented, run, and evaluated with saved outputs. The final comparison is stored in `unlearning_results.csv`.
