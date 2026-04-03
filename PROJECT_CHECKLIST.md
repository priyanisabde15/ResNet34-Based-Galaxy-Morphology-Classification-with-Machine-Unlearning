# Final Repo Checklist

This checklist is for the cleaned final project version.

## Core Identity

- [x] Project is framed as a `ResNet34` galaxy classification + machine unlearning project
- [x] Real dataset is `Galaxy Zoo 2`
- [x] Final classes are `Smooth`, `Featured/Disk`, `Artifact`
- [x] Final factual summary file exists: `FINAL_PROJECT_FACTS.md`

## Core Files

- [x] `train.py` is the main training entry point
- [x] `unlearn.py` runs standalone unlearning from a saved checkpoint
- [x] `app.py` is the demo/dashboard entry point
- [x] `model.py` reflects the ResNet34 project version

## Saved Outputs Present

- [x] `best_model.pth`
- [x] `training_log.csv`
- [x] `unlearning_results.csv`
- [x] `plots/`

## Cleanup Goals

- [x] Stale ResNet18 wording removed from key docs
- [x] Key docs rewritten around the final project state
- [x] Final facts file added with real saved values
- [x] Repo now points users to the saved final results instead of old draft explanations

## Before Commit

- [ ] Numerical wording is fully standardized everywhere
- [ ] Paper values are aligned to the final repo values
- [ ] Any legacy files kept only for history are either archived or clearly marked
