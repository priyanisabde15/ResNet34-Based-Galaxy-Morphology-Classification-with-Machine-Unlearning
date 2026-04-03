# Project Overview

## What The Project Does

This project performs galaxy morphology classification using a pretrained ResNet34 and then applies machine unlearning methods to reduce the effect of intentionally corrupted training samples.

The three final classes are:
- Smooth
- Featured/Disk
- Artifact

## Real Dataset Used

The project uses the real Galaxy Zoo 2 dataset.

Matched image count used by the loader:
- `61,578 images`

Class distribution from the final dataset mapping:
- `Smooth: 26,693`
- `Featured/Disk: 34,826`
- `Artifact: 59`

## Final Model Setup

- Backbone: `ResNet34`
- Transfer learning: yes, with pretrained ImageNet weights
- Input size: `224 x 224`
- Split: `70 / 15 / 15`
- Random seed: `42`
- Mixed precision: enabled on CUDA

## Final Training Story

The final saved training run in this repo reached:
- Best validation accuracy: `85.58%`
- Best balanced accuracy: `87.56%`
- Best macro-F1: `76.41%`
- Saved run length: `19 epochs`

## Final Unlearning Story

Three methods are implemented and compared:
- Gradient Ascent
- Fisher Forgetting
- Full Retrain

Saved comparison results:
- Gradient Ascent: test `70.80%`, forget `11.01%`, retain `76.39%`
- Fisher Forgetting: test `41.71%`, forget `28.36%`, retain `43.44%`
- Full Retrain: test `76.17%`, forget `8.36%`, retain `99.64%`

## Final App Story

The Streamlit app is the easiest way to present the project. It provides:
- classifier tab
- unlearning dashboard tab
- training metrics tab

## Final Repo Position

This repository is now meant to reflect the final ResNet34-based version of the project, not the older ResNet18-era wording.
