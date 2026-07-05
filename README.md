# Multi-Task Social NLP using Pretrained Transformers (pysentimiento)

A beginner-friendly deep learning project demonstrating **transfer learning** for Social NLP tasks — sentiment analysis, emotion detection, irony detection, and hate speech detection — using pretrained Transformer models via the [`pysentimiento`](https://github.com/pysentimiento/pysentimiento) library.

Instead of training a neural network from scratch, this project uses existing pretrained RoBERTa/BERTweet-based models and evaluates them against a real, human-labeled dataset — a common and practical real-world deep learning workflow.

## Project Overview

| | |
|---|---|
| **Tasks** | Sentiment Analysis, Emotion Detection, Irony Detection, Hate Speech Detection |
| **Approach** | Transfer learning with pretrained Transformer models (no training from scratch) |
| **Library** | [pysentimiento](https://github.com/pysentimiento/pysentimiento) |
| **Evaluation Dataset** | [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset) (Kaggle) |
| **Environment** | Google Colab |

## What This Notebook Does

1. **Loads four pretrained analyzers** (sentiment, emotion, irony, hate speech) once and reuses them for efficiency.
2. **Runs single-sentence demos** to show how each model interprets different types of text.
3. **Batch-analyzes a set of test sentences** and compiles results into a comparison table.
4. **Visualizes emotion distribution** across the test sentences with a bar chart.
5. **Downloads a real, human-labeled Kaggle dataset** (~25,000 tweets) and evaluates the pretrained hate speech model against a 100-tweet sample.
6. **Reports quantitative metrics** — accuracy, precision, recall, and a confusion matrix — comparing model predictions to ground-truth human labels.
7. **Discusses limitations and ethical considerations**, including label bias, domain mismatch, and the risks of using such a model for real-world moderation.

## How to Run

1. Open `Social_NLP_Project_pysentimiento.ipynb` in **Google Colab**.
2. Run the setup cells to install `pysentimiento` and other dependencies.
3. When prompted, upload your `kaggle.json` API key (from **Kaggle → Account → Create New Token**) to download the evaluation dataset.
4. Run all remaining cells in order — total runtime is roughly 10–15 minutes.

## Results Summary

The notebook reports the pretrained hate speech model's accuracy, precision, and recall on a 100-tweet sample from the Kaggle dataset, along with a confusion matrix. *(Fill in your actual numbers here after running the notebook, e.g. "Achieved ~XX% accuracy on the sample.")*

## Limitations

- The Kaggle dataset's human labels may reflect annotator bias, especially around slang and reclaimed language.
- The pretrained model was trained on different social media data, so its definition of "hateful" may not perfectly align with this dataset's labeling guidelines.
- Collapsing "offensive" and "neither" into a single "not hateful" class simplifies the original 3-class problem.
- This project is for educational purposes only and should not be used as a standalone content moderation system.

## Possible Extensions

- Evaluate on the full dataset instead of a 100-row sample.
- Compare pysentimiento's results against a custom-trained LSTM/CNN classifier built from scratch.
- Test the Spanish, Portuguese, or Italian pretrained models for multilingual comparison.

## References

- Pérez, J.M. et al. *pysentimiento: A Python Toolkit for Opinion Mining and Social NLP tasks*. [GitHub](https://github.com/pysentimiento/pysentimiento)
- Davidson, T. et al. *Hate Speech and Offensive Language Dataset*. [Kaggle](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)
