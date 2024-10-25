
# EEG-based Depression Classification via Transfer Learning

This project implements a deep learning approach for Major Depressive Disorder (MDD) classification using EEG data, leveraging transfer learning from sleep stage classification tasks.

## Project Overview

The project explores transfer learning techniques by first training a model on sleep stage classification (which has abundant data) and then fine-tuning it for MDD detection. The approach is based on the hypothesis that both tasks rely on similar low-frequency EEG patterns.

This work is inspired by and based on the research presented in: [Research Paper](http://biorxiv.org/lookup/doi/10.1101/2023.04.29.538813)

## Repository Structure
```
├── baseline_model.ipynb          # Main notebook for MDD classification models
├── convert_keras_weights.ipynb   # Weight conversion utility (TF to PyTorch)
├── peprosecc_mdd.ipynb          # MDD data preprocessing
├── preprocess_mdd.py            # MDD preprocessing utilities
├── pretrain.ipynb               # Model pretraining notebook
├── sleep_preprocessing.py        # Sleep data preprocessing
├── sleep_pretrain.pt            # Pretrained model on sleep data
└── sleep_train.py               # Sleep model training script
```

## Methodology

1. **Pretraining Phase**
   - Train a model on sleep stage classification task
   - Utilize large-scale sleep EEG datasets
   - Focus on learning relevant low-frequency patterns

2. **Transfer Learning Phase**
   - Fine-tune the pretrained model for MDD classification
   - Adapt the model architecture for binary classification
   - Validate on MDD dataset

## Data

The project uses two types of data:
- Sleep EEG data for pretraining
- MDD vs. Healthy Control EEG data for fine-tuning

## Setup and Installation

# Clone the repository

```bash
git clone https://github.com/ansafronov/neuroml_project
```

## Usage

1. **Data Preprocessing**
bash
python sleep_preprocessing.py
python preprocess_mdd.py

2. **Model Training**

bash
python sleep_train.py

3. **Fine-tuning and Evaluation**
- Open `baseline_model.ipynb` in Jupyter Notebook

## Results

[Add your model performance metrics and results here]

## Contributing

Feel free to open issues or submit pull requests for improvements.


## Acknowledgments

- based on Improving Multichannel Raw Electroencephalography-based Diagnosis of Major Depressive Disorder via Transfer Learning with Single Channel Sleep Stage Data*
Charles A. Ellis, Abhinav Sattiraju, Robyn L. Miller, Vince D. Calhoun
bioRxiv 2023.04.29.538813; doi: https://doi.org/10.1101/2023.04.29.538813