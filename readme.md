# Audio Deepfake Detection — Project Report

**Name:** Muskan Verma  
**Student ID:** st196668

---

## Overview

This project involves binary classification on audio data to detect whether a sample is **bonafide (real)** or a **deepfake**.  
The input consists of pre-extracted audio features of shape **[180, T]**, where:

- **180** represents concatenated LFCC, delta, and delta-delta features  
- **T** represents variable-length time frames

---

## Model Architecture

I implemented a **Bidirectional LSTM with an Attention mechanism** for sequence classification. The architecture consists of the following components:

### 1. Bidirectional LSTM
- Processes the audio feature sequence in both forward and backward directions  
- Hidden dimension of **128 per direction** (256 total)  
- **Two stacked LSTM layers** to capture hierarchical temporal dependencies  
- **Dropout = 0.3** applied between layers to prevent overfitting  

### 2. Attention Layer
- A single linear layer maps each time step’s LSTM output to a scalar attention score  
- Softmax normalization is applied across all time steps  
- Produces a weighted sum of LSTM outputs, allowing the model to focus on the most discriminative audio regions  

### 3. Classifier
- Two-layer feedforward network: **256 → 128 → 1**  
- ReLU activation and dropout  
- Outputs a single logit, converted to a probability using a sigmoid function  

---

## Methodology

### Data Handling
- Variable-length sequences are handled using a custom collate function that zero-pads all sequences in a batch to the longest sequence length  
- Features and labels are loaded from Pandas DataFrames stored as pickle files  
- Data is merged using utterance IDs  

### Training
- Loss function: **Binary Cross-Entropy with Logits**  
- Optimizer: **Adam** with a learning rate of **0.001**  
- Learning rate scheduler: **ReduceLROnPlateau**
  - Factor: 0.5  
  - Patience: 3 epochs  
- Training duration: **20 epochs**  
- Best model saved based on lowest validation loss  

### Evaluation
- Performance measured using **Equal Error Rate (EER)**  
- EER corresponds to the point where false acceptance and false rejection rates are equal  

---

## Results

- Model converged rapidly, exceeding **98% validation accuracy by epoch 2**  
- Stabilized at **99.85% validation accuracy by epoch 12**  
- Achieved an **EER of 0.81%**, indicating strong discrimination between real and deepfake audio  

---

## 🔧 Common Issues & Solutions

### - Issue: OMP Error on macOS
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### - Issue: Pandas Compatibility (2.x vs 3.x)
```bash
python -m pip install pandas==2.2.3
```
