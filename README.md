# English-Marathi Neural Machine Translation with Attention

This repository implements a sequence-to-sequence neural machine translation model with Bahdanau attention for translating English text to Marathi. The model uses an encoder-decoder architecture with LSTM layers and attention mechanism to improve translation quality.

## Project Structure

- `attention.py`: Custom implementation of Bahdanau attention layer for TensorFlow/Keras
- `transliteration_attention.py`: Main implementation of the English-Marathi translation model

## Features

- Custom Bahdanau attention layer implementation
- Triple-stacked LSTM encoder architecture
- Attention-based decoder
- Support for variable length sequences
- Comprehensive data preprocessing pipeline
- Model training with early stopping
- Inference model for translations

## Technical Details

### Model Architecture

- **Encoder**:
  - Embedding layer (500 dimensions)
  - 3 stacked LSTM layers with states preserved
  - Returns encoder outputs and final states

- **Decoder**:
  - Embedding layer (500 dimensions)
  - LSTM layer with encoder states as initial state
  - Attention layer combining encoder and decoder outputs
  - Time-distributed dense layer for output generation

- **Attention Mechanism**:
  - Implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf)
  - Uses three trainable weight matrices (W_a, U_a, V_a)
  - Computes attention scores and context vectors

### Requirements

```
tensorflow
pandas
scikit-learn
numpy
pydot
graphviz
```

### Dataset Format

The model expects a tab-separated text file with English-Marathi pairs:
```
english_word[TAB]marathi_word
```

### Usage

1. Data Preprocessing:
```python
# Load and preprocess the data
with open('mar.txt', 'r') as f:
    data = f.read()
# Follow preprocessing steps in transliteration_attention.py
```

2. Training:
```python
# Train the model
history = model.fit([X_train, y_train[:,:-1]], 
                   y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:,1:],
                   epochs=50,
                   callbacks=[es],
                   batch_size=512,
                   validation_data=([X_test, y_test[:,:-1]], 
                                  y_test.reshape(y_test.shape[0], y_test.shape[1], 1)[:,1:]))
```

3. Translation:
```python
# Translate text
decoded_sentence = decode_sequence(input_seq)
```

## Model Performance

The model achieves:
- Training accuracy: ~62.8%
- Validation accuracy: ~63.8%
- Early stopping typically occurs around epoch 15

## Saving/Loading

The model architecture and weights are saved separately:
- Model architecture: `NMT_model.json`
- Model weights: `NMT_model_weight.h5`
- Tokenizers and preprocessing data: Saved as pickle files


## References

1. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
