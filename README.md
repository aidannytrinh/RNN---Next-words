# RNN Next-Word Prediction

A hands-on exercise building a simple **RNN Language Model** in PyTorch to predict the next word in a sentence, using a short Vietnamese (unaccented) text about **water and the water cycle**.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Notebook Structure](#notebook-structure)
- [Technical Details](#technical-details)
  - [Text Preprocessing](#1-text-preprocessing)
  - [Building Training Data (Teacher Forcing)](#2-building-training-data-teacher-forcing)
  - [Model Architecture](#3-model-architecture)
  - [Weight Tying](#4-weight-tying)
  - [Training](#5-training)
  - [Inference / Text Generation](#6-inference--text-generation)
- [Results](#results)
- [How to Run](#how-to-run)
- [Limitations and Possible Extensions](#limitations-and-possible-extensions)

## Overview

The `RNN_next_words.ipynb` notebook walks through the full pipeline of a word-level language model, covering:

1. Text preprocessing: building a vocabulary and mapping words to integers.
2. Building a model made of `nn.Embedding`, `nn.RNN`, and `nn.Linear`.
3. Applying **Weight Tying** — sharing weights between the `Embedding` and output `Linear` layers.
4. Training the model with `CrossEntropyLoss` and `Adam`.
5. Applying **Teacher Forcing** when constructing the training data.
6. Predicting the next word and generating new text with the trained model.

## Requirements

- Python 3.8+
- PyTorch (`torch`)

```bash
pip install torch
```

The notebook automatically picks `cuda` if a GPU is available, otherwise it falls back to `cpu`:

```python
thiet_bi = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Notebook Structure

| Section | Content |
|---|---|
| 1 | Import libraries, select device (`cuda`/`cpu`), set `torch.manual_seed(42)` |
| 2 | Define the source text (water cycle topic) |
| 3 | Text preprocessing → build the `tu_sang_so` / `so_sang_tu` vocabularies |
| 4 | Build training data using Teacher Forcing, wrap it into a `DataLoader` |
| 5 | Define the `MoHinhRNN` model (Embedding → RNN → Linear, with Weight Tying) |
| 6 | Training loop with `CrossEntropyLoss` + `Adam` (300 epochs) |
| 7 | `du_doan_tu_tiep_theo`: predicts the single next word given a context sentence |
| 8 | `sinh_van_ban`: generates multiple words in a row (autoregressive generation) |

## Technical Details

### 1. Text Preprocessing

The `tien_xu_ly_van_ban` function lowercases the text, strips out any character that isn't a letter or digit using a regex (`[^a-zA-Z0-9\s]`), and splits the result into a list of words (`split()`).

From that word list, two vocabularies are built:

- `tu_sang_so`: word → index, used to encode the model's input.
- `so_sang_tu`: index → word, used to decode the model's predictions.

For the sample text: **127 words** total, with a vocabulary of **81 unique words**.

### 2. Building Training Data (Teacher Forcing)

This is the core idea behind next-word prediction: at every position in the sequence, the label is simply the input sequence **shifted right by one position**.

```text
Input : nuoc la mot tai nguyen
Label : la mot tai nguyen quan
```

In other words, at every timestep the model receives the **true current word** and is asked to predict the **true next word** — this is exactly what Teacher Forcing means: during training, the model is always fed the ground-truth word from the previous step rather than its own (possibly wrong) prediction, which makes training more stable and lets it converge faster than a fully autoregressive training scheme would.

The `tao_du_lieu_huan_luyen` function slides a window of length `do_dai_ngu_canh = 5` over the entire integer sequence to produce `(dau_vao, nhan)` pairs, which are then wrapped into a `TensorDataset` and `DataLoader` (`batch_size=4`, `shuffle=True`).

### 3. Model Architecture

```python
class MoHinhRNN(nn.Module):
    def __init__(self, kich_thuoc_tu_dien, kich_thuoc_embedding, kich_thuoc_an):
        super().__init__()
        self.embedding = nn.Embedding(kich_thuoc_tu_dien, kich_thuoc_embedding)
        self.rnn = nn.RNN(kich_thuoc_embedding, kich_thuoc_an, batch_first=True)
        self.fc = nn.Linear(kich_thuoc_an, kich_thuoc_tu_dien)
        self.fc.weight = self.embedding.weight  # Weight Tying

    def forward(self, dau_vao):
        vector_tu = self.embedding(dau_vao)
        dau_ra_rnn, trang_thai_an = self.rnn(vector_tu)
        du_doan = self.fc(dau_ra_rnn)
        return du_doan
```

The model has three main components:

1. **`nn.Embedding`** (81 → 64): maps each word index to a dense 64-dimensional vector, letting the model learn semantic relationships between words instead of treating them as isolated symbols (as a one-hot encoding would).
2. **`nn.RNN`** (input 64, hidden 64, `batch_first=True`): processes the embedding vectors sequentially, one timestep at a time, maintaining a hidden state that summarizes context from all previous words.
3. **`nn.Linear`** (64 → 81): projects the RNN's output at each timestep into a vector the size of the vocabulary, producing logits (scores) for every candidate word.

### 4. Weight Tying

```python
self.fc.weight = self.embedding.weight
```

The weight matrix of the `Embedding` layer (shape `[kich_thuoc_tu_dien, kich_thuoc_embedding]`) is **shared** with the weight matrix of the output `Linear` layer (shape `[kich_thuoc_tu_dien, kich_thuoc_an]`), which is only possible when `kich_thuoc_embedding == kich_thuoc_an`. The intuition is that the embedding layer learns "what a word looks like" in vector space, while the output layer learns to "match a hidden vector back to a word" — these are essentially dual problems, so sharing the weights between them offers a few benefits:

- Significantly fewer trainable parameters.
- Lower risk of overfitting, which matters a lot here since the training set is tiny.
- Better word representations overall, since the shared matrix receives gradient updates from both the embedding lookup and the output projection.

The notebook verifies this by checking `mo_hinh.fc.weight is mo_hinh.embedding.weight` → `True`.

### 5. Training

- **Loss function:** `nn.CrossEntropyLoss()` — well suited to this multi-class classification setup, where each word in the vocabulary is a class.
- **Optimizer:** `torch.optim.Adam` with `lr=0.01`.
- **Epochs:** 300.

Since the model's output has shape `(batch_size, do_dai_ngu_canh, kich_thuoc_tu_dien)` while `CrossEntropyLoss` expects a 2D input `(N, C)` and a 1D label `(N,)`, both `du_doan` and `nhan` are flattened via `reshape` before computing the loss:

```python
loss = ham_mat_mat(
    du_doan.reshape(-1, kich_thuoc_tu_dien),
    nhan.reshape(-1)
)
```

### 6. Inference / Text Generation

- `du_doan_tu_tiep_theo(mo_hinh, cau_dau_vao)`: preprocesses the input sentence, keeps only the last `do_dai_ngu_canh` words as context, runs the model in `eval()` mode (no gradient tracking), and takes the `argmax` at the final timestep to pick the highest-scoring word.
- `sinh_van_ban(mo_hinh, cau_bat_dau, so_tu_can_sinh)`: repeatedly predicts one word at a time, appending each new word to the current sentence and using that updated sentence as context for the next prediction — this is autoregressive generation, as opposed to Teacher Forcing, which is only used during training.

## Results

After 300 epochs, the average loss decreases and then plateaus at a low level (the model essentially memorizes the tiny training text):

```
Epoch  50 | Loss trung bình: 0.2068
Epoch 100 | Loss trung bình: 0.1868
Epoch 150 | Loss trung bình: 0.2039
Epoch 200 | Loss trung bình: 0.2070
Epoch 250 | Loss trung bình: 0.2142
Epoch 300 | Loss trung bình: 0.2196
```

Example prediction:

```
Input sentence: nuoc la mot tai
Predicted next word: nguyen
```

Example text generation (10 words, starting from "nuoc la mot"):

```
nuoc la mot tai nguyen quan trong doi voi su song tren trai
```

## How to Run

1. Clone the repo and open the notebook with Jupyter or Google Colab:

   ```bash
   jupyter notebook RNN_next_words.ipynb
   ```

2. Run the cells in order from top to bottom.
3. Feel free to tweak the hyperparameters to experiment:
   - `do_dai_ngu_canh` (context length, default 5)
   - `kich_thuoc_embedding`, `kich_thuoc_an` (default 64)
   - `batch_size`, `lr`, `so_epoch`
4. Try predicting/generating text with different input sentences by calling `du_doan_tu_tiep_theo(...)` or `sinh_van_ban(...)`.

## Limitations and Possible Extensions

- **The dataset is very small** (127 words, 81 unique) so the model tends to memorize rather than generalize — this is fine for illustrating the concepts, but not suitable for real-world use as is.
- **`nn.RNN` (vanilla RNN)** is prone to vanishing/exploding gradients on longer sequences; swapping in `nn.LSTM` or `nn.GRU` would improve the model's ability to retain longer-range context.
- There is no validation/test split to measure how well the model generalizes.
- The current decoding strategy is **greedy decoding** (`argmax`); techniques like temperature sampling, top-k sampling, or beam search could produce more diverse generated text.
- The project could be extended by training on a larger corpus, or by replacing the RNN architecture with a Transformer to compare performance.
