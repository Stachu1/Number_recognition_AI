# 🧠 Handwritten Digit Recognition with NumPy

This is a minimal neural network written from scratch in Python using `numpy`, without any deep learning frameworks like TensorFlow or PyTorch. It recognizes **handwritten digits** from the MNIST dataset using a simple feedforward architecture.

---

## 🛠 Features

- Implements a neural network from scratch using NumPy
- Supports:
  - Model training and saving
  - Testing model accuracy on test data
  - Predicting digits from images
- No external ML frameworks — just math

---

## 🧪 Network Architecture

- Input layer: 784 neurons (28x28 image pixels)
- Hidden layer: 80 neurons (configurable)
- Output layer: 10 neurons (digits 0–9)
- Activation: ReLU for hidden layers, Softmax for output

---

## Usage

### Train the Model

```bash
python main.py [iterations] [alpha] [alpha_decay]
```

- `iterations`: number of training iterations
- `alpha`: learning rate
- `alpha_decay`: multiplier applied to alpha every 10 iterations

**Example:**

```bash
python main.py 1000 0.2 0.99
```

---

### Run Test Dataset

```bash
python main.py model.npy
```

Evaluates model accuracy on `data/test.csv`.

---

### Predict from Image

```bash
python main.py model.npy digit.png
```

- Image should be 28x28 pixels, black on white or white on black.
- Automatically inverts and normalizes the image.
