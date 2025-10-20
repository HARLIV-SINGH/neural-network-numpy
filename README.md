# Neural Network from Scratch (NumPy)

This repository implements a **neural network** from scratch using only **NumPy**. It trains a model on the **Heart Disease dataset** using gradient descent, with no reliance on external libraries.

The goal is to practice the **mathematical foundations** that make up neural networks: layer creation, activation functions, loss calculation, forward pass, backpropagation and gradient descent 
---

## Features

* Data preprocessing with:

  * Median imputation for missing values
  * Train-test split (80/20)
  * Feature scaling (Z score scaling)
* Backpropagation using **gradient descent**.
* Evaluation with standard percentage calculation.
* Training loss and accuracy visualization with matplotlib.

---

## Files

* `Neural_Network.ipynb` - Notebook to understand step-by-step.
* `Neural_Network.py` - Code to get the testing results directly.
* `heart.csv` – Dataset (Heart Disease).

---

## Installation

Clone the repo:

```bash
git clone https://github.com/HARLIV-SINGH/neural-network.git
cd neural-network
```

Install dependencies:

```bash
pip install numpy pandas matplotlib
```

---

## Usage

### Option 1 – Run Jupyter Notebook

To explore the math and code step by step, launch the notebook:

```bash
jupyter notebook Neural_Network.ipynb
```

This will open an interactive interface in your browser. You can run cells one by one to see the implementation and results.

### Option 2 – Run Python Script

If you just want to execute the core code:

```bash
python Neural_Network.py
```

This will train the model and display results directly in the terminal.

The code:
* Trains the model on 80% of the dataset.
* Evaluates on the remaining 20%.
* Prints training and test accuracies.
* Plots the training loss and accuracy over epochs.

---

## Example Output

```
Train Accuracy: 0.9
Test Accuracy: 0.8829268292682927
```

Training loss and accuracy curve:

<img width="800" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/b909ad9d-e461-427e-9ad2-b212421be268" />
<img width="800" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/08a2fef9-dd2a-4d3c-aab5-2e4d851247b6" />

---

## Next Steps

* Explore random forest and XGBoost models.
* Compare with models from standard libraries.

