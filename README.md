# 🧠 RingML: Deep Learning Library for Ring

RingML is a modular, high-performance Deep Learning framework built from scratch in the Ring programming language. It leverages the FastPro C-extension to perform accelerated matrix operations, enabling the training of Neural Networks for tasks like regression, binary classification, and multi-class classification.

The goal is to provide a PyTorch-like object-oriented API for Ring developers, adhering to Jacob's Law by offering a familiar and intuitive interface.

**Current Version:** 1.0.6 (Stable - Hybrid Engine)

## 📚 Installation

```bash
ringpm install ringml from Azzeddine2017
```

## 🛠️ Tech Stack & Architecture

- **Core Language:** Ring Programming Language (v1.24+)
- **Acceleration:** FastPro Extension (C-based DLL/SO) for heavy lifting.
- **Architecture:** Modular OOP (Tensor, Layers, Model, Optim, Data).
- **Math Engine:** Hybrid Mode.
    - *Critical ops (Sub, MatMul, Transpose)* are implemented in optimized Ring loops to ensure precision and stability.
    - *Heavy ops (Add, Random, Activations)* utilize C-level FastPro speed.
- **Persistence:** Custom High-Precision Serializer (.rdata) preserving 15-18 decimal places.

## 🚀 Key Features by Module

### Module 1: Core Engine (src/core/)
- **Tensor Class:** Wraps math operations.
- **Operations:** Matrix Multiplication, Transpose, Broadcasting, Scalar Math.
- **Stability:** Implements Numerically Stable Softmax to prevent NaN during training.

### Module 2: Neural Building Blocks (src/layers/)
- **Dense:** Fully Connected Layer with smart weight initialization.
- **Activations:** ReLU, Sigmoid, Tanh, Softmax.
- **Regularization:** Dropout layer to prevent overfitting.

### Module 3: Model & Optimization (src/model/, src/optim/)
- **Sequential:** Stack layers linearly. Includes `summary()` to view architecture and parameter counts.
- **Optimizers:**
    - SGD (Stochastic Gradient Descent).
    - Adam (Adaptive Moment Estimation) for fast convergence.
- **Loss Functions:** MSELoss (Regression), CrossEntropyLoss (Classification).
- **Modes:** Support for `train()` and `evaluate()` modes (essential for Dropout).

### Module 4: Data Pipeline (src/data/)
- **DataSplitter:** Utility to shuffle and split raw data into Training/Testing sets (e.g., 80/20 split).
- **DataLoader:** Efficient Mini-Batch processing to handle large datasets without memory overflow.
- **Lazy Loading:** Custom Dataset support.

## ⚡ Quick Start Guide

### 1. Data Preparation
Use `DataSplitter` to handle raw CSV data and `DataLoader` for batching.

```ring
load "src/ringml.ring"
load "stdlib.ring"

# 1. Load Data
aRawData = [ [0,0,0], [0,1,1], [1,0,1], [1,1,0] ] # Example XOR data

# 2. Split (80% Train, 20% Test) with Shuffle
splitter = new DataSplitter
sets = splitter.splitData(aRawData, 0.2, true) 
trainData = sets[1]
testData  = sets[2]

# 3. Create Loader (Batch Size = 32)
dataset = new TensorDataset(trainData) 
loader  = new DataLoader(dataset, 32)
```

### 2. Building the Model
Construct a model using Tanh for hidden layers and Dropout for regularization.

```ring
model = new Sequential

# Input: 10 features -> Hidden: 64 neurons
model.add(new Dense(10, 64))   
model.add(new Tanh)        
model.add(new Dropout(0.2)) # Drop 20% of neurons during training

# Hidden: 64 -> Output: 3 classes
model.add(new Dense(64, 3)) 
model.add(new Softmax)

# View architecture
model.summary()
```
### 2.2 Freezing a layer
```ring
# freeze layer
model = new Sequential
l1 = new Dense(6, 32)
l1.freeze()  # freeze first layer
model.add(l1)
model.add(new Tanh)
model.add(new Dropout(0.2))
model.add(new Dense(32, 1))
model.add(new Softmax)

model.summary()
```

### 3. Training with Adam
The training loop handles Forward pass, Backward pass (Backprop), and Optimization.

```ring
criterion = new CrossEntropyLoss
optimizer = new Adam(0.001) 
nEpochs   = 50

# Enable Training Mode (Activates Dropout)
model.train() 

for epoch = 1 to nEpochs
    epochLoss = 0
    for b = 1 to loader.nBatches
        batch = loader.getBatch(b)
        inputs = batch[1] 
        targets = batch[2]
        
        # Forward & Loss
        preds = model.forward(inputs)
        loss  = criterion.forward(preds, targets)
        
        # Backward & Update
        grad = criterion.backward(preds, targets)
        model.backward(grad)
        for layer in model.getLayers() optimizer.update(layer) next
        
        epochLoss += loss
    next
    see "Epoch " + epoch + " Loss: " + (epochLoss / loader.nBatches) + nl
next
```

### 4. Saving
Switch to evaluation mode to disable Dropout, then save.

```ring
model.evaluate() 
model.saveWeights("mymodel.rdata")
see "Model Saved." + nl
```

## ✅ Development Progress (Changelog)
**Status:** ✅ Completed & Verified

- [x] **T01:** Core Tensor Engine & Math Operations.
- [x] **T02:** Layers (Dense, ReLU, Sigmoid, Tanh, Softmax, Dropout).
- [x] **T03:** Model Management (Sequential, Summary, Save/Load).
- [x] **T04:** Optimization (SGD, Adam) & Loss (MSE, CrossEntropy).
- [x] **T05:** Data Pipeline (Dataset, DataLoader, DataSplitter).
- [x] **T06:** Real-World Demos:
    - XOR Problem (Binary Classification).
    - Chess End-Game (18-Class Classification).
    - MNIST (Computer Vision / Digit Recognition).

## 📝 License
Open Source. 
