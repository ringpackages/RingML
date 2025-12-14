# File: examples/mnist_train.ring
# Description: Training MLP on MNIST Digits

load "csvlib.ring"
load "stdlib.ring"
load "../../src/ringml.ring"
load "mnist_dataset.ring"


decimals(5)
see "=== RingML MNIST Digit Recognition ===" + nl

# 1. Load Data
cFile = "data/mnist_test.csv" # Using the smaller test set for quick demo
if !fexists(cFile) raise("File missing: " + cFile) ok

see "Reading CSV (This might take a moment)..." + nl
aRawsData = CSV2List( read(cFile) )

# --- FIX: Remove Header if it exists ---
# Check if first cell is "label" (string) instead of a number
if len(aRawsData) > 0 
    firstCell = aRawsData[1][1]
    # Simple check: if it looks like "label", remove it.
    # Or just unconditionally remove first row if you are sure.
    if isString(firstCell) and lower(trim(firstCell)) = "label"
        del(aRawsData, 1)
        see "Header removed." + nl
    ok
ok

nTotal = len(aRawsData)
see "Loaded " + nTotal + " images." + nl

# 2. Setup Dataset & Loader
dataset = new MnistDataset(aRawsData)
batch_size = 256
loader = new DataLoader(dataset, batch_size)

# 3. Build Model (784 -> 128 -> 64 -> 10)
model = new Sequential

# Flattened Image Input (28*28 = 784)
model.add(new Dense(784, 32))   
model.add(new ReLU)
model.add(new Dropout(0.2)) # Prevent overfitting

model.add(new Dense(32, 16))  
model.add(new ReLU)
model.add(new Dropout(0.2))

model.add(new Dense(16, 10)) # 10 Digits
model.add(new Softmax)

model.summary()

# 4. Train
criterion = new CrossEntropyLoss
optimizer = new Adam(0.001) # Standard LR for Adam
nEpochs   = 5 # MNIST learns fast

# --- SETUP VISUALIZER ---
viz = new TrainingVisualizer(nEpochs, loader.nBatches)

see "Starting Training..." + nl
tTotal = clock()

model.train() # Enable Dropout

for epoch = 1 to nEpochs
    epochLoss = 0
    
    for b = 1 to loader.nBatches
        batch = loader.getBatch(b) 
        inputs  = batch[1]
        targets = batch[2]
        
        preds = model.forward(inputs)
        loss  = criterion.forward(preds, targets)
        epochLoss += loss
        
        grad = criterion.backward(preds, targets)
        model.backward(grad)
        
        for layer in model.getLayers() optimizer.update(layer) next
        
        if b % 10 = 0 callgc() ok
        
        # --- UPDATE VISUALIZER (Every 5 batches to be smooth) ---
        if b % 5 = 0
            # Calculate rough accuracy for display (optional, or just pass 0)
            # Here we just pass 0 for batch acc to save speed, or calculate it if fast enough.
            # Passing 0 for batch accuracy, focusing on Loss color.
            viz.update(epoch, b, loss, 0)
        ok
    next
    
    avgLoss = epochLoss / loader.nBatches
    
    # see nl + "Epoch " + epoch + " Avg Loss: " + avgLoss + nl

    viz.finishEpoch(epoch, avgLoss, 0)
next

see "Total Time: " + ((clock()-tTotal)/clockspersecond()) + "s" + nl

# 5. Save
model.evaluate() # Switch to Eval mode before saving
model.saveWeights("model/mnist_model.rdata")
see "Model Saved." + nl