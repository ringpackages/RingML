# File: examples/chess_final_model.ring
# Description: The Ultimate Chess Training Script with All Features
/* 
Adam Optimizer: For speed.
Dropout: To prevent overfitting.
Tanh: As an activation function (instead of Sigmoid) because it converges faster in deeper layers.
DataLoader: For data processing.
Summary: To display the structure.
SaveWeights: For saving.
*/
load "../../src/ringml.ring"
load "chess_utils.ring"
load "chess_dataset.ring"
load "csvlib.ring"
load "stdlib.ring"

# Better precision for console output
decimals(5)

see "=== RingML Final Chess Model ===" + nl

# 1. Load Data
cFile = "data/chess.csv"
if !fexists(cFile) raise("File missing") ok

see "Reading CSV..." + nl
aRawsData = CSV2List( read(cFile) )
if len(aRawsData) > 0 del(aRawsData, 1) ok 

# 2. Setup Dataset & Loader
dataset = new ChessDataset(aRawsData)
batch_size = 256 
loader = new DataLoader(dataset, batch_size)

see "Dataset: " + dataset.length() + " samples." + nl
see "Batches: " + loader.nBatches + " per epoch." + nl

# 3. Build Advanced Architecture
nClasses = 18
model = new Sequential

# Input -> Dense(32) -> Tanh -> Dropout(20%)
model.add(new Dense(6, 32))   
model.add(new Tanh)        
model.add(new Dropout(0.2))

# Hidden -> Dense(16) -> Tanh -> Dropout(20%)
model.add(new Dense(32, 16))  
model.add(new Tanh)
model.add(new Dropout(0.2))

# Output -> Dense(18) -> Softmax
model.add(new Dense(16, nClasses)) 
model.add(new Softmax)

# 4. Print Summary
model.summary()

# 5. Training Setup
criterion = new CrossEntropyLoss
optimizer = new Adam(0.01) 
nEpochs   = 50 # Increased epochs for better results

# --- SETUP VISUALIZER ---
viz = new TrainingVisualizer(nEpochs, loader.nBatches)

see "Starting Training..." + nl
tTotal = clock()

# Enable Training Mode (Activates Dropout)
model.train()

for epoch = 1 to nEpochs
    epochLoss = 0
    
    for b = 1 to loader.nBatches
        batch = loader.getBatch(b) 
        inputs  = batch[1]
        targets = batch[2]
        
        # Forward
        preds = model.forward(inputs)
        loss  = criterion.forward(preds, targets)
        epochLoss += loss
        
        # Backward
        grad = criterion.backward(preds, targets)
        model.backward(grad)
        
        # Optimizer Step
        for layer in model.getLayers() optimizer.update(layer) next
        
        # --- UPDATE VISUALIZER (Every 5 batches to be smooth) ---
        if b % 5 = 0
            # Calculate rough accuracy for display (optional, or just pass 0)
            # Here we just pass 0 for batch acc to save speed, or calculate it if fast enough.
            # Passing 0 for batch accuracy, focusing on Loss color.
            viz.update(epoch, b, loss, 0)
        ok
    next
    
    avgLoss = epochLoss / loader.nBatches
    
    # see "Epoch " + epoch + "/" + nEpochs + " | Loss: " + avgLoss + nl
    
    # --- FINISH EPOCH VISUALIZATION ---
    viz.finishEpoch(epoch, avgLoss, 0)
    
    
    if epoch % 5 = 0 callgc() ok
next

see "Training Time: " + ((clock()-tTotal)/clockspersecond()) + "s" + nl

# 6. Evaluation Mode (Disable Dropout)
model.evaluate()

# 7. Save Model
model.saveWeights("model/chess_final.rdata")
see "Model Saved Successfully." + nl