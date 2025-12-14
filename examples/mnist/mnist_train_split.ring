# File: examples/mnist_train_split.ring
# Description: MNIST Training with DataSplitter and Accuracy Validation

load "stdlib.ring"
load "../../src/ringml.ring"
load "mnist_dataset.ring"
load "csvlib.ring"

decimals(4)
see "=== RingML MNIST (Train/Test Split Strategy) ===" + nl

# 1. Load Data
cFile = "data/mnist_test.csv"
if !fexists(cFile) raise("File missing") ok

see "Reading CSV..." + nl
aRawsData = CSV2List( read(cFile) )

# Remove Header if exists
if len(aRawsData) > 0 
    if lower(aRawsData[1][1]) = "label" del(aRawsData, 1) ok
ok

# 2. Use DataSplitter (Refactored)
see "Splitting Data (70% Train, 30% Test)..." + nl

splitter = new DataSplitter
# split(Data, TestRatio=0.3, Shuffle=True)
sets = splitter.splitData(aRawsData, 0.3, true)

aTrainData = sets[1]
aTestData  = sets[2]

# Free memory of raw list
aRawsData = [] 
callgc()

see "Training Set: " + len(aTrainData) + " samples." + nl
see "Testing Set:  " + len(aTestData)  + " samples." + nl

# 3. Setup Datasets & Loaders
batch_size = 64

trainDataset = new MnistDataset(aTrainData)
testDataset  = new MnistDataset(aTestData)

trainLoader  = new DataLoader(trainDataset, batch_size)
testLoader   = new DataLoader(testDataset, batch_size)

# 4. Build Model
model = new Sequential

# Input(784) -> Dense(128) -> ReLU -> Dropout
model.add(new Dense(784, 64))   
model.add(new ReLU)
model.add(new Dropout(0.2)) 

# Hidden(64) -> ReLU -> Dropout
model.add(new Dense(64, 32))  
model.add(new ReLU)
model.add(new Dropout(0.2))

# Output(10) -> Softmax
model.add(new Dense(32, 10)) 
model.add(new Softmax)

model.summary()

# 5. Training Setup
criterion = new CrossEntropyLoss
optimizer = new Adam(0.001) 
nEpochs   = 10

# --- SETUP VISUALIZER ---
viz = new TrainingVisualizer(nEpochs, trainLoader.nBatches)

# 6. Training Loop
see nl + "Starting Training..." + nl
tTotal = clock()

for epoch = 1 to nEpochs
    # --- TRAINING PHASE ---
    model.train() # Enable Dropout
    trainLoss = 0
    
    for b = 1 to trainLoader.nBatches
        batch = trainLoader.getBatch(b) 
        inputs  = batch[1]
        targets = batch[2]
        
        preds = model.forward(inputs)
        loss  = criterion.forward(preds, targets)
        trainLoss += loss
        
        grad = criterion.backward(preds, targets)
        model.backward(grad)
        
        for layer in model.getLayers() optimizer.update(layer) next
        
        # --- UPDATE VISUALIZER (Every 5 batches to be smooth) ---
        if b % 5 = 0
            # Calculate rough accuracy for display (optional, or just pass 0)
            # Here we just pass 0 for batch acc to save speed, or calculate it if fast enough.
            # Passing 0 for batch accuracy, focusing on Loss color.
            viz.update(epoch, b, loss, 0)
        ok
    next
    
    avgTrainLoss = trainLoss / trainLoader.nBatches
    
    # --- VALIDATION PHASE (Test Set) ---
    model.evaluate() # Disable Dropout
    correct = 0
    total = 0
    
    for b = 1 to testLoader.nBatches
        batch = testLoader.getBatch(b)
        inputs  = batch[1]
        targets = batch[2] # One-Hot
        
        preds = model.forward(inputs)
        
        # Calculate Accuracy
        nBatchSize = preds.nRows
        for i = 1 to nBatchSize
            # Get ArgMax for Prediction
            predIdx = 0
            predMax = -1
            for k = 1 to 10 
                if preds.aData[i][k] > predMax 
                    predMax = preds.aData[i][k]
                    predIdx = k
                ok
            next
            
            # Get ArgMax for Target
            targetIdx = 0
            targetMax = -1
            for k = 1 to 10
                if targets.aData[i][k] > targetMax
                    targetMax = targets.aData[i][k]
                    targetIdx = k
                ok
            next
            
            if predIdx = targetIdx
                correct++
            ok
            total++
        next
    next
    
    accuracy = (correct / total) * 100
    
    # see "Epoch " + epoch + " | Train Loss: " + avgTrainLoss +  " | Test Accuracy: " + accuracy + "%" + nl
    
    # --- FINISH EPOCH VISUALIZATION ---
    viz.finishEpoch(epoch, avgTrainLoss, accuracy)
        
    if epoch % 2 = 0 callgc() ok
next

see "Total Time: " + ((clock()-tTotal)/clockspersecond()) + "s" + nl

# 7. Save
model.saveWeights("model/mnist_split_model.rdata")
see "Done." + nl