# File: examples/chess_train_split.ring
# Description: Chess Training using built-in DataSplitter

load "stdlib.ring"
load "../../src/ringml.ring"
load "chess_utils.ring"
load "chess_dataset.ring"
load "csvlib.ring"

decimals(5)
see "=== RingML Chess Training (Automated Split) ===" + nl

# 1. Load Data
cFile = "data/chess.csv"
if !fexists(cFile) raise("File missing") ok

see "Reading CSV..." + nl
aRawsData = CSV2List( read(cFile) )
if len(aRawsData) > 0 del(aRawsData, 1) ok 

# 2. Use DataSplitter (The new feature)
see "Splitting Data (80% Train, 20% Test)..." + nl

splitter = new DataSplitter
# split(Data, TestRatio=0.2, Shuffle=True)
sets = splitter.splitData(aRawsData, 0.2, true) 

aTrainData = sets[1]
aTestData  = sets[2]

# Free memory of original list
aRawsData = [] 
callgc()

see "Training Set: " + len(aTrainData) + " samples." + nl
see "Testing Set:  " + len(aTestData)  + " samples." + nl

# 3. Setup Datasets & Loaders
batch_size = 512 

trainDataset = new ChessDataset(aTrainData)
testDataset  = new ChessDataset(aTestData)

trainLoader  = new DataLoader(trainDataset, batch_size)
testLoader   = new DataLoader(testDataset, batch_size)

# 4. Build Model
nClasses = 18
model = new Sequential

model.add(new Dense(6, 64))   
model.add(new Tanh)        
model.add(new Dropout(0.2))

model.add(new Dense(64, 32))  
model.add(new Tanh)
model.add(new Dropout(0.2))

model.add(new Dense(32, nClasses)) 
model.add(new Softmax)

model.summary()

# 5. Training Setup
criterion = new CrossEntropyLoss
optimizer = new Adam(0.002) 
nEpochs   = 20

# --- SETUP VISUALIZER ---
viz = new TrainingVisualizer(nEpochs, trainLoader.nBatches)

see "Starting Training..." + nl
tTotal = clock()

for epoch = 1 to nEpochs
    
    # --- Training ---
    model.train() 
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
    
    # --- Validation ---
    model.evaluate()
    correct = 0
    total = 0
    
    for b = 1 to testLoader.nBatches
        batch = testLoader.getBatch(b)
        inputs  = batch[1]
        targets = batch[2]
        
        preds = model.forward(inputs)
        
        # Calculate Accuracy logic...
        nBatchSize = preds.nRows
        for i = 1 to nBatchSize
            # ArgMax Pred
            pMax = -1 pIdx = 0
            for k=1 to nClasses if preds.aData[i][k] > pMax pMax=preds.aData[i][k] pIdx=k ok next
            
            # ArgMax Target
            tMax = -1 tIdx = 0
            for k=1 to nClasses if targets.aData[i][k] > tMax tMax=targets.aData[i][k] tIdx=k ok next
            
            if pIdx = tIdx correct++ ok
            total++
        next
    next
    
    accuracy = (correct / total) * 100
    # see "Epoch " + epoch  + "/" + nEpochs + " | Loss: " + avgTrainLoss + " | Val Acc: " + accuracy + "%" + nl
    
    # --- FINISH EPOCH VISUALIZATION ---
    viz.finishEpoch(epoch, avgTrainLoss, accuracy)

    if epoch % 5 = 0 callgc() ok
next

see "Total Time: " + ((clock()-tTotal)/clockspersecond()) + "s" + nl

model.saveWeights("model/chess_split_model.rdata")

see "Done." + nl
