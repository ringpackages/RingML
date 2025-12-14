# File: examples/fast_viz_demo.ring
# Description: Quick Demo with GUARANTEED convergence to see visuals

load "stdlib.ring"
load "../src/ringml.ring"


decimals(10)

see "=== RingML Visual Demo (Guaranteed Learning) ===" + nl

# 1. Generate Very Clear Synthetic Data (3 Distinct Clusters)
see "Generating distinct data..." + nl
aRawsData = []

# Class 0: Near (0, 0)
for i = 1 to 200
    aRawsData + [[random(10)/100.0, random(10)/100.0], [1.0, 0.0, 0.0]] # Use Floats
next

# Class 1: Near (1, 1)
for i = 1 to 200
    aRawsData + [[0.9 + random(10)/100.0, 0.9 + random(10)/100.0], [0.0, 1.0, 0.0]]
next

# Class 2: Near (0, 1)
for i = 1 to 200
    aRawsData + [[random(10)/100.0, 0.9 + random(10)/100.0], [0.0, 0.0, 1.0]]
next

# 2. Split Data
splitter = new DataSplitter
sets = splitter.splitData(aRawsData, 0.2, true) 
aTrain = sets[1]
aTest  = sets[2]


trainDataset = MakeTensorDataset(aTrain)
testDataset  = MakeTensorDataset(aTest)

batch_size = 16 # Small batch = More updates = Cool visualization
trainLoader = new DataLoader(trainDataset, batch_size)
testLoader  = new DataLoader(testDataset, batch_size)

# 4. Build Model
model = new Sequential
model.add(new Dense(2, 16))
model.add(new Tanh)
model.add(new Dense(16, 3))
model.add(new Softmax)

model.summary()
//sleep(1)

# 5. Training
criterion = new CrossEntropyLoss
optimizer = new Adam(0.001) # Fast Learning Rate
nEpochs   = 20

viz = new TrainingVisualizer(nEpochs, trainLoader.nBatches)

for epoch = 1 to nEpochs
    model.train()
    epochLoss = 0
    
    for b = 1 to trainLoader.nBatches
        batch = trainLoader.getBatch(b)
        inputs = batch[1]
        targets = batch[2]
        
        preds = model.forward(inputs)
        loss  = criterion.forward(preds, targets)
        epochLoss += loss
        
        grad = criterion.backward(preds, targets)
        model.backward(grad)
        
        for l in model.getLayers() optimizer.update(l) next
        
        # Slow down slightly so human eye can see the bar moving
        # sleep(0.01) 
        viz.update(epoch, b, loss, 0)
    next
    
    avgLoss = epochLoss / trainLoader.nBatches
    
    # Validation
    model.evaluate()
    correct = 0
    total = 0
    for b = 1 to testLoader.nBatches
        batch = testLoader.getBatch(b)
        preds = model.forward(batch[1])
        targets = batch[2]
        
        Rows = preds.nRows
        for r=1 to Rows
            pMax = -1 pIdx=0
            tMax = -1 tIdx=0
            for k=1 to 3
                if preds.aData[r][k] > pMax pMax=preds.aData[r][k] pIdx=k ok
                if targets.aData[r][k] > tMax tMax=targets.aData[r][k] tIdx=k ok
            next
            if pIdx = tIdx correct++ ok
            total++
        next
    next
    
    acc = (correct/total) * 100
    viz.finishEpoch(epoch, avgLoss, acc)
next

see nl + "Done! Look at those colors! 🎨" + nl

# Helper to Convert Lists to Tensors
func MakeTensorDataset aDataList
    Rows = len(aDataList)
    if Rows = 0 return NULL ok
    nInCols  = len(aDataList[1][1])
    nOutCols = len(aDataList[1][2])
    tIn  = new Tensor(Rows, nInCols)
    tOut = new Tensor(Rows, nOutCols)
    for i = 1 to Rows
        tIn.aData[i]  = aDataList[i][1]
        tOut.aData[i] = aDataList[i][2]
    next
    return new TensorDataset(tIn, tOut)