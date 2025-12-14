# File: src/utils/visualizer.ring
# Description: Console Visualization for Training Process using RingConsoleColors
# Author: Azzeddine Remmal


class TrainingVisualizer
    nTotalEpochs
    nTotalBatches
    
    # Spinner animation frames
    aSpinner = ["|", "/", "-", "\\"]
    nSpinIdx = 1

    # ASCII Brain Icon lines
    aBrain = [
        "   (  )   ",
        "  (    )  ",
        "   (  )   "
    ]

    func init nEpochs, nBatches
        nTotalEpochs  = nEpochs
        nTotalBatches = nBatches
        
        # Initial Clear/Header
        cc_print(CC_FG_CYAN, "==========================================" + nl)
        cc_print(CC_FG_CYAN, "      RingML Training Dashboard           " + nl)
        cc_print(CC_FG_CYAN, "==========================================" + nl)

    func update nEpoch, nBatch, nLoss, nAcc
        # 1. Calculate Progress
        nPercent = (nBatch / nTotalBatches) * 100
        nBarLen  = 20
        nFilled  = floor((nPercent / 100) * nBarLen)
        
        # 2. Build Progress Bar String
        cBar = "["
        cBar += copy("=", nFilled)
        if nFilled < nBarLen cBar += ">" ok
        cBar += copy(" ", nBarLen - nFilled)
        cBar += "]"

        # 3. Determine Colors based on performance
        # Loss Color: Red (Bad) -> Yellow -> Green (Good < 1.0)
        nLossColor = CC_FG_RED
        if nLoss < 2.0 nLossColor = CC_FG_YELLOW ok
        if nLoss < 1.0 nLossColor = CC_FG_GREEN ok

        # Accuracy Color: Red -> Yellow -> Green (> 50%)
        nAccColor = CC_FG_RED
        if nAcc > 20 nAccColor = CC_FG_YELLOW ok
        if nAcc > 50 nAccColor = CC_FG_GREEN ok

        # 4. Draw the UI (Using \r to overwrite line)
        # Note: We print multiple components. 
        # Standard Ring `see` adds newlines, so we simulate a dashboard by clearing previous block 
        # or just printing a single dynamic line for the batch progress.
        
        # Move cursor to start of line (Carriage Return)
        see char(13) 
        
        # Print Spinner
        see aSpinner[nSpinIdx] + " "
        nSpinIdx++ 
        if nSpinIdx > 4 nSpinIdx = 1 ok

        # Print Epoch info
        cc_print(CC_FG_CYAN, " Epoch " + nEpoch + "/" + nTotalEpochs + " ")
        
        # Print Progress Bar
        cc_print(CC_FG_WHITE, cBar + " " + floor(nPercent) ) + "%"
        
        # Print Metrics
        cc_print(CC_FG_WHITE, "| Loss: ")
        cc_print(nLossColor, "" + nLoss + " ")
        
        cc_print(CC_FG_WHITE, "| Accuracy: ")
        cc_print(nAccColor, "" + nAcc ) + "%"
        
        # No new line at the end to allow overwriting!

    func finishEpoch nEpoch, nAvgLoss, nValAcc
        # When epoch is done, we print a permanent summary line
        see char(13) + copy(" ", 80) + char(13) # Clear line
        
        cc_print(CC_FG_GREEN, "[DONE] Epoch " + nEpoch + ": ")
        cc_print(CC_FG_WHITE, "Avg Loss = " + nAvgLoss)
        cc_print(CC_FG_WHITE, " | Val Accuracy = " + nValAcc ) + "%"
        see nl 
        # Print a separator or Brain animation frame if desired
    