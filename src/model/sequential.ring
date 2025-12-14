# File: src/model/sequential.ring
# Description: Sequential Model Container with Save/Load support
# Author: Azzeddine Remmal

class Sequential
    aLayers = []

    func add oLayer
        aLayers + oLayer
        return self

    func forward oInput
        oCurrent = oInput
        for oLayer in aLayers
            oCurrent = oLayer.forward(oCurrent)
        next
        return oCurrent

    func backward oGradOutput
        oCurrentGrad = oGradOutput
        for i = len(aLayers) to 1 step -1
            oCurrentGrad = aLayers[i].backward(oCurrentGrad)
        next
        return oCurrentGrad
        
    func getLayers
        return aLayers

    # --- Mode Switching ---

    func train
        for oLayer in aLayers
            oLayer.train()
        next

    func evaluate
        for oLayer in aLayers
            oLayer.evaluate()
        next

    # --- Save & Load Functionality ---

    func saveWeights cFileName
        see "Saving model to " + cFileName + "..." + nl
        
        # 1. Collect all weights/biases into a single flat list
        aAllParams = []
        
        for oLayer in aLayers
            if hasParams(oLayer)
                # We save the raw list data (aData)
                aAllParams + oLayer.oWeights.aData
                aAllParams + oLayer.oBias.aData
            ok
        next
        
        # 2. Serialize using High-Precision utility
        cData = SerializeData(aAllParams)
        
        write(cFileName, cData)
        see "Done." + nl

    func loadWeights cFileName
        see "Loading model from " + cFileName + "..." + nl
        
        if !fexists(cFileName)
            raise("Error: File not found - " + cFileName)
        ok
        
        # 1. Read and Deserialize
        # eval() is used to parse the code string back into a list
        cCode = "return " + read(cFileName)
        aAllParams = eval(cCode)
        
        # 2. Distribute params back to layers
        nIdx = 1
        for oLayer in aLayers
            if hasParams(oLayer)
                if nIdx > len(aAllParams) 
                    raise("Error: Model architecture mismatch (not enough params)")
                ok
                
                # Restore Weights
                oLayer.oWeights.aData = aAllParams[nIdx]
                nIdx++
                
                # Restore Bias
                oLayer.oBias.aData = aAllParams[nIdx]
                nIdx++
            ok
        next
        see "Done." + nl

	 # --- Model Summary (Colorful Version) ---

    func summary
        # Ensure console colors are available (usually loaded by ringml.ring)
        # If not, this might throw error, but we assume environment is set.
        
        see nl
        cc_print(CC_FG_CYAN,  "_________________________________________________________________" + nl)
        cc_print(CC_FG_WHITE, pad("Layer (Type)", 29) + pad("Output Shape", 26) + "Param #" + nl)
        cc_print(CC_FG_CYAN,  "=================================================================" + nl)
        
        nTotalParams = 0
        nTrainableParams = 0
        nNonTrainableParams = 0
        
        cLastOutputShape = "Input" 

        for oLayer in aLayers
            cName = classname(oLayer)
            nParams = 0
            
            # --- Logic to extract info ---
            if cName = "dense"
                nW = oLayer.nInputSize * oLayer.nNeurons
                nB = oLayer.nNeurons
                nParams = nW + nB
                cOutputShape = "(None, " + oLayer.nNeurons + ")"
                cLastOutputShape = cOutputShape
            else
                nParams = 0
                cOutputShape = cLastOutputShape 
            ok
            
            nTotalParams += nParams
            
            # --- Count Trainable vs Non-Trainable ---
            if hasAttribute(oLayer, "bTrainable") and oLayer.bTrainable
                nTrainableParams += nParams
            else
                nNonTrainableParams += nParams
            ok
            
            # --- Colorful Printing ---
            cCol1 = pad(cName, 29)
            cCol2 = pad(cOutputShape, 26)
            cCol3 = "" + nParams
            
            # Name in Yellow
            cc_print(CC_FG_YELLOW, cCol1)
            
            # Shape in Cyan
            cc_print(CC_FG_CYAN,   cCol2)
            
            # Params in Green
            cc_print(CC_FG_GREEN,  cCol3 + nl)
            
            # Separator in Dark Gray (Subtle)
            cc_print(CC_FG_DARK_GRAY, "_________________________________________________________________" + nl)
        next
        
        # --- Footer ---
        see nl
        cc_print(CC_FG_WHITE, "Total params:         ")
        cc_print(CC_FG_CYAN,  "" + nTotalParams + nl)
        
        cc_print(CC_FG_WHITE, "Trainable params:     ")
        cc_print(CC_FG_GREEN, "" + nTrainableParams + nl)
        
        cc_print(CC_FG_WHITE, "Non-trainable params: ")
        
        if nNonTrainableParams > 0
            cc_print(CC_FG_RED, "" + nNonTrainableParams + nl)
        else
            cc_print(CC_FG_GRAY, "0" + nl)
        ok
        
        cc_print(CC_FG_CYAN,  "_________________________________________________________________" + nl + nl)

    private
    
    # Helper to check if layer is trainable
	func hasParams oLayer
        aAttrs = attributes(oLayer)
        bHasW = false
        bHasB = false
        for cAttr in aAttrs
            if lower(cAttr) = "oweights" bHasW = true ok
            if lower(cAttr) = "obias"    bHasB = true ok
        next
        return (bHasW and bHasB)

	# Helper for string padding
    func pad cStr, nLen
        if len(cStr) >= nLen return cStr ok
        return cStr + copy(" ", nLen - len(cStr))
