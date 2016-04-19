filename = "Log3.txt"

f = open(filename)

totalT = 0
minT = 1000
maxT = 0
numEpoch = 200
trainLoss = []
trainAcc = []
valLoss = []
valAcc = []

for lineNo in range(1, (numEpoch+1)):
    epochLine = f.readline()
    dataLine = f.readline()
    dL  = dataLine.strip().split()
    t, tl, ta, vl, va  = float(dL[3][:-1]), float(dL[6]), float(dL[9]), float(dL[12]), float(dL[15])
    totalT += t
    if t <= minT:
        minT = t
    if t > maxT:
        maxT = t
    trainLoss.append(tl)
    trainAcc.append(ta)
    valLoss.append(vl)
    valAcc.append(va)

f.close()

print '{}&{}&{}'.format(minT, maxT, totalT/numEpoch)

lineFile = ""

for i in range(numEpoch):
    nL = "{}&{}&{}&{}&{}\n".format(float(i+1), trainLoss[i], valLoss[i], trainAcc[i], valAcc[i])
    
    lineFile += nL

valFile = filename[:-4]+"Data.txt"

lossF = open(valFile, "w")

lossF.write(lineFile)

