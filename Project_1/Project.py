from simpful import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


FS = FuzzySystem()

# Antecedents
# HW Subsystem
M1 = FuzzySet( points=[[0, 1],  [0.25, 0]], term="low" )
M2 = FuzzySet( points=[[0.2, 0], [0.35, 1], [0.60,0]], term="average" )
M3 = FuzzySet( points=[[0.5, 0], [0.70, 1.]],term="high" )
M4 = FuzzySet( points=[[0.85, 0], [1, 1]], term="critical" )
FS.add_linguistic_variable("Memory", LinguisticVariable([M1, M2, M3, M4], universe_of_discourse=[0, 1]))

P1 = FuzzySet( points=[[0, 1],  [0.25, 0]], term="low" )
P2 = FuzzySet( points=[[0.2, 0], [0.35, 1], [0.60,0]], term="average" )
P3 = FuzzySet( points=[[0.5, 0], [0.70, 1.]],term="high" )
P4 = FuzzySet( points=[[0.85, 0], [1, 1]], term="critical" )
FS.add_linguistic_variable("Processor", LinguisticVariable([P1, P2, P3, P4], universe_of_discourse=[0, 1]))

SP1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0.375), term="low")
SP2 = FuzzySet(function=Triangular_MF(a=0.125,b=0.5,c=0.75), term="balanced")
SP3 = FuzzySet(function=Triangular_MF(a=0.675,b=0.85,c=1), term="high")
FS.add_linguistic_variable("HW_Usage", LinguisticVariable([SP1, SP2, SP3], universe_of_discourse=[0, 1]))

# HW Subsystem Rules

R_CRITICAL = "IF (Memory IS critical) OR (Processor IS critical) THEN (HW_Usage IS high)"

R_HIGH = "IF (Processor IS high) AND (Memory IS high) THEN (HW_Usage IS high)"

R_BAL1 = "IF (Processor IS average) AND ((Memory IS average) OR (Memory IS high)) THEN (HW_Usage IS balanced)"
R_BAL2 ="IF (Memory IS average) AND ((Processor IS average) OR (Processor IS high)) THEN (HW_Usage IS balanced)"

R_LOW1 = "IF (Memory IS low) AND (NOT(Processor IS critical)) THEN (HW_Usage IS low)"
R_LOW2 = "IF (Memory IS low) AND (NOT(Processor IS critical)) THEN (HW_Usage IS low)"



FS.add_rules([R_CRITICAL, R_HIGH, R_BAL1, R_BAL2, R_LOW1, R_LOW2])

#Subsystem2
INP1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0.58), term="low")
INP2 = FuzzySet(function=Triangular_MF(a=0.42,b=1,c=1), term="high")
FS.add_linguistic_variable("Input", LinguisticVariable([INP1, INP2], universe_of_discourse=[0, 1]))

ONP1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0.58), term="low")
ONP2 = FuzzySet(function=Triangular_MF(a=0.42,b=1,c=1), term="high")
FS.add_linguistic_variable("Output", LinguisticVariable([ONP1, ONP2], universe_of_discourse=[0, 1]))

VINP1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0.58), term="low")
VINP2 = FuzzySet(function=Triangular_MF(a=0.42,b=1,c=1), term="high")
FS.add_linguistic_variable("VInput", LinguisticVariable([VINP1, VINP2], universe_of_discourse=[0, 1]))

VONP1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0.58), term="low")
VONP2 = FuzzySet(function=Triangular_MF(a=0.42,b=1,c=1), term="high")
FS.add_linguistic_variable("VOutput", LinguisticVariable([VONP1, VONP2], universe_of_discourse=[0, 1]))

N1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0.375), term="bad")
N2 = FuzzySet(function=Triangular_MF(a=0.125,b=0.5,c=0.875), term="average")
N3 = FuzzySet(function=Triangular_MF(a=0.675,b=1,c=1), term="good")
FS.add_linguistic_variable("Network", LinguisticVariable([N1, N2, N3], universe_of_discourse=[0, 1]))

#Subsystem Rule2

R17 = "IF (Input IS low) AND (Output IS low) THEN (Network IS bad)"
R18 = "IF (Input IS low) AND (Output IS high) THEN (Network IS average)"
R19 = "IF (Input IS high) AND (Output IS low) THEN (Network IS average)"
R20 = "IF (Input IS high) AND (Output IS high) THEN (Network IS good)"
R21 = "IF (VInput IS low) AND (VOutput IS low) THEN (Network IS bad)"
R22 = "IF (VInput IS low) AND (VOutput IS high) THEN (Network IS average)"
R23 = "IF (VInput IS high) AND (VOutput IS low) THEN (Network IS average)"  
R24 = "IF (VInput IS high) AND (VOutput IS high) THEN (Network IS good)"
R25 = "IF (Input IS low) AND (VOutput IS low) THEN (Network IS bad)"
R26 = "IF (Input IS low) AND (VOutput IS high) THEN (Network IS average)"
R27 = "IF (Input IS high) AND (VOutput IS low) THEN (Network IS average)"
R28 = "IF (Input IS high) AND (VOutput IS high) THEN (Network IS good)"
R29 = "IF (VInput IS low) AND (Output IS low) THEN (Network IS bad)"
R30 = "IF (VInput IS low) AND (Output IS high) THEN (Network IS average)"
R31 = "IF (VInput IS high) AND (Output IS low) THEN (Network IS average)"
R32 = "IF (VInput IS high) AND (Output IS high) THEN (Network IS good)"
FS.add_rules([R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, R32])

#Subsystem3

B1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0.58), term="low")
B2 = FuzzySet(function=Triangular_MF(a=0.42,b=1,c=1), term="high")
FS.add_linguistic_variable("Bandwidth", LinguisticVariable([B1, B2], universe_of_discourse=[0, 1]))

L1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0.58), term="low")
L2 = FuzzySet(function=Triangular_MF(a=0.42,b=1,c=1), term="high")
FS.add_linguistic_variable("Latency", LinguisticVariable([L1, L2], universe_of_discourse=[0, 1]))

VB1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0.58), term="low")
VB2 = FuzzySet(function=Triangular_MF(a=0.42,b=1,c=1), term="high")
FS.add_linguistic_variable("VBandwidth", LinguisticVariable([VB1, VB2], universe_of_discourse=[0, 1]))

VL1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0.58), term="low")
VL2 = FuzzySet(function=Triangular_MF(a=0.42,b=1,c=1), term="high")
FS.add_linguistic_variable("VLatency", LinguisticVariable([VL1, VL2], universe_of_discourse=[0, 1]))

NS1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0.375), term="bad")
NS2 = FuzzySet(function=Triangular_MF(a=0.125,b=0.5,c=0.875), term="average")
NS3 = FuzzySet(function=Triangular_MF(a=0.675,b=1,c=1), term="good")
FS.add_linguistic_variable("NetworkSpeed", LinguisticVariable([NS1, NS2, NS3], universe_of_discourse=[0, 1]))

#Subsystem Rule3

R33 = "IF (Bandwidth IS low) AND (Latency IS low) THEN (NetworkSpeed IS average)"
R34 = "IF (Bandwidth IS low) AND (Latency IS high) THEN (NetworkSpeed IS bad)"
R35 = "IF (Bandwidth IS high) AND (Latency IS low) THEN (NetworkSpeed IS good)"
R36 = "IF (Bandwidth IS high) AND (Latency IS high) THEN (NetworkSpeed IS average)"
R37 = "IF (VBandwidth IS low) AND (VLatency IS low) THEN (NetworkSpeed IS average)"
R38 = "IF (VBandwidth IS low) AND (VLatency IS high) THEN (NetworkSpeed IS bad)"
R39 = "IF (VBandwidth IS high) AND (VLatency IS low) THEN (NetworkSpeed IS good)"
R40 = "IF (VBandwidth IS high) AND (VLatency IS high) THEN (NetworkSpeed IS average)"
R41 = "IF (Bandwidth IS low) AND (VLatency IS low) THEN (NetworkSpeed IS average)"
R42 = "IF (Bandwidth IS low) AND (VLatency IS high) THEN (NetworkSpeed IS bad)"
R43 = "IF (Bandwidth IS high) AND (VLatency IS low) THEN (NetworkSpeed IS good)"
R44 = "IF (Bandwidth IS high) AND (VLatency IS high) THEN (NetworkSpeed IS average)"
R45 = "IF (VBandwidth IS low) AND (Latency IS low) THEN (NetworkSpeed IS average)"
R46 = "IF (VBandwidth IS low) AND (Latency IS high) THEN (NetworkSpeed IS bad)"
R47 = "IF (VBandwidth IS high) AND (Latency IS low) THEN (NetworkSpeed IS good)"
R48 = "IF (VBandwidth IS high) AND (Latency IS high) THEN (NetworkSpeed IS average)"
FS.add_rules([R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46, R47, R48])

# Consequents
CLP1 = FuzzySet(function=Triangular_MF(a=-1,b=0,c=0), term="decrease")
CLP2 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0), term="keep")
CLP3 = FuzzySet(function=Triangular_MF(a=0,b=0,c=1), term="decrease")
FS.add_linguistic_variable("CLP", LinguisticVariable([CLP1, CLP2, CLP3], universe_of_discourse=[-1, 1]))

R49 = "IF (NetworkSpeed IS good) AND (Network IS good) AND (HW_Usage IS high) THEN (CLP IS decrease)"
R50 = "IF (NetworkSpeed IS good) AND (Network IS good) AND (HW_Usage IS balanced) THEN (CLP IS decrease)"
R51 = "IF (NetworkSpeed IS good) AND (Network IS good) AND (HW_Usage IS low) THEN (CLP IS increase)"
R52 = "IF (NetworkSpeed IS good) AND (Network IS average) AND (HW_Usage IS high) THEN (CLP IS decrease)"
R53 = "IF (NetworkSpeed IS good) AND (Network IS average) AND (HW_Usage IS balanced) THEN (CLP IS keep)"
R54 = "IF (NetworkSpeed IS good) AND (Network IS average) AND (HW_Usage IS low) THEN (CLP IS increase)"
R55 = "IF (NetworkSpeed IS good) AND (Network IS bad) AND (HW_Usage IS high) THEN (CLP IS decrease)"
R56 = "IF (NetworkSpeed IS good) AND (Network IS bad) AND (HW_Usage IS balanced) THEN (CLP IS increase)"
R57 = "IF (NetworkSpeed IS good) AND (Network IS bad) AND (HW_Usage IS low) THEN (CLP IS increase)"
R58 = "IF (NetworkSpeed IS average) AND (Network IS good) AND (HW_Usage IS high) THEN (CLP IS decrease)"
R59 = "IF (NetworkSpeed IS average) AND (Network IS good) AND (HW_Usage IS balanced) THEN (CLP IS keep)"
R60 = "IF (NetworkSpeed IS average) AND (Network IS good) AND (HW_Usage IS low) THEN (CLP IS increase)"
R61 = "IF (NetworkSpeed IS average) AND (Network IS average) AND (HW_Usage IS high) THEN (CLP IS keep)"
R62 = "IF (NetworkSpeed IS average) AND (Network IS average) AND (HW_Usage IS balanced) THEN (CLP IS keep)"
R63 = "IF (NetworkSpeed IS average) AND (Network IS average) AND (HW_Usage IS low) THEN (CLP IS increase)"
R64 = "IF (NetworkSpeed IS average) AND (Network IS bad) AND (HW_Usage IS high) THEN (CLP IS decrease)"
R65 = "IF (NetworkSpeed IS average) AND (Network IS bad) AND (HW_Usage IS balanced) THEN (CLP IS increase)"
R66 = "IF (NetworkSpeed IS average) AND (Network IS bad) AND (HW_Usage IS low) THEN (CLP IS increase)"
R67 = "IF (NetworkSpeed IS bad) AND (Network IS good) AND (HW_Usage IS high) THEN (CLP IS decrease)"
R68 = "IF (NetworkSpeed IS bad) AND (Network IS good) AND (HW_Usage IS balanced) THEN (CLP IS increase)"
R69 = "IF (NetworkSpeed IS bad) AND (Network IS good) AND (HW_Usage IS low) THEN (CLP IS increase)"
R70 = "IF (NetworkSpeed IS bad) AND (Network IS average) AND (HW_Usage IS high) THEN (CLP IS keep)"
R71 = "IF (NetworkSpeed IS bad) AND (Network IS average) AND (HW_Usage IS balanced) THEN (CLP IS increase)"
R72 = "IF (NetworkSpeed IS bad) AND (Network IS average) AND (HW_Usage IS low) THEN (CLP IS increase)"
R73 = "IF (NetworkSpeed IS bad) AND (Network IS bad) AND (HW_Usage IS high) THEN (CLP IS keep)"
R74 = "IF (NetworkSpeed IS bad) AND (Network IS bad) AND (HW_Usage IS balanced) THEN (CLP IS increase)"
R75 = "IF (NetworkSpeed IS bad) AND (Network IS bad) AND (HW_Usage IS low) THEN (CLP IS increase)"
FS.add_rules([R49, R50, R51, R52, R53, R54, R55, R56, R57, R58, R59, R60, R61, R62, R63, R64, R65, R66, R67, R68, R69, R70, R71, R72, R73, R74, R75])

df = pd.read_csv('CINTE24-25_Proj1_SampleData.csv')
input_data = df.iloc[:, :12].values.tolist()

Error = 0
CLP = []
for i in range(len(input_data)):
    FS.set_variable("Memory", input_data[i][0])
    FS.set_variable("Processor", input_data[i][1])
    FS.set_variable("VMemory", input_data[i][2])
    FS.set_variable("VProcessor", input_data[i][3])
    FS.set_variable("Input", input_data[i][4])
    FS.set_variable("Output", input_data[i][5])
    FS.set_variable("VInput", input_data[i][6])
    FS.set_variable("VOutput", input_data[i][7])
    FS.set_variable("Bandwidth", input_data[i][8])
    FS.set_variable("Latency", input_data[i][9])
    FS.set_variable("VBandwidth", input_data[i][10])
    FS.set_variable("VLatency", input_data[i][11])
    
    HW_Usage = FS.Mamdani_inference(["HW_Usage"]) 
    Network = FS.Mamdani_inference(["Network"])
    NetworkSpeed = FS.Mamdani_inference(["NetworkSpeed"])
    
    FS.set_variable("HW_Usage", HW_Usage["HW_Usage"])
    FS.set_variable("Network", Network["Network"])
    FS.set_variable("NetworkSpeed", NetworkSpeed["NetworkSpeed"])
    
    Result = FS.Mamdani_inference(["CLP"])
    CLP.append(Result["CLP"])
    
    Error += (df.iloc[i, 12] - Result["CLP"])**2
    print(str(df.iloc[i, 12]) + "  -  " + str(Result["CLP"]))

MSE = Error/len(input_data)
print("MSE :",MSE)
# print("CLP :",CLP)

