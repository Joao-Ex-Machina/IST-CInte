from simpful import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


FS = FuzzySystem()

# Antecedents
# HW Subsystem
M1 = FuzzySet( points=[[0, 1],  [0.25, 0]], term="low" )
M2 = FuzzySet( points=[[0.2, 0], [0.35, 1], [0.60,0]], term="average" )
M3 = FuzzySet( points=[[0.5, 0], [0.70, 1.],[0.85,0]],term="high" )
M4 = FuzzySet( points=[[0.75, 0], [1, 1]], term="critical" )
FS.add_linguistic_variable("Memory", LinguisticVariable([M1, M2, M3, M4], universe_of_discourse=[0, 1]))

P1 = FuzzySet( points=[[0, 1],  [0.25, 0]], term="low" )
P2 = FuzzySet( points=[[0.2, 0], [0.35, 1], [0.60,0]], term="average" )
P3 = FuzzySet( points=[[0.5, 0], [0.70, 1.],[0.85,0]],term="high" )
P4 = FuzzySet( points=[[0.75, 0], [1, 1]], term="critical" )
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
INP1 = FuzzySet( points=[[0, 1],  [0.25, 0]], term="low" )
INP2 = FuzzySet( points=[[0.2, 0], [0.35, 1], [0.60,0]], term="average" )
INP3 = FuzzySet( points=[[0.5, 0], [0.70, 1.],[0.85,0]], term="high" )
INP4 = FuzzySet( points=[[0.75, 0], [1, 1]], term="very_high" )
FS.add_linguistic_variable("Input", LinguisticVariable([INP1, INP2, INP3, INP4], universe_of_discourse=[0, 1]))

ONP1 = FuzzySet( points=[[0, 1],  [0.25, 0]], term="low" )
ONP2 = FuzzySet( points=[[0.2, 0], [0.35, 1], [0.60,0]], term="average" )
ONP3 = FuzzySet( points=[[0.5, 0], [0.70, 1.],[0.85,0]], term="high" )
ONP4 = FuzzySet( points=[[0.75, 0], [1, 1]], term="very_high" )
FS.add_linguistic_variable("Output", LinguisticVariable([ONP1, ONP2, ONP3, ONP4], universe_of_discourse=[0, 1]))


N1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0.375), term="slow")
N2 = FuzzySet(function=Triangular_MF(a=0.125,b=0.5,c=0.875), term="average")
N3 = FuzzySet(function=Triangular_MF(a=0.675,b=1,c=1), term="fast")
FS.add_linguistic_variable("Network", LinguisticVariable([N1, N2, N3], universe_of_discourse=[0, 1]))

#Subsystem Rule2

R_FAST1 = "IF NOT(Input IS very_high) AND (Output IS very_high) THEN (Network IS fast)"
R_FAST2 = "IF ((Input IS low) OR (Input IS average)) AND (Output IS high) THEN (Network IS fast)"

R_AVG1 = "IF ((Input IS low) OR (Input IS average)) AND (Output IS average) THEN (Network IS average)"
R_AVG2 = "IF (Input IS high) AND (Output IS high) THEN (Network IS average)"
R_AVG3 = "IF (Input IS very_high) AND (Output IS very_high) THEN (Network IS average)"

R_SLOW1 = "IF (Output IS low) THEN (Network IS slow)"
R_SLOW2 = "IF ((Input IS high) OR (Input IS very_high)) AND (Output IS average) THEN (Network IS slow)"

FS.add_rules([R_FAST1, R_FAST2, R_AVG1, R_AVG2, R_AVG3, R_SLOW1, R_SLOW2])

#Subsystem3

B1 = FuzzySet( points=[[0, 1],  [0.25, 0]], term="low" )
B2 = FuzzySet( points=[[0.2, 0], [0.35, 1], [0.60,0]], term="average" )
B3 = FuzzySet( points=[[0.5, 0], [0.70, 1.],[0.85,0]], term="high" )
B4 = FuzzySet( points=[[0.75, 0], [1, 1]], term="very_high" )
FS.add_linguistic_variable("Bandwidth", LinguisticVariable([B1, B2, B3, B4], universe_of_discourse=[0, 1]))

L1 = FuzzySet( points=[[0, 1],  [0.25, 0]], term="low" )
L2 = FuzzySet( points=[[0.2, 0], [0.35, 1], [0.60,0]], term="average" )
L3 = FuzzySet( points=[[0.5, 0], [0.70, 1.],[0.85,0]], term="high" )
L4 = FuzzySet( points=[[0.75, 0], [1, 1]], term="very_high" )
FS.add_linguistic_variable("Latency", LinguisticVariable([L1, L2, L3, L4], universe_of_discourse=[0, 1]))

NS1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0.35), term="slow")
NS2 = FuzzySet(function=Triangular_MF(a=0.30,b=0.5,c=0.75), term="average")
NS3 = FuzzySet(function=Triangular_MF(a=0.70,b=1,c=1), term="fast")
FS.add_linguistic_variable("NetworkSpeed", LinguisticVariable([NS1, NS2, NS3], universe_of_discourse=[0, 1]))

#Subsystem Rule3

R_FAST3 = "IF ((Bandwidth IS very_high) OR (Bandwidth IS high)) AND ((Latency IS low) OR (Latency IS average)) THEN (NetworkSpeed IS fast)"

R_AVG4 = "IF (Bandwidth IS average) AND ((Latency IS low) OR (Latency IS average)) THEN (NetworkSpeed IS average)"
R_AVG5 = "IF (Bandwidth IS very_high) AND (Latency IS high) THEN (NetworkSpeed IS average)"

R_SLOW3 = "IF (Bandwidth IS low) OR (Latency IS very_high) THEN (NetworkSpeed IS slow)"
R_SLOW4 = "IF ((Bandwidth IS average) OR (Bandwidth IS high)) AND (Latency IS high) THEN (NetworkSpeed IS slow)"

FS.add_rules([R_FAST3, R_AVG4, R_AVG5, R_SLOW3, R_SLOW4])

# Consequents
CLP1 = FuzzySet(function=Triangular_MF(a=-1,b=-1,c=-0.35), term="increase")
CLP2 = FuzzySet(function=Triangular_MF(a=-0.45,b=0,c=0.45), term="keep")
CLP3 = FuzzySet(function=Triangular_MF(a=0.35,b=1,c=1), term="decrease")
FS.add_linguistic_variable("CLP", LinguisticVariable([CLP1, CLP2, CLP3], universe_of_discourse=[-1, 1]))

R49 = "IF (NetworkSpeed IS fast) AND (Network IS fast) AND (HW_Usage IS high) THEN (CLP IS decrease)"
R50 = "IF (NetworkSpeed IS fast) AND (Network IS fast) AND (HW_Usage IS balanced) THEN (CLP IS decrease)"
R51 = "IF (NetworkSpeed IS fast) AND (Network IS fast) AND (HW_Usage IS low) THEN (CLP IS increase)"
R52 = "IF (NetworkSpeed IS fast) AND (Network IS average) AND (HW_Usage IS high) THEN (CLP IS decrease)"
R53 = "IF (NetworkSpeed IS fast) AND (Network IS average) AND (HW_Usage IS balanced) THEN (CLP IS keep)"
R54 = "IF (NetworkSpeed IS fast) AND (Network IS average) AND (HW_Usage IS low) THEN (CLP IS increase)"
R55 = "IF (NetworkSpeed IS fast) AND (Network IS slow) AND (HW_Usage IS high) THEN (CLP IS decrease)"
R56 = "IF (NetworkSpeed IS fast) AND (Network IS slow) AND (HW_Usage IS balanced) THEN (CLP IS increase)"
R57 = "IF (NetworkSpeed IS fast) AND (Network IS slow) AND (HW_Usage IS low) THEN (CLP IS increase)"

R58 = "IF (NetworkSpeed IS average) AND (Network IS fast) AND (HW_Usage IS high) THEN (CLP IS decrease)"

R59 = "IF (NetworkSpeed IS average) AND (Network IS fast) AND (HW_Usage IS balanced) THEN (CLP IS keep)"

R60 = "IF (NetworkSpeed IS average) AND (Network IS fast) AND (HW_Usage IS low) THEN (CLP IS decrease)"

R61 = "IF (NetworkSpeed IS average) AND (Network IS average) AND (HW_Usage IS high) THEN (CLP IS increase)"

R62 = "IF (NetworkSpeed IS average) AND (Network IS average) AND (HW_Usage IS balanced) THEN (CLP IS decrease)"

R63 = "IF (NetworkSpeed IS average) AND (Network IS average) AND (HW_Usage IS low) THEN (CLP IS increase)"
R64 = "IF (NetworkSpeed IS average) AND (Network IS slow) AND (HW_Usage IS high) THEN (CLP IS decrease)"
R65 = "IF (NetworkSpeed IS average) AND (Network IS slow) AND (HW_Usage IS balanced) THEN (CLP IS increase)"
R66 = "IF (NetworkSpeed IS average) AND (Network IS slow) AND (HW_Usage IS low) THEN (CLP IS increase)"

R67 = "IF (NetworkSpeed IS slow) AND (Network IS fast) AND (HW_Usage IS high) THEN (CLP IS increase)"

R68 = "IF (NetworkSpeed IS slow) AND (Network IS fast) AND (HW_Usage IS balanced) THEN (CLP IS increase)"

R69 = "IF (NetworkSpeed IS slow) AND (Network IS fast) AND (HW_Usage IS low) THEN (CLP IS increase)"

R70 = "IF (NetworkSpeed IS slow) AND (Network IS average) AND (HW_Usage IS high) THEN (CLP IS increase)"

R71 = "IF (NetworkSpeed IS slow) AND (Network IS average) AND (HW_Usage IS balanced) THEN (CLP IS decrease)"

R72 = "IF (NetworkSpeed IS slow) AND (Network IS average) AND (HW_Usage IS low) THEN (CLP IS increase)"
R73 = "IF (NetworkSpeed IS slow) AND (Network IS slow) AND (HW_Usage IS high) THEN (CLP IS keep)"
R74 = "IF (NetworkSpeed IS slow) AND (Network IS slow) AND (HW_Usage IS balanced) THEN (CLP IS increase)"
R75 = "IF (NetworkSpeed IS slow) AND (Network IS slow) AND (HW_Usage IS low) THEN (CLP IS increase)"
FS.add_rules([R49, R50, R51, R52, R53, R54, R55, R56, R57, R58, R59, R60, R61, R62, R63, R64, R65, R66, R67, R68, R69, R70, R71, R72, R73, R74, R75])

df = pd.read_csv('CINTE24-25_Proj1_SampleData.csv')

input_data = df.iloc[:, :12].values.tolist()

Error = 0
Intermediates = []
for i in range(len(input_data)):
    FS.set_variable("Memory", input_data[i][0])
    FS.set_variable("Processor", input_data[i][1])
    
    
    FS.set_variable("Input", input_data[i][4])
    FS.set_variable("Output", input_data[i][5])
    
    
    FS.set_variable("Bandwidth", input_data[i][8])
    FS.set_variable("Latency", input_data[i][9])

    
    HW_Usage = FS.Mamdani_inference(["HW_Usage"]) 
    Network = FS.Mamdani_inference(["Network"])
    NetworkSpeed = FS.Mamdani_inference(["NetworkSpeed"])
    
    
    
    
    FS.set_variable("HW_Usage", HW_Usage["HW_Usage"])
    FS.set_variable("Network", Network["Network"])
    FS.set_variable("NetworkSpeed", NetworkSpeed["NetworkSpeed"])
    
    Result = FS.Mamdani_inference(["CLP"])
    Intermediates.append([HW_Usage["HW_Usage"], Network["Network"], NetworkSpeed["NetworkSpeed"], Result["CLP"], df.iloc[i, 12]])
    
    Error += (df.iloc[i, 12] - Result["CLP"])**2
    # print(str(df.iloc[i, 12]) + "  -  " + str(Result["CLP"]))
    
df1 = pd.DataFrame(Intermediates)
df1.columns = ["HW_Usage", "Network", "NetworkSpeed", "Result", "CLPVariation"]
df.drop(columns=['V_MemoryUsage', 'V_ProcessorLoad','V_InpNetThroughput', 'V_OutNetThroughput', 'V_OutBandwidth','V_Latency','CLPVariation'], inplace=True)
print(df)
print(df1)

MSE = Error/len(input_data)
print("MSE :",MSE)


def plot_fuzzy_set(fuzzy_set, color):
    # Unpack points from the fuzzy set
    x_vals = [point[0] for point in fuzzy_set._points]
    y_vals = [point[1] for point in fuzzy_set._points]
    
    # Plot the fuzzy set
    plt.plot(x_vals, y_vals, label=fuzzy_set._term, color=color)

# Assuming M1, M2, M3, and M4 are already defined as FuzzySet objects
colors = ['blue', 'green', 'orange', 'red']

# Plot each fuzzy set with its corresponding color
plot_fuzzy_set(M1, colors[0])
plot_fuzzy_set(M2, colors[1])
plot_fuzzy_set(M3, colors[2])
plot_fuzzy_set(M4, colors[3])

# Add plot labels and legends
plt.title("Fuzzy Sets")
plt.xlabel("X (input)")
plt.ylabel("Membership Degree")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

