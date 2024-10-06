from simpful import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

def create_dataset(iteration):
    random_data_set = []
    
    for i in range(iteration):
        ar=random.choices(range(200), k=6)
        ar = [(ar[i]-100)/100 for i in range(len(ar))]
        random_data_set.append({'Memory':ar[0], 'Processor':ar[1], 'Input':ar[2], 'Output':ar[3], 'Bandwidth':ar[4], 'Latency':ar[5]})
    data_set = pd.DataFrame(random_data_set, columns=['Memory', 'Processor', 'Input', 'Output', 'Bandwidth', 'Latency'])
    return data_set

FS = FuzzySystem()

# Antecedents
# HW Subsystem
M1 = FuzzySet( points=[[0, 1] , [0.35,1], [0.5, 0]], term="low" )
M2 = FuzzySet( points=[[0.3, 0], [0.45, 1],  [0.55,1] ,[0.65,0]], term="average" )
M3 = FuzzySet( points=[[0.6, 0], [0.7, 1.],[0.85,0]],term="high" )
M4 = FuzzySet( points=[[0.8, 0], [0.85, 1]], term="critical" )
FS.add_linguistic_variable("Memory", LinguisticVariable([M1, M2, M3, M4], universe_of_discourse=[0, 1]))

P1 = FuzzySet( points=[[0, 1], [0.35,1],  [0.5, 0]], term="low" )
P2 = FuzzySet( points=[[0.3, 0], [0.45, 1],  [0.55,1] ,[0.65,0]], term="average" )
P3 = FuzzySet( points=[[0.6, 0], [0.70, 1.],[0.85,0]],term="high" )
P4 = FuzzySet( points=[[0.8, 0], [0.85, 1]], term="critical" )
FS.add_linguistic_variable("Processor", LinguisticVariable([P1, P2, P3, P4], universe_of_discourse=[0, 1]))

SP1 = FuzzySet( points=[[0, 1], [0.35,1],  [0.49, 0]] , term="low")
SP2 = FuzzySet(points=[[0.3, 0], [0.49, 1],  [0.51,1] ,[0.6,0]], term="balanced")
SP3 = FuzzySet(points=[[0.6, 0], [0.7, 1],  [0.85,1] ,[0.88,0]], term="high")
SP4 = FuzzySet(points=[[0.85, 0], [0.88, 1]], term="critical")
FS.add_linguistic_variable("HW_Usage", LinguisticVariable([SP1, SP2, SP3, SP4], universe_of_discourse=[0, 1]))

# HW Subsystem Rules

R_CRITICAL = "IF (Memory IS critical) OR (Processor IS critical) THEN (HW_Usage IS critical)"

R_HIGH = "IF (Processor IS high) AND (Memory IS high) THEN (HW_Usage IS high)"

R_BAL1 = "IF (Processor IS average) AND ((Memory IS average) OR (Memory IS high)) THEN (HW_Usage IS balanced)"
R_BAL2 ="IF (Memory IS average) AND ((Processor IS average) OR (Processor IS high)) THEN (HW_Usage IS balanced)"

R_LOW1 = "IF (Memory IS low) AND (NOT(Processor IS critical)) THEN (HW_Usage IS low)"
R_LOW2 = "IF (Processor IS low) AND (NOT(Memory IS critical)) THEN (HW_Usage IS low)"



FS.add_rules([R_CRITICAL, R_HIGH, R_BAL1, R_BAL2, R_LOW1, R_LOW2])

#Subsystem2
INP1 = FuzzySet( points=[[0, 1],  [0.25, 0]], term="low" )
INP2 = FuzzySet( points=[[0.2, 0], [0.35, 1], [0.60,0]], term="average" )
INP3 = FuzzySet( points=[[0.5, 0], [0.70, 1.],[0.85,0]], term="high" )
INP4 = FuzzySet( points=[[0.7, 0], [1, 1]], term="very_high" )
FS.add_linguistic_variable("Input", LinguisticVariable([INP1, INP2, INP3, INP4], universe_of_discourse=[0, 1]))

ONP1 = FuzzySet( points=[[0, 1],  [0.25, 0]], term="low" )
ONP2 = FuzzySet( points=[[0.2, 0], [0.35, 1], [0.60,0]], term="average" )
ONP3 = FuzzySet( points=[[0.5, 0], [0.70, 1.],[0.85,0]], term="high" )
ONP4 = FuzzySet( points=[[0.75, 0], [1, 1]], term="very_high" )
FS.add_linguistic_variable("Output", LinguisticVariable([ONP1, ONP2, ONP3, ONP4], universe_of_discourse=[0, 1]))


N1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=0.375), term="congested")
N2 = FuzzySet(function=Triangular_MF(a=0.25,b=0.5,c=0.75), term="balanced")
N3 = FuzzySet(function=Triangular_MF(a=0.675,b=0.8,c=1), term="de_congested")
FS.add_linguistic_variable("Network", LinguisticVariable([N1, N2, N3], universe_of_discourse=[0, 1]))

#Subsystem Rule2

R_DECON1 = "IF NOT(Input IS very_high) AND (Output IS very_high) THEN (Network IS de_congested)"
R_DECON2 = "IF (Input IS average) AND (Output IS high) THEN (Network IS de_congested)"
R_DECON3 = "IF (Input IS low) THEN (Network IS de_congested)"


R_BAL1 = "IF (Input IS average) AND (Output IS average) THEN (Network IS balanced)"
R_BAL2 = "IF (Input IS high) AND (Output IS high) THEN (Network IS balanced)"
R_BAL3 = "IF (Input IS very_high) AND (Output IS very_high) THEN (Network IS balanced)"

R_CON1 = "IF (Output IS low) AND (NOT(Input IS low)) THEN (Network IS congested)"
R_CON2 = "IF ((Input IS high) OR (Input IS very_high)) AND (Output IS average) THEN (Network IS congested)"
R_CON3 = "IF (Input IS very_high) AND (NOT(Output IS very_high)) THEN (Network IS congested)"


FS.add_rules([R_DECON1, R_DECON2, R_DECON3, R_BAL1, R_BAL2, R_BAL3, R_CON1, R_CON2, R_CON3])

#Subsystem3

B1 = FuzzySet( points=[[0, 1],  [0.25, 0]], term="low" )
B2 = FuzzySet( points=[[0.2, 0], [0.4, 1], [0.55,0]], term="average" )
B3 = FuzzySet( points=[[0.5, 0], [0.6, 1.],[0.725,0]], term="high" )
B4 = FuzzySet( points=[[0.65, 0], [0.8, 1]], term="very_high" )
FS.add_linguistic_variable("Bandwidth", LinguisticVariable([B1, B2, B3, B4], universe_of_discourse=[0, 1]))

TNC1 = FuzzySet(points=[[0, 1], [0.3, 1.],[0.5,0]], term="fullycongested")
TNC2 = FuzzySet(function=Triangular_MF(a=0.3,b=0.5,c=0.7), term="balanced")
TNC3 = FuzzySet(function=Triangular_MF(a=0.5,b=0.7,c=1), term="decongestable")
FS.add_linguistic_variable("TrueNetCongestion", LinguisticVariable([TNC1, TNC2, TNC3], universe_of_discourse=[0, 1]))



#Subsystem Rule3

R_NCON1 = "IF (Network IS congested) AND (NOT(Bandwidth IS very_high)) THEN (TrueNetCongestion IS fullycongested)"
R_NCON2 = "IF (Network IS congested) AND (Bandwidth IS very_high) THEN (TrueNetCongestion IS decongestable)" 

R_NBAL1 = "IF (Network IS balanced) AND (NOT((Bandwidth IS low) OR (Bandwidth IS very_high))) THEN (TrueNetCongestion IS balanced)"
R_NBAL2 = "IF (Network IS balanced) AND (Bandwidth IS very_high) THEN (TrueNetCongestion IS decongestable)"
R_NBAL3 = "IF (Network IS balanced) AND (Bandwidth IS low) THEN (TrueNetCongestion IS fullycongested)"

R_NDECON1 = "IF (Network IS de_congested) AND (Bandwidth IS low) THEN (TrueNetCongestion IS decongestable)"
R_NDECON2 = "IF (Network IS de_congested) AND (NOT(Bandwidth IS low)) THEN (TrueNetCongestion IS balanced)" 



FS.add_rules([R_NCON1, R_NCON2, R_NBAL1, R_NBAL2, R_NBAL3, R_NDECON1, R_NDECON2])

# Super-Fuzzy System 

#Big Power variable
L1 = FuzzySet( points=[[0, 1], [0.25, 1], [0.5, 0]], term="low" )
L2 = FuzzySet( points=[[0.25, 0], [0.5, 1], [0.7, 0.5],[0.9,0]], term="average" )
L3 = FuzzySet( points=[[0.5, 0], [0.7, 0.5], [0.8, 0.975], [0.9, 1]], term="high" )
FS.add_linguistic_variable("Latency", LinguisticVariable([L1, L2, L3], universe_of_discourse=[0, 1]))


# Consequents
CLP1 = FuzzySet(function=Triangular_MF(a=-1,b=-1,c=-0.45), term="decrease")
CLP2 = FuzzySet(function=Triangular_MF(a=-0.15,b=0,c=0.15), term="keep")
CLP3 = FuzzySet(function=Triangular_MF(a=0.45,b=1,c=1), term="increase")
FS.add_linguistic_variable("CLP", LinguisticVariable([CLP1, CLP2, CLP3], universe_of_discourse=[-1, 1]))

R_HWLow= "IF (HW_Usage IS low) THEN (CLP IS increase)"
R_HWCritical= "IF (HW_Usage IS critical) THEN (CLP IS decrease)"
R_HighLatency1 = "IF (Latency IS high) AND (NOT(HW_Usage IS critical)) THEN (CLP IS increase)"
R_NetDeCongested = "IF (TrueNetCongestion IS decongestable) AND (NOT((HW_Usage IS low) OR (HW_Usage IS balanced) OR (Latency IS high))) THEN (CLP IS decrease)"
R_NetBalDecon = "IF (TrueNetCongestion IS decongestable) AND (HW_Usage IS balanced) AND (NOT(Latency IS high)) THEN (CLP IS keep)"
R_NetCongested = "IF (TrueNetCongestion IS fullycongested) AND (NOT(HW_Usage IS critical)) THEN (CLP IS increase)"
R_HWBAL1 = "IF (HW_Usage IS balanced) AND (NOT((TrueNetCongestion IS decongestable) AND (NOT(Latency IS high)))) THEN (CLP IS increase)"
R_HWBAL2 = "IF (HW_Usage IS balanced) AND ((TrueNetCongestion IS balanced) AND (NOT(Latency IS high))) THEN (CLP IS keep)"
R_HWHi1 = "IF (HW_Usage IS high) AND ((TrueNetCongestion IS balanced) AND (Latency IS low)) THEN (CLP IS decrease)"
R_HWHi2 = "IF (HW_Usage IS high) AND ((TrueNetCongestion IS balanced) AND (Latency IS average)) THEN (CLP IS keep)"
FS.add_rules([R_HWLow, R_HWCritical, R_HighLatency1, R_NetCongested,R_NetBalDecon,R_NetDeCongested, R_HWBAL1, R_HWBAL2, R_HWHi1, R_HWHi2])

Flag = False
if(Flag == True):
    randomset = create_dataset(10000)
    DataSet = []

for i in range(len(randomset)):
        FS.set_variable("Memory", randomset.iloc[i, 0])  # Accessing row i, column 0
        FS.set_variable("Processor", randomset.iloc[i, 1])  # Accessing row i, column 1
        FS.set_variable("Input", randomset.iloc[i, 2])  # and so on for the remaining columns
        FS.set_variable("Output", randomset.iloc[i, 3])
        FS.set_variable("Bandwidth", randomset.iloc[i, 4])
        FS.set_variable("Latency", randomset.iloc[i, 5])
        
        HW_Usage = FS.Mamdani_inference(["HW_Usage"]) 
        Network = FS.Mamdani_inference(["Network"])
        FS.set_variable("Network", Network["Network"])
        TrueNetCongestion = FS.Mamdani_inference(["TrueNetCongestion"])
        FS.set_variable("HW_Usage", HW_Usage["HW_Usage"])
        FS.set_variable("TrueNetCongestion", TrueNetCongestion["TrueNetCongestion"])
        Result = FS.Mamdani_inference(["CLP"])
        
        DataSet.append([randomset.iloc[i, 0], randomset.iloc[i, 1], randomset.iloc[i, 2], randomset.iloc[i, 3], randomset.iloc[i, 4], randomset.iloc[i, 5], Result["CLP"]])
    DataSet = pd.DataFrame(DataSet, columns=['Memory', 'Processor', 'Input', 'Output', 'Bandwidth', 'Latency', 'CLP'])
    DataSet.to_csv('Project_1\\randomset.csv', index=False)
    print("Saved")

df = pd.read_csv('CINTE24-25_Proj1_SampleData.csv')
#df = pd.read_csv('Project_1\\Random_IoT_Data_With_CLP.csv')
input_data = df.iloc[:, :12].values.tolist()

Error = 0
Intermediates = []
for i in range(len(input_data)):
    # FS.set_variable("Memory", input_data[i][0])
    # FS.set_variable("Processor", input_data[i][1])
    
    
    # FS.set_variable("Input", input_data[i][4])
    # FS.set_variable("Output", input_data[i][5])
    
    
    # FS.set_variable("Bandwidth", input_data[i][8])
    # FS.set_variable("Latency", input_data[i][9])
    
    FS.set_variable("Memory", input_data[i][0])
    FS.set_variable("Processor", input_data[i][1])
    
    
    FS.set_variable("Input", input_data[i][2])
    FS.set_variable("Output", input_data[i][3])
    
    
    FS.set_variable("Bandwidth", input_data[i][4])
    FS.set_variable("Latency", input_data[i][5])

    
    HW_Usage = FS.Mamdani_inference(["HW_Usage"]) 
    
    Network = FS.Mamdani_inference(["Network"])
    FS.set_variable("Network", Network["Network"])
    
    TrueNetCongestion = FS.Mamdani_inference(["TrueNetCongestion"])
    
    
    
    
    FS.set_variable("HW_Usage", HW_Usage["HW_Usage"])
    FS.set_variable("TrueNetCongestion", TrueNetCongestion["TrueNetCongestion"])
    
    Result = FS.Mamdani_inference(["CLP"])
    Intermediates.append([HW_Usage["HW_Usage"], Network["Network"], TrueNetCongestion["TrueNetCongestion"], Result["CLP"], df.iloc[i, 12]])
    
    Error += (df.iloc[i, 12] - Result["CLP"])**2
    # print(str(df.iloc[i, 12]) + "  -  " + str(Result["CLP"]))
    
df1 = pd.DataFrame(Intermediates)
df1.columns = ["HW_Usage", "Network", "TrueNetCongestion", "Result", "CLPVariation"]
df.drop(columns=['V_MemoryUsage', 'V_ProcessorLoad','V_InpNetThroughput', 'V_OutNetThroughput', 'V_OutBandwidth','V_Latency','CLPVariation'], inplace=True)
print(df)
print(df1)

MSE = Error/len(input_data)
print("MSE :",MSE)

array = []
for i in range(df1.shape[0]):
    array.append(i)
    
plt.scatter(array,df1["Result"],label='Computed CLP Var.')
plt.scatter(array,df1["CLPVariation"],label='Reference CLP Var.')
plt.legend()
plt.show()

def plot_fuzzy_set(fuzzy_set, color):
    # Unpack points from the fuzzy set
    x_vals = [point[0] for point in fuzzy_set._points]
    y_vals = [point[1] for point in fuzzy_set._points]
    
    # Plot the fuzzy set
    plt.plot(x_vals, y_vals, label=fuzzy_set._term, color=color)

# Assuming M1, M2, M3, and M4 are already defined as FuzzySet objects
colors = ['blue', 'green', 'orange', 'red']

# Plot each fuzzy set with its corresponding color
# plot_fuzzy_set(SP1, colors[0])
# plot_fuzzy_set(SP2, colors[1])
# plot_fuzzy_set(SP3, colors[2])
# plot_fuzzy_set(SP4, colors[3])

# # Add plot labels and legends
# plt.title("Fuzzy Sets")
# plt.xlabel("X (input)")
# plt.ylabel("Membership Degree")
# plt.legend()

# #Show the plot
# plt.grid(True)
# plt.show()

