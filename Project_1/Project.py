import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simpful as sf

df = pd.read_csv('CINTE24-25_Proj1_SampleData.csv')

#[----[Hardware FIS]----]
# Define Processor linguistic variable input.
S_1 = sf.FuzzySet( points=[[0, 1],  [0.25, 0]],          term="Low" )
S_2 = sf.FuzzySet( points=[[0.2, 0], [0.35, 1], [0.60,0]], term="Average" )
S_3 = sf.FuzzySet( points=[[0.5, 0], [0.70, 1.]],          term="High" )
S_3 = sf.FuzzySet( points=[[0.85, 0], [1, 1]],          term="Critical" )
FS.add_linguistic_variable("ProcLoad", sf.LinguisticVariable( [S_1, S_2, S_3] ))
# Define Memory linguistic variable input.
S_1 = sf.FuzzySet( points=[[0, 1],  [0.25, 0],          term="Low" )
S_2 = sf.FuzzySet( points=[[0.2, 0], [0.35, 1], [0.60,0]], term="Average" )
S_3 = sf.FuzzySet( points=[[0.5, 0], [0.70, 1.]],          term="High" )
S_3 = sf.FuzzySet( points=[[0.85, 0], [1, 1]],          term="Critical" )
FS.add_linguistic_variable("MemUsage", sf.LinguisticVariable( [S_1, S_2, S_3] ))
# Define HWLoad linguistic variable .
S_1 = sf.FuzzySet( points=[[0, 1],  [0.25, 0],          term="Low" )
S_2 = sf.FuzzySet( points=[[0.2, 0], [0.35, 1], [0.60,0]], term="Average" )
S_3 = sf.FuzzySet( points=[[0.5, 0], [0.70, 1.]],          term="High" )
S_3 = sf.FuzzySet( points=[[0.85, 0], [1, 1]],          term="Critical" )
FS.add_linguistic_variable("HWLoad", sf.LinguisticVariable( [S_1, S_2, S_3] ))

#[----[NetworkCongestion FIS]----]

#[----[Network FIS]----]

#[----[CLP FIS]----]



# Define a linguistic variable.
S_1 = sf.FuzzySet( points=[[0.9, 0],  [1, 1]],          term="Infinite" )
S_2 = sf.FuzzySet( points=[[0, 1], [3500, 0]], term="High" )
S_2 = sf.FuzzySet( points=[[0, 1], [3500, 0]], term="Average" )
S_2 = sf.FuzzySet( points=[[0, 1], [3500, 0]], term="Low" )
FS.add_linguistic_variable("Latency", sf.LinguisticVariable( [S_1, S_2] ))


# Define a linguistic variable.
S_1 = sf.FuzzySet( points=[[0, 1],  [0.05,0 ]],          term="Minimal" )
S_1 = sf.FuzzySet( points=[[, ],  [, ]],          term="Reduced" )
S_1 = sf.FuzzySet( points=[[, ],  [, ]],          term="Average" )
S_2 = sf.FuzzySet( points=[[, ], [, ]], term="High" )
FS.add_linguistic_variable("CLP", sf.LinguisticVariable( [S_1, S_2] ))


# Define fuzzy rules.


 
print (FS.inference())
