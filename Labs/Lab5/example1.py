import simpful as sf
import matplotlib.pyplot as plt

# A simple fuzzy model describing how the heating power of a gas burner depends on the oxygen supply.

FS = sf.FuzzySystem()

# Define a linguistic variable.
S_1 = sf.FuzzySet( points=[[30, 0],  [50, 1.],  [70, 0]],          term="warm" )
S_2 = sf.FuzzySet( points=[[0, 1], [50, 0]], term="cold" )
S_3 = sf.FuzzySet( points=[[50, 0], [100, 1.]],          term="hot" )
FS.add_linguistic_variable("CoreTemp", sf.LinguisticVariable( [S_1, S_2, S_3] ))

# Define a linguistic variable.
S_1 = sf.FuzzySet( points=[[0.5, 0],  [2, 1.],  [3.5, 0]],          term="normal" )
S_2 = sf.FuzzySet( points=[[0, 1], [1.5, 0]], term="low" )
S_3 = sf.FuzzySet( points=[[2.5, 0], [4, 1.]],          term="turbo" )
FS.add_linguistic_variable("ClockSpeed", sf.LinguisticVariable( [S_1, S_2, S_3] ))

# Define a linguistic variable.
S_1 = sf.FuzzySet( points=[[2500, 0],  [6000, 1]],          term="FAST" )
S_2 = sf.FuzzySet( points=[[0, 1], [3500, 0]], term="LOW" )
#LV3=LinguisticVariable([S_1,S_2], concept="FanSpeed", universe_of_discourse=[0,6000])
FS.add_linguistic_variable("FAN", sf.LinguisticVariable( [S_1, S_2] ))
#LV3.plot


# Define fuzzy rules.
RULE1 = "IF (ClockSpeed IS turbo) THEN (FAN IS FAST)"
RULE2 = "IF (ClockSpeed IS normal) AND (NOT(CoreTemp IS hot)) THEN (FAN IS LOW)"
RULE3 = "IF (ClockSpeed IS normal) AND (CoreTemp IS hot) THEN (FAN IS FAST)"
RULE4 = "IF (ClockSpeed IS low) AND (NOT(CoreTemp IS hot)) THEN (FAN IS LOW)"
RULE5 = "IF (ClockSpeed IS low) AND (CoreTemp IS hot) THEN (FAN IS FAST)"

FS.add_rules([RULE1, RULE2, RULE3, RULE4, RULE5])

# Set antecedents values, perform inference and print output values.
FS.set_variable("CoreTemp",50)
FS.set_variable("ClockSpeed",0.5)
 
print (FS.inference())

