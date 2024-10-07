---
title: "Applied Computational Intelligence Project Report"
author: "David Gao Xia nº 99907 , João Barreiros C. Rodrigues nº 99668"
date: "Oct. 2024"
output: 
  pdf_document: 
    keep_tex: yes
header-includes:
---

## Fuzzy System

### General Architecture

![Fuzzy System Architecture](./images/FuzzyArch.png){width=85%}

From the 12 given variables we opted to only kept the 6 non-variations, since a Fuzzy System with 6 inputs will be smaller (and therefore simpler to fine-tune and write rules for) and the variations give no absolute current information which is the basis for our decision system.

Similiar to the idea of Comb's method, we broke our FIS (Fuzzy Inference System) into smaller, 2 to 3 input FISs. This allowed to have some granularity in their input and output membership sets without the risk of exploding. Our Architecture is composed 4 FIS.

\pagebreak

### Hardware Usage FIS

This system aggregates the hardware-related inputs: **Memory Usage** and **Processor Load**
Our rules set is fairly simple, having a 4 linguistic term granularity for inputs and output

|   |  **Memory**  |          | **Processor** | **Load** |              |
|---|:------------:|:--------:|:-------------:|:--------:|:------------:|
|   |              | **Low**  |  **Average**  | **High** | **Critical** |
|   |   **Low**    |   Low    |      Low      |   Low    |   Critical   |
|   | **Average ** |   Low    |   Balanced    | Balanced |   Critical   |
|   |   **High**   |   Low    |   Balanced    |   High   |   Critical   |
|   | **Critical** | Critical |   Critical    | Critical |   Critical   |

And so are the membership function for this FIS, following the Project guidelines and common sense

![Membership Functions for Hardware Input and Output variables](./images/HW_FIS_MF.png){}

\pagebreak

### Network Congestion FIS

This system outputs a "quantitative" network congestion value, based on the I/O throughput of a network.
Has with the last FIS both inputs have a granularity of 4 linguistic terms, however the output has only 3, since this seemed granular enough and simplified our FIS.
![Membership Function for Net. Congestion Input and Output variables](./images/Net_FIS_MF.png){}

|   | **Input Th.** |              |  **Output**  |   **Th.**    |               |
|---|:-------------:|:------------:|:------------:|:------------:|:-------------:|
|   |               |   **Low**    | **Average**  |   **High**   | **Very High** |
|   |    **Low**    | Decongested | Decongested | Decongested | Decongested  |
|   | **Average **  |     Congested      |   Balanced   | Decongested | Decongested  |
|   |   **High**    |     Congested      |  Congested   |     High     | Decongested  |
|   | **Very High** |     Congested      |  Congested   |   Balanced   |   Balanced    |


\pagebreak

### True Network Congestion FIS

Different from the last FIS the output of this System is not as much a "quantitative" variable as it is an indicative of an action to be taken.
This FIS combined the previously produced output with the Output Bandwidth variable, in order to determine if the Network is too congested for data forwarding (Fully Congested) and therefore should prioritize local data processing, able to be de-congested if data forwarding is prioritized  or if it is overall balanced and does not require further action. 

![Membership Function for True Net. Congestion Input and Output variables](./images/TNC_FIS_MF.png){}

|   | **Out. Bandwidth** |                 | **Previous Calculated Net.** |                 |
|---|:------------------:|:---------------:|:----------------------------:|:---------------:|
|   |                    |  **Congested**  |         **Balanced**         | **DeCongested** |
|   |      **Low**       | F. Congested |       F.  Congested       | F. Congested |
|   |    **Average **    | F. Congested |           Balanced           |    Balanced     |
|   |      **High**      | F. Congested |           Balanced           |    Balanced     |
|   |   **Very High**    |   DeCongest.    |          DeCongest.          |    Balanced     |

\pagebreak


### Final CLP FIS

Since this System takes 3 inputs we reduced the Latency granularity to 3 linguistic terms, so to simplify the rule set. 
As such the system-exclusive variables membership function is as presented below:

![Membership Function for CLP-exclusive Input and Output variables](./images/CLP_FIS_MF.png){width=125%}

Note that the high Latency membership has a sharper growth above 0.8 ms, since this is was taken as a stong indicative of a very poor network connection.

The membership function for the linguistic variable "keep" of the CLP output variable was decided to be made extemely fine, this was made so to pull "keep" cases to 0 the most.

\pagebreak
The rule set for the final FIS is slightly more complex. By fixing the Latency variable we can obtain the rule set tables.
For a **Low Latency** value:

|   | **HW Usage** |                     | **True Net. Congestion** |                   |
|---|:------------:|:-------------------:|:------------------------:|:-----------------:|
|   |              |  **F. Congested**   |       **Balanced**       |  **DeCongest.**   |
|   |   **Low**    |      Increase       |         Increase         |     Increase      |
|   | **Balanced** |   Increase   |      Keep/Increase       |       Keep        |
|   |   **High**   |   Increase   |         Decrease         |     Decrease      |
|   | **Critical** |      Decrease       |         Decrease         |     Decrease      |


For a **Average Latency** value:

|   | **HW Usage** |                     | **True Net. Congestion** |                   |
|---|:------------:|:-------------------:|:------------------------:|:-----------------:|
|   |              | **F. Congested** |       **Balanced**       | **DeCongest.** |
|   |   **Low**    |      Increase       |         Increase         |     Increase      |
|   | **Balanced** |      Increase       |      Keep/Increase       |       Keep        |
|   |   **High**   |      Increase       |           Keep           |     Decrease      |
|   | **Critical** |      Decrease       |         Decrease         |     Decrease      |


For a **High Latency** value:

|   | **HW Usage** |                     | **True Net. Congestion** |                   |
|---|:------------:|:-------------------:|:------------------------:|:-----------------:|
|   |              | **F. Congested** |       **Balanced**       |   **DeCongest.**   |
|   |   **Low**    |      Increase       |         Increase         |     Increase      |
|   | **Balanced** |      Increase       |         Increase         |     Increase      |
|   |   **High**   |      Increase       |         Increase         |     Increase      |
|   | **Critical** |      Decrease       |         Decrease         |     Decrease      |

The primary rule is the hardware protection-rule, since avoiding malfunctions/crashing on the Edge device and security of the network are our priorities - If the Hardware has reached criticality the CLP must be decreased.

Another rule obviosuly-prioritized is that when the Hardware Usage is Low or the Network is Fully congested (i.e. very difficult to de-congest) the CLP must be increased(except of course when conflicting with the first rule)

Another clear rule is that when latency is considered too high, priority is given to local data processing (which should translate into a CLP increase), once again exception is given when conflicting with rule number one.


\pagebreak

### Performance Metrics and Final Comments

| Computed CLP Variation | Tabled CLP Variation | Error |
|:----------------------:|:--------------------:|:-----:|
|        0.834000        |         0.85         | 0.016 |
|        0.834000        |         0.85         | 0.016 |
|        0.834000        |         0.80         | 0.034 |
|        0.700957        |         0.73         | 0.029 |
|        0.487149        |         0.50         | 0.013 |
|        0.265243        |         0.12         | 0.145 |
|       -0.316886        |        -0.31         | 0.007 |
|       -0.834000        |        -0.65         | 0.184 |
|       -0.834000        |        -0.82         | 0.014 |
|       -0.834000        |        -0.85         | 0.016 |

Our fuzzy system solution has a MSE (Mean Squared Error) of **0.0058** , which indicates a seemingly strong performance for the provided dataset.
The comparison between the computed Computing Load Percentage variation (CLPv) and the given reference can be observed below.

![Computed CLPv compared to reference CLPv using initial dataset](./images/FuzzyComparison.png){width=85%}

The major problem identified in our FIS solution is related to softer transitions - due to our hard limits related to low or critical Hardware Usage, predicted softer CLP transitions are sharper in our solution having less granularity in for greater CLP Increases and Decreases.


\newpage

## Neural Network

To build our neural network model, we started by generating a random dataset with 10,000 rows. This dataset was processed using the Fuzzy System, which provided the input features for the neural network.

The dataset was then split into three subsets: 70% of the data was allocated for training, 15% for validation, and the remaining 15% for testing. The training set was used to train the neural network through cross-validation, ensuring that the model generalized well across different splits of the data. After completing cross-validation, we used the validation set to fine-tune and identify the optimal hyperparameters for the neural network. Finally, we evaluated the model's performance on the test set to assess its final predictive accuracy.

### Architecture

The final neural network model consisted of six input features and two hidden layers. The first hidden layer contained 18 neurons, while the second hidden layer consisted of 5 neurons. We selected the logistic function as the activation function. The solver used for optimizing the network was the lbfgs algorithm.

This configuration of the neural network was chosen based on the best performance observed during the validation phase, ensuring that the final model achieved a strong balance between accuracy and generalization. We then tested this model on the test set to evaluate its real-world performance.

### Performance Metrics

## Conclusions
