# LitJetNet
Pytorch-Lightning Normalizing Flows for [jetnet](https://github.com/jet-net/JetNet)
However the data used for training is downloaded and saved as .csv to be independent on changes in the jetnet repository. 
The code is in LitNF/
To reproduce the results either use the notebook, in which the weights are loaded to reproduce the results, or retrain the model with main.py (takes around 11h on an NVIDIA P100 GPU)
