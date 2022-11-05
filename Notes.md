# Notes
## GMM
1. Use Gaussian mixture Model on SemSeg Dataset
2. Test different K Values for the GMM Module 
3. Assumption : there might be different clusters in the Data
4. These Clusters might offer a hint at how different Distributions model the two Data Sessions
5. Common Clusters might be responsible for modelling the Task 
6. To check: If number of Distributions make a difference on how the main cluster is divided.
7. To check : if there are caommon clusters hwo similar are they -> Similarity Methods (SSIM,Cosine Distance, etc)
8. Might be interestion to test GMMs on two diffenrt Modalities of Data. The problem would be to Normalize the two different modes so they fall in the same range.



# LSTMs
## RNNS and LSTM Gating
1. LSTMs typically used as Cells.
2. Tanh activation to address the Exploding/Vanishing Gradients Problem 
3. Mostly used as Sequence Modelers to generate predictions 
4. To be researched: Different types of RNNs (Many to Many, one to many, one to one,many to one-> might be relevant in my case)
5. Two possibilities of Combination Parallel Data Input model 
6. Concatenated Model with a CNN/LSTM Structures 
7. To be checked : How Losses behave over such architectures
8. Backprop over LSTM to be revised 
9. Different Modalities have to be normalized accordingly 
10. 
