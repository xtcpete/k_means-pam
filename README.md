# K_means and pam Clustering
 
This repository contains k-means and k-mediod algorithms for clustering, and a evaluation metrics for contigency matrix and adjusted rand index for evaluating the algorithm.

## Test Algorithms
Both algorithms are tested under stimulation experiment：

60 data points are sampled form the following distributions each with random seed 20

![image](https://user-images.githubusercontent.com/47229849/167925101-85ebab7c-9909-44b4-8b5b-40c95d6ffff4.png)

to then concantenate these points sequentially to form a sample of size  180.

Fit both algorithms with the stimulated data, the resaults are:
k_means method acheive a ARS of 0.783 and the plot


![image](https://user-images.githubusercontent.com/47229849/167925782-e5b10f2c-76de-4699-bf0e-f36217fae4f9.png)


pam method acheive a ARS of 0.743 with parameter p=2 and the plot


![image](https://user-images.githubusercontent.com/47229849/167925970-1c405b18-4133-4914-87f1-33aa8d3d8ba3.png)


All data used for testing are under folder ref_data 

  sim.csv - stimulated data 
  
  sim_labels.csv - stimulated labels 
