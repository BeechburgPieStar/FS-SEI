# Few-Shot-Specific-Emitter-Identification-via-Deep-Metric-Ensemble-Learning
The code is about Few-Shot-Specific-Emitter-Identification-via-Deep-Metric-Ensemble-Learning under review

Requirements: keras=2.2.4, tf=1.14.0

A brief introduction to this code:
1. STC-CVCNN_Train: train feature embedding on auxiliary dataset of 90 classes, and visualization based on test dataset of 10 classes
2. STC-CVCNN_Test: train LR classifer with few-shot training dataset (1-5-10-15-20 shots), and test it on test dataset. Here, this code executes 1000 times, because different few-shot training datasets have different performance.
3. STC-CVCNN_SC: get silhouette coefficient

Model weight is given in Model/
 
Dataset will be uploaded soon.
