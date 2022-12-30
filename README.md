# Radio-Frequency-Fingerprinting: Few-Shot-Specific-Emitter-Identification-via-Deep-Metric-Ensemble-Learning

Requirements: keras=2.1.4, tf=1.14.0

Paper: http://arxiv.org/abs/2207.06592 or Y. Wang, et al., "Few-Shot Specific Emitter Identification via Deep Metric Ensemble Learning," in IEEE Internet of Things Journal, 2022, doi: 10.1109/JIOT.2022.3194967.

# Change ADS-B 6000-> ADS-B 4800 (Remove the ICAO code) model weight, dataset and results are updated
A brief introduction to this code: (change 6000 to 4800)
1. STC-CVCNN_Train: train feature embedding on auxiliary dataset of 90 classes, and visualization based on test dataset of 10 classes
2. STC-CVCNN_Test: train LR classifer with few-shot training dataset (1-5-10-15-20 shots), and test it on test dataset. Here, this code executes 1000 times, because different few-shot training datasets have different performance.
3. STC-CVCNN_SC: get silhouette coefficient

# New result （100 Monte Carlo simulations）
|  1  |  2  |  3  |  4  |  5  |
|----|----|----|----|----|
| 1   |   2 |   3 | 4   | 5   |
![image](https://user-images.githubusercontent.com/107237593/200116737-5bf14012-04d3-47f8-9d5f-8f345c7ac80a.png)

![image](https://user-images.githubusercontent.com/107237593/200116816-067b8b0a-0913-46bc-b0ed-e2cdaf43d807.png)


# Model weight and Dataset
Link: https://pan.baidu.com/s/13qW5mnfgUHBvWRid2tY2MA 
Passwd：eogv
