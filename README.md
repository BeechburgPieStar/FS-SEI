# Radio-Frequency-Fingerprinting: Few-Shot-Specific-Emitter-Identification-via-Deep-Metric-Ensemble-Learning

Requirements: keras=2.1.4, tf=1.14.0

Paper: http://arxiv.org/abs/2207.06592 or Y. Wang, et al., "Few-Shot Specific Emitter Identification via Deep Metric Ensemble Learning," in IEEE Internet of Things Journal, 2022, doi: 10.1109/JIOT.2022.3194967.

# Change ADS-B 6000-> ADS-B 4800 (Remove the ICAO code) model weight, dataset and results are updated
A brief introduction to this code: (change 6000 to 4800)
1. STC-CVCNN_Train: train feature embedding on auxiliary dataset of 90 classes, and visualization based on test dataset of 10 classes
2. STC-CVCNN_Test: train LR classifer with few-shot training dataset (1-5-10-15-20 shots), and test it on test dataset. Here, this code executes 1000 times, because different few-shot training datasets have different performance.
3. STC-CVCNN_SC: get silhouette coefficient

|	C-K	|	FS CVCNN	|	Softmax	|	Siamese	|	Triplet	|	SR2CNN	|	STC	|	ST	|	SC	|
|	----	|	----	|	----	|	----	|	----	|	----	|	----	|	----	|	----	|
|	10-1	|	10.00%	|	41.30%	|	50.60%	|	75.98%	|	85.04%	|	87.66%	|	73.35%	|	85.37%	|
|	10-5	|	47.80%	|	75.26%	|	77.51%	|	90.18%	|	93.01%	|	93.99%	|	88.66%	|	93.05%	|
|	10-10	|	69.40%	|	87.48%	|	83.34%	|	93.00%	|	94.32%	|	95.28%	|	92.31%	|	94.04%	|
|	10-15	|	77.20%	|	91.03%	|	89.47%	|	93.78%	|	94.81%	|	95.88%	|	93.29%	|	94.45%	|
|	10-20	|	84.30%	|	93.56%	|	91.47%	|	94.18%	|	94.82%	|	96.15%	|	94.34%	|	94.99%	|
|	20-1	|	5.00%	|	35.59%	|	38.02%	|	57.02%	|	64.52%	|	66.11%	|	51.33%	|	61.20%	|
|	20-5	|	38.80%	|	61.73%	|	57.00%	|	71.27%	|	76.05%	|	80.01%	|	71.37%	|	75.50%	|
|	20-10	|	53.60%	|	72.18%	|	66.73%	|	74.96%	|	80.94%	|	84.42%	|	77.46%	|	79.05%	|
|	20-15	|	63.25%	|	76.99%	|	69.78%	|	77.93%	|	82.94%	|	86.53%	|	81.06%	|	81.22%	|
|	20-20	|	72.50%	|	81.18%	|	72.38%	|	79.29%	|	84.63%	|	87.74%	|	83.02%	|	82.59%	|
|	30-1	|	3.33%	|	27.68%	|	28.81%	|	46.18%	|	51.28%	|	55.94%	|	41.81%	|	52.04%	|
|	30-5	|	26.90%	|	53.70%	|	47.77%	|	62.02%	|	68.91%	|	72.46%	|	61.40%	|	66.20%	|
|	30-10	|	47.40%	|	64.22%	|	58.30%	|	67.54%	|	73.33%	|	77.60%	|	68.96%	|	71.22%	|
|	30-15	|	54.57%	|	69.99%	|	62.79%	|	70.12%	|	75.58%	|	80.14%	|	73.68%	|	73.89%	|
|	30-20	|	63.30%	|	74.04%	|	65.70%	|	72.62%	|	77.77%	|	81.37%	|	76.20%	|	75.52%	|


# New result （100 Monte Carlo simulations）
![image](https://user-images.githubusercontent.com/107237593/200116737-5bf14012-04d3-47f8-9d5f-8f345c7ac80a.png)

![image](https://user-images.githubusercontent.com/107237593/200116816-067b8b0a-0913-46bc-b0ed-e2cdaf43d807.png)


# Model weight and Dataset
Link: https://pan.baidu.com/s/13qW5mnfgUHBvWRid2tY2MA 
Passwd：eogv
