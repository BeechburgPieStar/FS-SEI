# Radio-Frequency-Fingerprinting: Few-Shot-Specific-Emitter-Identification-via-Deep-Metric-Ensemble-Learning

Requirements: keras=2.1.4, tf=1.14.0

Paper: http://arxiv.org/abs/2207.06592 or Y. Wang, G. Gui, Y. Lin, H. -C. Wu, C. Yuen and F. Adachi, "Few-Shot Specific Emitter Identification via Deep Metric Ensemble Learning," in IEEE Internet of Things Journal, vol. 9, no. 24, pp. 24980-24994, 15 Dec.15, 2022, doi: 10.1109/JIOT.2022.3194967.

# Change ADS-B 6000-> ADS-B 4800 (Remove the ICAO code, which can result in inflated accuracy) 
![image](https://github.com/user-attachments/assets/8b4c15fb-2b1c-4440-8587-5fe447850113)

Model weight, dataset and results are updated
1. STC-CVCNN_Train: train feature embedding on auxiliary dataset of 90 classes, and visualization based on test dataset of 10 classes
2. STC-CVCNN_Test: train LR classifer with few-shot training dataset (1-5-10-15-20 shots), and test it on test dataset. Here, this code executes 100 times, because different few-shot training datasets have different performance.
3. STC-CVCNN_SC: feature visualization & get silhouette coefficient

# New result （100 Monte Carlo simulations）
# Different feature embedding with LR classifier
|	C-K	|	FS CVCNN	|	Softmax	|	Siamese	|	Triplet	|	SR2CNN	|	*STC	|	ST	|	SC	|
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

# STC-based feature embedding with diffferent classifiers
|	C-K	|	LR	|	LR-3models	|	LR-5models	|	LR-7models	|	KNN	|	RF	|	SVM
|	---- |	----	|	----	|	----	|	----	|	----	|	----	|	----
|	10-1	|	87.66%	|	89.07%	|	90.12%	|	89.84%	|	21.12%	|	78.21%	|	86.75%
|	10-5	|	93.99%	|	95.22%	|	95.62%	|	95.54%	|	91.86%	|	93.81%	|	93.48%
|	10-10	|	95.28%	|	96.35%	|	96.53%	|	96.48%	|	93.92%	|	94.70%	|	94.33%
|	10-15	|	95.88%	|	96.97%	|	97.05%	|	97.06%	|	94.73%	|	95.19%	|	94.88%
|	10-20	|	96.15%	|	97.23%	|	97.40%	|	97.40%	|	95.25%	|	95.37%	|	95.22%
|	20-1	|	66.11%	|	69.33%	|	69.97%	|	71.17%	|	22.76%	|	58.72%	|	65.24%
|	20-5	|	80.01%	|	82.77%	|	83.56%	|	83.96%	|	73.51%	|	78.29%	|	77.26%
|	20-10	|	84.42%	|	87.61%	|	87.83%	|	88.19%	|	80.67%	|	83.33%	|	81.65%
|	20-15	|	86.53%	|	89.95%	|	90.22%	|	90.52%	|	83.58%	|	85.41%	|	84.45%
|	20-20	|	87.74%	|	91.41%	|	91.34%	|	91.63%	|	85.49%	|	87.13%	|	86.04%
|	30-1	|	55.94%	|	60.40%	|	60.89%	|	61.69%	|	21.74%	|	48.27%	|	54.74%
|	30-5	|	72.46%	|	77.12%	|	77.87%	|	78.35%	|	64.81%	|	70.42%	|	68.95%
|	30-10	|	77.60%	|	82.28%	|	82.93%	|	83.63%	|	72.53%	|	76.01%	|	74.52%
|	30-15	|	80.14%	|	84.85%	|	85.40%	|	85.88%	|	75.36%	|	78.57%	|	77.51%
|	30-20	|	81.37%	|	86.36%	|	86.79%	|	87.48%	|	77.62%	|	80.19%	|	79.26%

# STC-based feature embedding with different classifiers (PyTorch)
* The values in parentheses represent the 95% confidence interval.

|	C-K	 |	 LR	 |	KNN	 |	SVM	 |
|	---- |	---- |	---- |	---- |
|	10-1	|	46.94  (±1.46) 	|	15.80  (±0.76)	|	42.37  (±1.40)	|
|	10-5	|	75.66  (±0.95)	|	53.80  (±0.98)	|	66.67  (±1.18)	|
|	10-10	|	81.89  (±0.77)	|	64.22  (±0.93)	|	74.22  (±1.57)	|
|	10-15	|	85.17  (±0.37)	|	69.35  (±0.56) 	|	79.48  (±0.59)	|
|	10-20	|	86.22  (±0.76)	|	72.05  (±0.72) 	|	80.72  (±1.18)	|
|	20-1	|	37.64  (±0.81)	|	12.81  (±0.68)	|	34.20  (±0.81)	|
|	20-5	|	60.51  (±1.13)	|	44.72  (±0.87)	|	52.14  (±1.31)	|
|	20-10	|	68.65  (±0.57)	|	54.01  (±0.55)	|	61.45  (±0.77) 	|
|	20-15	|	70.68  (±1.23) 	|	57.43  (±1.01) 	|	64.28  (±1.68) 	|
|	20-20	|	73.60  (±1.02) 	|	60.62  (±0.81)	|	67.82  (±1.39)	|
|	30-1	|	30.25  (±0.77) 	|	11.63  (±0.55) 	|	27.79  (±0.71)	|
|	30-5	|	50.86  (±0.80) 	|	37.94  (±0.61)	|	42.60  (±0.93)	|
|	30-10	|	58.22  (±0.66)	|	45.75  (±0.54)	|	50.28  (±0.85)	|
|	30-15	|	62.53  (±0.58)	|	50.17  (±0.42)	|	55.30  (±0.65)	|
|	30-20	|	65.15  (±0.58)	|	52.91  (±0.47)	|	58.88  (±0.67)	|

# The influence of different sets of few-shot training samples (left: 10-way-shot with STC CVCNN and LR; right: 10-way-1-shot with STC CVCNN)
![image](https://user-images.githubusercontent.com/107237593/211043674-bd5b21e6-e5f7-4208-9298-787d90820bf4.png)
![image](https://user-images.githubusercontent.com/107237593/211043693-e96c4216-1498-4445-a9d1-2d4c3a171a1b.png)

# Model weight and Dataset
Link: https://pan.baidu.com/s/13qW5mnfgUHBvWRid2tY2MA 
Passwd：eogv

or 

https://www.dropbox.com/s/ruu3qxfx69k69h0/Dataset.rar?dl=0
