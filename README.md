# Human-Activity-Recognize-HARdataset-  
Prepare data from accelerometer, use diffrent convolutional neural network to classify human activity.  
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones - link to data repositories  
1) I Model  
3x....py - files with three models to create a structural recognize  
First: Classify Static/Dynamic  
Next: Classification standing, laying, sitting or walk, walk up, walk down.  
2) II-III Model  
accelerometer.py - there are two 1D CNN models  
3) IV Model  
CNN2D.py - 2D CNN using wavelet transform to prepare data  
4) Another files:  
acc-gyro.py - plot sample (one time sequence) of signal  
cwt_vs_STFT.py - simple show digital defrences between CWT and STFT  
filter.py - simple filter project, to get coeficients and use them on the mobile app  
Har_histo.py - data histograms  
Wavelet.py - show few wavelet kernels  
