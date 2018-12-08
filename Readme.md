# Readme
Author: Ali Zaidi<p>
Version 0.1<p>
Date: 06.12.2018<p>
(c) All rights reserved.
### What is this repo for?
This code provides an early proof of concept for feature extraction based on spectral normalization from EEG data for classification of seizeures based on the Temple University EEG Corpus.

This is code has been developed as part of the ICU Cockpit project at UZH.

### How do I get set up?
1. Clone repo (checkout the develop branch for latest version of the code)
2. Updade the path in 'data\_handling.py:load\_data()' with TU corpus data
3. Run the code

### How do I use the code?
The code uses a method called simulate() to demonstrate the data processing pipeline. It is based on spectral decomposition and normalization. To train a binary SVM classifier, call the simulate method with the index of the data_label. Eg:

	>>> dh = data_handling()
	>>> dh.simulate(15,6)
	
will output something like:

	Loading dataset...
	summary of valid labels is below:
	Format: [Label name, label index, Label count]
	['null', 0, 879]
	['bckg', 6, 6864]
	['gnsz', 9, 67]
	['cpsz', 11, 184]
	['tcsz', 15, 6]
	Using label 15 as target
	Using label 0 as baseline
	Normalizing data...
	Generating training dataset...
	Training classifier with 5x5 CV...
	Matthews corr coeff = 0.95
	balanced test accuracy = 0.96
	Cross-validation accuracy: 0.98 (+/- 0.02)
	

### How do I play with the parameters
Currently all modifiable parameters are coded inside various methods.
This will be changed in the future where they will reside in a params.yaml file.

- For fourier decomposition: the "\_\_init\_\_()" method has relevant variables
- For classification: see the "classify()" method

### Who do I talk to?

For Temple University dataset: Gagan Narula

For bugs or issues with the code: Ali Zaidi
