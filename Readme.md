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
The code uses a method called simulate() to demonstrate the data processing pipeline. It is based on spectral decomposition and normalization. The code trains a multi-class SVM. Eg.:

	>>> dh = data_handling()
	>>> dh.simulate()
	
The current version only uses 3 features per channel, for a total of 18 features. A full-scale analysis of optimal features and classifier models is beyond the scope of this example, but is the next step in the analysis.

Use 'dh.scores' to get the cross-validation scores. Eg.:

	>>> dh.scores
	array([0.82069756, 0.83945468, 0.83773429, 0.82047812, 0.82577517])
	
### How do I play with the parameters
Currently all modifiable parameters are coded inside various methods.
This will be changed in the future where they will reside in a params.yaml file.

- For fourier decomposition: the "\_\_init\_\_()" method has relevant variables
- For classification: see the "classify()" method

### Who do I talk to?

For Temple University dataset: Gagan Narula

For bugs or issues with the code: Ali Zaidi
