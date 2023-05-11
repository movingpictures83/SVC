# SVC
# Language: Python
# Input: TXT
# Output: CSV
# Tested with: PluMA 1.1, Python 3.6
# Dependency: sklearn==0.23.1, pandas==1.1.5

Plugin plugin to perform C-Support Vector Classification (Cortes and Vapnik, 1995)

Input is a TXT file of tab-delimited keyword-value pairs:
training: Training dataset
traininggroups: Classification of training samples
testing: Test dataset

The output will be a CSV file of the classifications of the test dataset

