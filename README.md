# Keyword-Extraction-for-Pathology-Reports-with-BERT
This is Keyword extraction for pathology report based on BERT.
The original repo of BERT (Devlin, et al.) is [BERT]

## For NOT programmer users.
Please visit: [KEP]
The URL is web-based keyword extraction for pathology report.
![screensh](./img/website.jpg)


## Environments
Requirements:
	python 3.6
	pytorch 1.2.0
	pytorch-pretrained-bert 0.6.2

## Usage
Example:
	python Keyword_Extraction_BERT.py --data sample.csv

	arguments:
	--data DATA      Data
	--maxlen MAXLEN  Max Length
	--bs BS          Batch Size
	--lr LR          Learning Rate
	--epoch EPOCH    Epochs
	
[KEP]: http://cdal.korea.ac.kr/KEP/kep.php
[BERT]: https://github.com/google-research/bert