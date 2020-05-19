
The enclosed code corresponds to the paper:

Synthetic oversampling with the majority class: A new perspective on handling extreme imbalance (IEEE ICDM 2018)

and 

Framework for extreme imbalance classification: SWIM—sampling with the majority class (KAIS 2019)

This is a general framework for synthetic oversampling to correct cases of extreme imbalance. Standard methods, such as SMOTE, perform poorly on domains with extreme imbalance. The proposed framework defines a majority focused strategy that reduces bias and improves classifier performance. 

The python scripts:

swimMahademo.py
swimRBFdemo.py

include the Mahalanobis and RBF implementations for the framework. The demo code show the effect of synthesizing new minority samples with each method and with SMOTE. 

The Python code requires the manual installation of:

Sklearn
MatPlotLib
Imblearn
scipy

Please direct questions / comments to 

Colin Bellinger
National Research Council of Canada
Ottawa, Canada
colin.bellinger@gmail.com
https://web.cs.dal.ca/~bellinger/

