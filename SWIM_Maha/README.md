Code applies to:

Synthetic oversampling with the majority class: A new perspective on handling extreme imbalance
S Sharma, C Bellinger, B Krawczyk, O Zaiane, N Japkowicz
2018 IEEE International Conference on Data Mining (ICDM), 447-456

This method is used to generate synthetic training samples for classification domains with extreme class imbalance. 


Example use of the Swim_Maha code:

import Swim_Maha as Swim
sw = Swim.SwimMaha(sd=0.5)
X_res, y_res = sw.mahaSampling(Training_Data, Labels, numSamples)

The <sd> parameter controls the spread of the sythetic samples around the minority seeds. A larger seed will give more spread. Generally, sd=0.5 is a reasonable values. However, this can be optimized via cross-validation.

The mahSampling function returns a new training dataset and label set with the original samples plus <numSamples> synthetic minority samples. 

In the case that the majority traning set forms a signular matrix, the SWIM algorithm degrades to random oversampling with Gaussian jitter. 

In the case that the majority training set has linearly dependent columns, the sub-space of independent columns is used for synthetic oversampling. 

Please see the paper for more details.