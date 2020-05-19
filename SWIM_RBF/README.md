Code applies to:

Bellinger, C., Sharma, S., Japkowicz, N. et al. Framework for extreme imbalance classification: SWIM—sampling with the majority class. Knowl Inf Syst (2019) doi:10.1007/s10115-019-01380-z

This method is used to generate synthetic training samples for classification domains with extreme class imbalance. 


Example use of the Swim_RBF code:

import Swim_RBF as Swim
erbo = Swim.extremeRBO(epsilon=None)
X_res, y_res = erbo.extremeRBOSample(x_trainImb, y_trainImb, numSamples)


The <epsilon> parameter controls the spread of the sythetic samples around the minority seeds. Epsilon values closer to zero result in a flatter, wider basis function. Generally, epsilon=1.5 is a reasonable values. However, this can be optimized via cross-validation.

The extremeRBOSample function returns a new training dataset and label set with the original samples plus <numSamples> synthetic minority samples. 

Please see the paper for more details.