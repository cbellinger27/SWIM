
####################################################################################
# <one line to give the program's name and a brief idea of what it does.>
# Copyright (C) <2020>  <Colin Bellinger>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# Please direct any questions / comments to myself, Colin Bellinger, at
# colin.bellinger@gmail.com. For additional software and publications
# please see https://web.cs.dal.ca/~bellinger and researchgate
# https://www.researchgate.net/profile/Colin_Bellinger
#
# Relevant publications include: 
#
# 1. Synthetic oversampling with the majority class: A new perspective on handling extreme imbalance
#
# S Sharma, C Bellinger, B Krawczyk, O Zaiane, N Japkowicz
# 2018 IEEE International Conference on Data Mining (ICDM),
#
# 2. Framework for extreme imbalance classification: SWIM—sampling with the majority class. 
# 
#   Bellinger, C., Sharma, S., Japkowicz, N., & Zaïane, O. R. 
#   2019 Knowledge and Information Systems
#
####################################################################################



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

