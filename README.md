# Hybrid_Model_Tox2D_Abstract

## Under review in Ameical Chemical Society (ACS-Omega)

### Submission date: 29-Apr-2018


In recent times, toxicological classification of chemical compounds is considered to
be a grand challenge for pharmaceutical and environment regulators. Advancement in
machine learning techniques has enabled efficient toxicity prediction pipelines. Random
forests, support vector machines and deep neural networks are often used in building
prediction models for toxic effects of chemical compounds. However, complexityaccuracy
trade-off of a model still needs to be accounted in order to improve its efficiency
and to make it suitable for commercial deployment. Moreover, these machine learning
approaches are used as “black box”; which means no insights are available from them
about the problem or the solution structures. In this study, using a shallow neural
network and a decision tree classifier, we propose a hybrid framework to build a simple
machine learning model that can be explained in terms of feature relevance and that
can help elucidate the final solution. We then construct a prediction model based on
the proposed hybrid framework and train it on nuclear receptor (NR), stress response
(SR) and ames mutagenicity (AM) data sets. 
The NR and SR data sets are from Tox21 data repository while the AM data set is by 
Hansen et. al.. For all three data sets, we
calculate only 2D chemical descriptors, which are less multifarious in nature and easy to
calculate. However, our model still achieved better ensembled average accuracy of 0.836
AUC-ROC (area under the receiver operating characteristic curve), 0.862, and 0.878 for
NR, SR, and AM respectively while the best known existing methods achieved 0.826,
0.858, and 0.860 respectively. For this, our model uses a shallow neural network with
only one hidden layer consisted of only 10 neurons. Its average training time for each
task is only ~1 minute on a single CPU while methods using deep neural networks take
about 10 minutes on NVidia Tesla K40 GPU. Furthermore, in our hybrid approach, the
neural network is trained with significantly fewer features (in the range of hundreds),
which makes the model simpler and less compute intensive, but it still maintains the
high accuracy level. Our method also enables us to elucidate the interpretation of the
descriptors that are the most responsible for NR, SR and AM toxicity types. These
descriptors showed high classification strength to discriminate toxic compounds and
could be used as initial indicators for detecting NR, SR and AM toxicity types.

We also verify the our results using 2D features for four additional toxicity tasks such as
IGC50, LD50, LC50DM and LC50.

-----------------------------------------------------------------------------------------

# System setup
Pip install Tensorflow
Pip install Keras
Pip install sklearn
Pip install PIL
Pip install pandas
Pip install numpy
Pip install scipy
Pip install openpyxl
Pip install xlsxwriter
Pip install h5py
Pip install matplotlib
 

-----------------------------------------------------------------------------------------

# Descreption of the necessary files in each folder to run the models on tests sets
There are total of 17 Toxicity Tasks. Each folder contain trained models and test set. The python code 
to verify the results on tests sets is also given. 
-----------------------------------------------------------------------------------------

# Descreption of the necessary files in each folder to run the models on tests sets
There are total of 17 Toxicity Tasks. Each folder contain trained models and test set. The python code 
to verify the results on tests sets is also given. 
-----------------------------------------------------------------------------------------

# Descreption of the necessary files in each folder to run the models on tests sets
There are total of 17 Toxicity Tasks. Each folder contain trained models and test set. The python code 
to verify the results on tests sets is also given. 



--Jupyter Notebook file checking the reported results in the paper
--------------------------
The notebook file with a name Saved_Model_Checking takes the selected features from the test set and the 4 already trainined models with all parameters to reproduce the same results as reported in the paper.

--Hybrid Model code
--------------------------
This file contains the generic code for Hybrid Model.
