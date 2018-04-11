# Hybrid_Model_Tox21_Abstract
In recent times, toxicological classification of chemical compounds is considered to be a grand challenge for pharma-ceutical and environment regulators. Advancement in machine learning techniques enabled efficient toxicity predic-tion pipelines. Random forests (RF), support vector machines (SVM) and deep neural networks (DNN) are often ap-plied to model the toxic effects of chemical compounds. However, complexity-accuracy tradeoff still needs to be ac-counted in order to improve the efficiency and commercial deployment of these methods.  In this study, we implement a hybrid framework consists of a shallow neural network and a decision classifier for toxicity prediction of chemicals that interrupt nuclear receptor (NR) and stress response (SR) signaling pathways. A model based on proposed hybrid framework is trained on Tox21 data using 2D chemical descriptors that are less multifarious in nature and easy to calcu-late.  Our method achieved the highest accuracy of 0.847 AUC (area under the curve) using a shallow neural network with only one hidden layer consisted of 10 neurons. Furthermore, our hybrid model enabled us to elucidate the inter-pretation of most important descriptors responsible for NR and SR toxicity. 


# Descreption of the files in each folder of Tox21 Tasks
There are total of 12 Tox21 Tasks. There is a separate project folder for each task.

--Data
--------------------------
Train, Test and CV data files. These .xlsx files contain 2D features for training, cross-validation and test. Moreover, the feature_name.xlsx containes the names of the features selected for individual task.

--Jupyter Notebook file for training the individual model with optimized parameters
--------------------------
Each folder has a jupyter notebook file with a name task_code.ipynb. Here the task refers to spesific task name. please note that all the necessary libraries must be installed. This part creates 4 individual models for a spesific task and then ensamble it. This also creates a list of selected features and select the reduced number of features from test set for the final testing.

--Jupyter Notebook file checking the reported results in the paper
--------------------------
The notebook file with a name Saved_Model_Checking takes the selected features from the test set and the 4 already trainined models with all parameters to reproduce the same results as reported in the paper.


