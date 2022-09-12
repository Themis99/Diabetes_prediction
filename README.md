# Diabetes_prediction
In this project we use the diabetes dataset in which we try to predict disease progression one year after baseline. This is a regression problem in which we're trying to predict a value.

## Dataset
The dataset is provided by the Sklearn library used to develop the prediction algorithms. The dataset consists of 11 columns where 10 are the predictive variables and the eleventh is the target which is essentially a quantitative measure of disease progression one year after baseline. More information about the dataset can be found [here](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html).

## Data preparation
The first 300 samples are used training and the rest for testing. In addition, all values (including the target values) were normalized to the interval [0,1].

## Training and testing
Five algorithms were developed and compared in terms of their performance. These were:
1) Support Vector Machines
2) Desicion Trees
3) Random Forest
4) Grandient Boost 
5) ADA Boost
6) XG Boost

All algorithms were evaluated using the same evaluation metrics: 

1) Mean Absolute Error (MAE)
2) Mean Squared Error (MSE)
3) Root Mean Squared Error (RMSE)
4) R^2 Score (R^2)

The evaluation method was performed by cross-validation with 10 folds. Each algorithm was fine-tuned to determine the optimal hyperparameters. Randomized Search algorithm was used for the search for the best hyperparameters. In contrast to the Grid Search algorithm, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions. Ten trials for every algorithm were performed.

## Results
The table below compares the performance of the different algorithms:

| Algorithm     | MAE           | MSE           | RMSE          | R^2           |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Support Vector Machines |  0.15   |    0.03           |    0.18           |      0.36         |
| Desicion Trees  |     0.16        |      0.04         |       0.20        |       0.24        |
| Random Forest   |    0.15         |        0.03       |         0.19      |       0.32        |
| Grandient Boost   |    0.15         |        0.03       |         0.18      |       0.34        |
| ADA Boost  |       0.14      |     0.03         |       0.18        |        0.37       |
| XG Boost  |        0.14     |       0.03        |      0.18         |       0.40        |

Of all the algorithms, XG boost has the best performance.
