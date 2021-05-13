# av1-datascience-unifesp
BernouilliNP algorithm using Holdout and K-folds (cross validation) model selection. The result contain Accuracy, Confusion Matrix, Precision, Recall, and f1-score.

## Requirements
The minimum requirements for running this projects is:
```sh
pip and python - version 3.x
```

## Install

```sh
pip install -r requirements.txt
```

## Execute

This project execute the Bernoulli - Naive Bayes algorithm using determinate model selection. The command line to execute this forecast is:

```sh
python BernoulliNB.py [dataset] [y_label] [test_size] [model_selection]
```

[dataset] - The binary dataset information in csv format. Those dataset download in UCI repo.

[y_label] - y_label is the dependent variable i.e. the variable that needs to be estimated and predicted.

[test_size] - If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.

[model_selection] - Model selection is the task of selecting a statistical model from a set of candidate models, given data. In the simplest cases, a pre-existing set of data is considered. In his case, you can select the holdout or k_fold (cross validation)   

Example:

```sh
python BernoulliNB.py creditcard.csv Class 0.2 k_folds
```

## Result

The result is generate the two files. The first file is the confusion matrix called in png format. The second file is the log file that contain Accuracy, Confusion Matrix, Precision, Recall, and f1-score.