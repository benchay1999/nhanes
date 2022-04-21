# nhanes
### nhanes dental clinic revisit probability classification

This machine learning project aims to predict the probability of a paitent needing a dental care within 6 months given demographical data, lab test values, dietary data, examination data, and questionarre data.

Run the following command to install required packages:

```
pip install -r requirements.txt
```

The data for this project is NHANES dataset with years spanning 2009-2018.

# Details
XGBoost (`nhanes_xgboost.ipynb') worked best for the project, yielding approximately 0.99 in accuracy.

To reproduce, you need the NHANES dataset (unzip `nhanes_0918.zip`)
and `preprocessing_xgboost.ipynb` with `nhanes_xgboost.ipynb'.

Other files are for DNN and logistic regression training on the dataset, which worked poorly in this highly-sparse data setting where categorical features dominate numerical features.
