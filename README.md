# NHANES - Dental
### NHANES : need for dental care probability classification

<p align="center">
  <b>***This project is best compatible with Google Colab.***</b>
</p>
  



This machine learning project aims to predict the probability of a subject needing dental treatment within 6 months given demographical data, lab test values, dietary data, examination data, and questionarre data.

Run the following command to install required packages:

```
pip install -r requirements.txt
```

The data for this project is the NHANES dataset with years spanning 2009-2018.

# Details
XGBoost (`nhanes_xgboost.ipynb`) worked best for the project, yielding approximately 0.99 in accuracy.

## Model Deployment
FCN : `nhanes.ipynb`

XGBoost : `nhanes_xgboost.ipynb`

To reproduce, you need the NHANES dataset (unzip `nhanes_data/nhanes_0918.zip`)
and `preprocessing_xgboost.ipynb` together with `nhanes_xgboost.ipynb`.

## Data
You can also download my preprocessed data in `nhanes_data/`

`nhanes_data/data_label_encoded.dat` : label encoded version of preprocessed data (no NaNs)

`nhanes_data/data_onehot_encoded_final.dat` : one-hot encoded version of preprocessed data

`nhanes_data/label.dat` : labels (whether a patient should have dental treatment within 6 months)


`nhanes_data/final_df.pkl` : pandas dataframe for preprocessed nhanes data.

`nhanes_data/final_df_xgboost_with_ohx.pkl` : pandas dataframe for xgboost (with the feature 'OHARNF') - contains NaNs

`nhanes_data/final_df_xgboost_without_ohx.pkl` : pandas dataframe for xgboost (without the feature 'OHARNF') - contains NaNs

`nhanes_data/final_df_xgboost_with_ohx_nan_preprocessed.pkl` : pandas dataframe for xgboost (with the feature 'OHARNF') - no NaNs * same with `final_df.pkl`

`nhanes_data/final_df_xgboost_without_ohx_nan_preprocessed.pkl` : pandas dataframe for xgboost (without the feature 'OHARNF') -no NaNs * same with `final_df.pkl` with no 'OHARNF'




# Miscellaneous

Other files are for training FCNs on the dataset, which worked poorly in this highly-sparse data setting where categorical features(one-hot encoded) dominate numerical features.


