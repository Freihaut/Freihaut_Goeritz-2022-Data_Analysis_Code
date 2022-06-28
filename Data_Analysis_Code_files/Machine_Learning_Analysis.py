'''
Code to run the machine learning analysis with the keyboard feature dataset
A random seed was explicitly set wherever necessary to create reproducible results
The code was run in Pycharm with scientific mode turned on. The #%% symbol separates the code into cells, which
can be run separately from another (similar to a jupyter notebook)
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'''


# package imports

# basic python packages
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random

# sklearn for all machine learning procedures
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GroupShuffleSplit
from sklearn.inspection import permutation_importance
from sklearn.base import RegressorMixin, TransformerMixin, BaseEstimator

# import statsmodel for the linear mixed model analysis
import statsmodels.api as smf

# import plotline to use the ggplot2 functionality (because creating one plot in matplotlib was annoying and we
# already have a similar ready-to-go solution for ggplot from a past analysis project
import plotnine as p9

#%%

# import the keyboard features dataset
keyboard_data = pd.read_csv("Keyboard_Features.csv")


#%%

# Transform selected string columns into label columns
keyboard_data["ID"] = LabelEncoder().fit_transform(keyboard_data["ID"])
keyboard_data["app_version"] = LabelEncoder().fit_transform(keyboard_data["app_version"])
# dummy code the occupation with 3 categories (0 = Working, 1 = Student, 3 = Other)
keyboard_data = pd.get_dummies(keyboard_data, columns=["occupation"], drop_first=True)

#%%

# list all predictors
predictors = [
       'tot_ev', 'corr_ev', 'type_time',
       'dwelltime_mean', 'dwelltime_median', 'dwelltime_sd', 'latency_mean',
       'latency_median', 'latency_sd', 'down_down_mean', 'down_down_median',
       'down_down_sd', 'up_up_mean', 'up_up_median', 'up_up_sd',
       'no_key_pushed', 'num_trials']
# list all covariates
covariates = ["age", "sex", "occupation_1", "occupation_2", "app_version", "num_data_colls", "ID"]
# list all targets
targets = ["arousal", "valence"]

# dictionary to "convert" the variable names into prettier names in result plots and tables
pretty_variable_names = {
    'tot_ev': "Tot. Keyboard Events", 'corr_ev': "Tot. Correction Events",
    'type_time': "Typing Time",
    'dwelltime_mean': "Dwelltime (Mean)", 'dwelltime_median': "Dwelltime (Median)", 'dwelltime_sd': "Dwelltime (SD)",
    'latency_mean': "Latency (Mean)", 'latency_median': "Latency (Median)", 'latency_sd': "Latency (SD)",
    'down_down_mean': "Down-Down (Mean)", 'down_down_median': "Down-Down (Median)", 'down_down_sd': "Down-Down (SD)",
    'up_up_mean': "Up-Up (Mean)", 'up_up_median': "Up-Up (Median)", 'up_up_sd': "Up-Up (SD)",
    'no_key_pushed': "No Key Pushed %",
    'num_trials': "Password Trials",
    "age": "Age", "sex": "Sex", "occupation_1": "Occupation: Student", "occupation_2": "Occupation: Other",
    "app_version": "App Language", "num_data_colls": "Tot. Data Collections", "ID": "ID",
    "valence": "Valence", "arousal": "Arousal"
}

#%%

###############################################
# Step 1: Setup the Dataset Creation Pipeline #
###############################################

# -----------------------------------------------------------------
# Helper Functions to split the dataset into training and test data
# -----------------------------------------------------------------

# helper function to split the dataset by the Timestamp
# The dataset is split in a way that the first 80% of the completed measurement trials of each participant represent
# the training dataset and the last 20% of the completed measures represent the testing dataset. This resembles
# an individualized prediction approach
def split_by_timestamp(dataset):

    # first, split the entire dataset in data subsets for each participant
    split_dataframes = [y for x, y in dataset.groupby("ID", as_index=True)]

    # setup a list that holds the training and test datasets for each participant
    train_dfs = []
    test_dfs = []
    # iterate the dataframe of each participant
    # measurements
    for df in split_dataframes:
        # sort it by the timestamp column to order the trials chronologically
        df = df.sort_values("time").reset_index()
        # add an order column
        df["order"] = range(len(df))
        # get the first 80% of the dataset
        df_80 = df.head(int(len(df) * .8))
        # get the remaining 20% of the dataset
        df_20 = df.iloc[max(df_80.index) + 1:]
        # add the participant train and test datasets to the list of training and test dfs
        train_dfs.append(df_80)
        test_dfs.append(df_20)

    # combine the training and test datasets of each participant to create the final training and test dataset with data
    # from all participants
    train_df, test_df = pd.concat(train_dfs, ignore_index=True), pd.concat(test_dfs, ignore_index=True)

    return train_df, test_df


# helper function to split the dataset by participants
# The dataset is split in a ways that all individual data collections of a participant are either only used for
# training or only for testing. Again, we split the dataset in a way that the training data is made up of about
# 80% of the sample and the test dataset is made up of about 20% of the sample
# set a random state for each time the function is called in order to create reproducible train-test splits
def split_by_participant(dataset, rand_state):

    # use the scikitlearn splitter to do the job
    gss = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=rand_state)
    # split the dataset by the grouping variable ID
    splits = gss.split(dataset, groups=dataset["ID"])
    # get the train and test split ids
    train_ids, test_ids = next(splits)
    # create the training and test dataset
    train_df = dataset.iloc[train_ids]
    test_df = dataset.iloc[test_ids]

    return train_df, test_df


#%%

# ---------------------------------
# Data Preprocessing Helper Classes
# ----------------------------------

# A simple debugger class to get information about the data inside the pipeline (can be customized)
class DebugPipeline(BaseEstimator, TransformerMixin):

    # do nothing in the fit
    def fit(self, X):
        return self

    # log info about the dataset in the transform step
    def transform(self, X):
        # return information about the dataset
        print(X.shape)
        print(X.columns)
        # return the original data (just logging, no transformation)
        return X


# a class to be used inside the pipeline that visualizes the distribution of the input feature variables
# to visualize what effect a data transformation step has on the features
class VisualizeFeatureDistributions(BaseEstimator, TransformerMixin):

    def __init__(self, features=None, name="", create_plot=False):
        self.features = features
        self.name = name
        self.create_plot = create_plot

    # static helper function to visualize the feature distributions
    @staticmethod
    def _multi_kde_plot(data, name):

        data_to_plot = data.copy()
        # rename the columns to pretty names
        data_to_plot = data_to_plot.rename(columns=pretty_variable_names)

        # set a style
        sns.set_style("white")

        # create a plot with an appropriate number of columns and rows (depending of the number of the columns to plot)
        num_cols = data_to_plot.shape[1]

        fig, axes = plt.subplots(nrows=int(np.sqrt(num_cols)) + 1, ncols=int(np.sqrt(num_cols)) + 1,
                                 figsize=(30, 30), sharex=False, sharey=False)
        axes = axes.ravel()  # array to 1D
        cols = list(data_to_plot.columns)  # create a list of dataframe columns to use

        for col, ax in zip(cols, axes):
            sns.set(font_scale=2.25)
            sns.kdeplot(data=data_to_plot, x=col, shade=True, ax=ax)
            ax.set(title=col, xlabel=None, xticklabels=[], yticklabels=[])

        # delete the empty subplots
        ax_to_del = [i for i in range(num_cols, len(axes))]

        for i in ax_to_del:
            fig.delaxes(axes[i])

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(name)
        # save the figure
        plt.savefig(name + '_kde_plot.png')
        # plot the figure
        # plt.show()
        plt.close('all')

        return

    # visualize the data inside the fit function, then just returns the dataset without any alterations
    def fit(self, X):

        if self.create_plot:
            self._multi_kde_plot(X.loc[:, self.features], self.name)

        return self

    # the transform function has no purpose and just returns the dataset
    def transform(self, X):
        return X


# a class to be used inside the pipeline that visualizes the distribution of the input feature variables
# to visualize what effect a data transformation step has on the features
class VisualizeFeatureCorrelations(BaseEstimator, TransformerMixin):

    def __init__(self, features=None, name="", create_plot=False):
        self.features = features
        self.name = name
        self.create_plot = create_plot

    # static helper function to visualize the feature distributions
    @staticmethod
    def _correlation_heatmap(data_to_plot, fig_size, font_scale, name, add_text=True):

        # rename the columns to pretty names
        data_to_plot = data_to_plot.rename(columns=pretty_variable_names)

        # calculate the correlation matrix of the data
        corr = data_to_plot.corr()

        # set a figure size
        plt.figure(figsize=fig_size)
        # set a scale size to scale all texts
        sns.set(font_scale=font_scale)
        sns.set_style("white")
        # create a mask to only plot one diagonal of the heatmap
        mask = np.tril(np.ones(corr.corr().shape)).astype(bool)
        # get the correlation matrix
        corr = corr.where(mask)
        # create the heatmap
        ax = sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            linewidth=0.8,
            cmap=sns.diverging_palette(220, 20, n=200),
            fmt='.2f',
            cbar_kws={"shrink": .45, "label": 'correlation coeff.'},
            annot=add_text,
            square=True,
        )
        # set the axis labels
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right',
        )

        # if specified, add the correlation coefficient as text into the tiles
        if add_text:
            # change the text format to remove the 0 before the decimals and to replace 1 with an empty string
            for t in ax.texts: t.set_text(t.get_text().replace('0.', '.').replace('1.00', ''))

        # set a tight layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(name)
        # save the figure
        plt.savefig(name + '.png')
        # plot the heatmap
        # plt.show()
        plt.close('all')

        return

    # visualize the data inside the fit function, don`t do anything else
    def fit(self, X):

        if self.create_plot:
            self._correlation_heatmap(data_to_plot=X.loc[:, self.features], fig_size=(48, 38), font_scale=3.8,
                                      name=self.name, add_text=True)

        return self

    # the transform function has no purpose and just returns the dataset
    def transform(self, X):
        return X


# A custom transformer for the scikit learn pipeline to remove highly correlated features of a given feature set based
# on a specified correlation threshold
class RemoveCorrelatedFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None, cor_thresh=.90):

        self.cols = cols
        self.cor_thresh = cor_thresh
        self.sel_columns = []

    def fit(self, X, y=None):

        # Identify highly correlated features
        # If two variables have a high correlation, the function looks at the mean absolute correlation of each
        # variable and removes the variable with the largest mean absolute correlation
        # This procedure is copyed from the findCorrelation function of the caret package in R
        # https://rdrr.io/cran/caret/man/findCorrelation.html

        # get the correlation matrix of the dataset
        corrmat = X.loc[:, self.cols].corr().abs()
        # get the average correlation of each column
        average_corr = corrmat.abs().mean(axis=1)
        # set lower triangle and diagonal of correlation matrix to NA
        corrmat = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(bool))
        # where a pairwise correlation is greater than the cutoff value, check whether mean abs.corr of a or b
        # is greater and cut it
        to_delete = list()
        for col in range(0, len(corrmat.columns)):
            for row in range(0, len(corrmat)):
                if corrmat.iloc[row, col] > self.cor_thresh:
                    # print(f"Compare {corrmat.index.values[row]} with {corrmat.columns.values[col]}")
                    if average_corr.iloc[row] > average_corr.iloc[col]:
                        to_delete.append(corrmat.index.values[row])
                    else:
                        to_delete.append(corrmat.columns.values[col])
        self.sel_columns = list(set(to_delete))
        return self

    def transform(self, X):
        # return the dataframe without the columns that were deleted
        return X.drop(self.sel_columns, axis=1)


# Apply the standardscaler by group (or by the entire column if no group is specified) for a set of specified columns
# from: https://stackoverflow.com/questions/68356000/how-to-standardize-scikit-learn-by-group
class GroupByScaler(BaseEstimator, TransformerMixin):
    def __init__(self, by=None, sel_cols=None):
        self.scalers = dict()
        self.by = by
        self.cols = sel_cols

    def fit(self, X, y=None):
        # make a copy of X to silence setcopy warning
        X = X.copy()
        # if no group was specified, standardize the columns by the entire column (sample)
        if not self.by:
            x_sub = X.loc[:, self.cols]
            self.scalers["no_group"] = StandardScaler().fit(x_sub)
        # if a group was specified, standardize the selected columns by group
        else:
            for val in X.loc[:, self.by].unique():
                mask = X.loc[:, self.by] == val
                X_sub = X.loc[mask, self.cols]
                self.scalers[val] = StandardScaler().fit(X_sub)
        return self

    def transform(self, X, y=None):
        # make a copy of X to silence setcopy warning
        X = X.copy()
        # if no group was specified, standardize the columns by the entire column (sample)
        if not self.by:
            # transform the specified columns with the standardscaler
            X.loc[:, self.cols] = self.scalers["no_group"].transform(X.loc[:, self.cols])
        # if a group was specified, standardize the selected columns by group
        else:
            for val in X.loc[:, self.by].unique():
                mask = X.loc[:, self.by] == val
                X.loc[mask, self.cols] = self.scalers[val].transform(X.loc[mask, self.cols])
        return X


# custom class to handle potential outliers in the dataset
# there are two three removal procedures:
#  "use_all" = do not include anyone
#  "selected_thresh" = exclude participants based on a predefined type time and dwell_time_median threshold
# "iqr" = exclude participants based on the IQR method for selected features
# WARNING: This class removes rows from the dataset (X). It can not be used inside the sk learn pipeline
# the class is highly specified for the current data analysis (code does not generalize well and is not that great)
# Note that specifying a threshold and choosing features that are used for removal was partly done using information
# from the entire dataset, which presents data leakage to some degree!
# Depending on a generalized or individualized prediction, the IQR method could also be applied for the entire
# dataset or per participant. Because of the "small" size of individual datasets, we chose to always use the entire
# training sample for specifying the iqr params. This procedure could be biased towards excluding data from one
# participant more likely than from another if there is a strong difference in typing behavior across participants
class RemoveOutliers(BaseEstimator, TransformerMixin):

    def __init__(self, method="use_all", sel_cols=None):
        self.iqr = None
        self.method = method
        self.sel_cols = sel_cols

    def fit(self, X):
        # if the method is the interquantile range method
        if self.method == "iqr":
            # only use the columns that specified for the outlier removal
            outlier_features = X.loc[:, self.sel_cols]

            # calculate the critical values to identify outliers based on the interquantile range method
            q1 = outlier_features.quantile(0.25)
            q3 = outlier_features.quantile(0.75)
            iqr = q3 - q1
            # save the calculated values in a dictionary
            self.iqr = {"q1": q1, "q3": q3, "iqr": iqr}

        return self

    # perform the outlier removal
    def transform(self, X):
        # if all datasets are included, simply return the entire dataframe
        if self.method == "use_all":
            X = X

        # if the outliers are removed based on specified thresholds ("expert outlier evaluation")
        elif self.method == "selected_thresh":
            # simply use loc (this is not a good generalized solution that can take specified filters
            X = X.loc[(X["type_time"] < 120) &
                      (X["dwelltime_median"] > 10)]

        # if the specified method is the iqr outlier removal method
        elif self.method == "iqr":
            # exclude the data that fall out of the 1.5 * IQR range
            X = X[~((X[self.sel_cols] < (self.iqr["q1"] - 1.5 * self.iqr["iqr"])) |
                    (X[self.sel_cols] > (self.iqr["q3"] + 1.5 * self.iqr["iqr"]))).any(axis=1)]

        return X


#%%

# ------------------
# Dataset Creation
# ------------------
# Before the Machine Learning Algorithm is trained with the data (and its performance is tested on the test dataset),
# the dataset is preprocessed. This preprocessing includes multiple steps and for some steps, there are multiple
# preprocessing options. Following the idea of a multiverse analysis (Steegen et al., 2016; for the paper see:
# https://doi.org/10.1177/1745691616658637),we create multiple datasets using multiple preprocessing routines. In such
# a way, we also get multiple prediction results (which is a little bit similar to evaluationg the prediction
# performance using repeated-cross-validation such as n-fold-cv). In general, the procedure helps to get a sense for
# the robustness (variability) of the prediction results and therefore about the potential validity of the results.

# The preprocessing includes:
# - Splitting the entire dataset into a training and test dataset
# - Removing Outliers
# - Standardization
# - Removal of highly correlated features

# In order to prevent data leakage in the machine learning analysis, all preprocessing steps are done using the
# training dataset and then applied to the testing dataset. We use the scikit-learn preprocessing pipeline for
# handling the preprocessing


# helper function that handles the preprocessing procedure
# the input is the original raw dataset, and a bool that indicates if the dataset is processed for the individualized
# prediction approach or the generalized prediction approach
def create_processed_datasets(dataset, prediction_approach, include_visualization=False):

    # Create different Training-Test dataset pairs based on the specified preprocessing options
    dataset_pairs = {}

    # Specify the outlier removal options that will be looped
    outlier_options = ["use_all", "selected_thresh", "iqr"]

    # loop all option combinations (outlier options * datatransformation option)
    for rand_index, opt in enumerate(outlier_options):

        print(f"Creating the train-test dataset pair for outlier removal option: {opt}")

        # first, split the data into the training dataset and the test dataset
        if prediction_approach == "individualized":
            # from a performance point-of-view, this should be done outside the loop, because the timestamp split
            # is always the same
            training_data, test_data = split_by_timestamp(dataset)
        elif prediction_approach == "generalized":
            # pass the function a random index to create reproducible splits
            training_data, test_data = split_by_participant(dataset, rand_index)

        # print some info about the datasets
        print(f"Shape of the training dataset: {training_data.shape}")
        print(f"Shape of the test dataset: {test_data.shape}")

        # next, set up a data transformation pipeline that handles the data transformation in the training and test
        # dataset with the specified preprocessing options
        # The data transformation pipeline includes a visualisation of the keyboard input features to see how the
        # preprocessing changes them
        dataset_transformation_pipeline = Pipeline([
            # First, visualize the unprocessed features
            ('Unprocessed Visualization', VisualizeFeatureDistributions(features=predictors,
                                                                        name=opt + "_step0",
                                                                        create_plot=include_visualization)),
            # Second, apply the outlier removal procedure
            # We used the same features for the IQR method as we used when we specfied a threshold
            # (this procedure is up to debate)
            ("Outlier_Handling", RemoveOutliers(method=opt, sel_cols=["type_time", "dwelltime_median"])),
            # Third, visualize the distribution again
            ('Outlier Visualization', VisualizeFeatureDistributions(features=predictors,
                                                                    name=opt + "_step1",
                                                                    create_plot=include_visualization)),
            # Fourth, standardize the features (standardize by participants if the prediction is individualized or
            # by the entire sample if the prediction is generalized
            ('Standardize Features', GroupByScaler(by=None if prediction_approach == "generalized" else "ID",
                                                   sel_cols=predictors)),
            # Fifth, visualize the "finale feature distribution
            ('Standardization Visualization', VisualizeFeatureDistributions(features=predictors,
                                                                            name=opt + "_step2",
                                                                            create_plot=include_visualization)),
            # Sixth, visualize the correlations of all features
            ('Visualize Feature Correlations', VisualizeFeatureCorrelations(features=predictors,
                                                                            name=opt,
                                                                            create_plot=include_visualization)),
            # Seventh, remove multicollinear features
            ('remove_multicoll', RemoveCorrelatedFeatures(cols=predictors, cor_thresh=0.9)),
        ])

        # Now fit the data transformation pipeline on the training dataset and transform the train dataset as well as
        # the test dataset
        transformed_train_df = dataset_transformation_pipeline.fit_transform(training_data)
        transformed_test_df = dataset_transformation_pipeline.transform(test_data)

        # print some info about the processed datasets
        print(f"Shape of the processed training dataset: {transformed_train_df.shape}")
        print(f"Shape of the processed test dataset: {transformed_test_df.shape}")
        print("\n")

        # save the train-test pair in the train-test-pair dictionary
        dataset_pairs[opt] = {"train": transformed_train_df, "test": transformed_test_df}

    return dataset_pairs


#%%

# Create the Datasets for predictions on unknown participants (general prediction approach)
# ******************************************************************************************
generalized_prediction_datasets = create_processed_datasets(keyboard_data, prediction_approach="generalized",
                                                            include_visualization=False)

#%%

# Create the Datasets for predictions on known participants (individualized prediction approach)
# ******************************************************************************************

individualized_prediction_datasets = create_processed_datasets(keyboard_data, prediction_approach="individualized",
                                                               include_visualization=False)


#%%

###################################################
# Step 2: Setup for the Machine Learning Analysis #
###################################################

# ------------------------------------------------------------------------------------------------------------------
# helper functions and classes to run the machine learning analysis with a given dataset and a given target variable
# -------------------------------------------------------------------------------------------------------------------

# custom class to make the statsmodel linear mixed model work similar to any other model provided by scikit-learn
# inspired by:
# https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible
# this is not a generalized solution and only fits the need of the present analysis/dataset setup
class LinearMixedModel(BaseEstimator, RegressorMixin):
    # takes as input a group identifier string to get the grouping variable
    # and bool that indicates that the model includes an intercept (or not)
    def __init__(self, group_identifier):

        self.group_identifier = group_identifier
        self.model_ = None
        self.results_ = None

    # fit the linear mixed model to the dataset (no customization is possible here)
    def fit(self, X, y):

        # add an intercept
        X = smf.add_constant(X, has_constant="add")

        # fits a linear mixed model that uses the ID column as the random intercept
        # no other random effects are specified (should be made more flexible)
        self.model_ = smf.MixedLM(endog=y, exog=X.drop([self.group_identifier], axis=1), groups=X[self.group_identifier])
        self.results_ = self.model_.fit()

        return self

    # make a prediction with the mixed linear model on the test dataset
    # additional input is a bool that indicates if the model should make a prediction about a known participant
    # or an unknown participant
    def predict(self, X, pred_option):

        # add an intercept
        X = smf.add_constant(X, has_constant="add")

        # if the prediction is made for data from an unknown participants, the prediction can only use the fixed
        # effects of the linear mixed model (default prediction setting of statsmodel)
        if pred_option == "generalized":
            # make a prediction on the test dataset
            preds = self.results_.predict(exog=X.drop([self.group_identifier], axis=1))
        # if the prediction is made for data from a known participant, the estimated random intercept (and potentially
        # the random slope) of that participant can be used
        # Statsmodel does not offer a straightforward way to include the random effects in the prediction and requires
        # a "hack", see
        # https://stats.stackexchange.com/questions/467543/including-random-effects-in-prediction-with-linear-mixed-model
        elif pred_option == "individualized":
            # first, create a pseudo linear mixed model for the test dataset to be able to extract relevant
            # information from the model in the next step
            pseudo_dataset = X.copy()
            pseudo_dataset["y"] = 0
            pseudo_mod = smf.MixedLM(endog=pseudo_dataset["y"], exog=pseudo_dataset.drop([self.group_identifier], axis=1), groups=pseudo_dataset[self.group_identifier])
            # second, get the predicted random effects of the linear mixed model fitted on the train data
            re = self.results_.random_effects
            # Multiply each predicted random effect by the random effects design matrix for one group
            # here, it basically gets the difference between the mean intercept of the model and each group intercept
            rex = [np.dot(pseudo_mod.exog_re_li[j], re[k]) for (j, k) in enumerate(pseudo_mod.group_labels)]
            # Add the fixed and random terms to get the overall prediction
            # it basically adjusts for the difference between the grand mean intercept and each group mean intercept
            # if random slopes had been added, the group specific slope would also be taken into account
            rex = np.concatenate(rex)
            preds = self.results_.predict(exog=X.drop([self.group_identifier], axis=1)) + rex

        return preds

    # additional function to get the model results in order to extract the model parameters (get a summary of the model)
    def get_model_coeffs(self):
        return self.results_


# helper function for the mixed model machine learning analysis
# The linear mixed model can also be used to make predictions about the emotional state of known and unknown
# participants and the model results are "easier" to interpret than in a nonlinear ml model
# here we use a straightforward random intercept model and no additional random effects
# https://stats.stackexchange.com/questions/250277/are-mixed-models-useful-as-predictive-models
def ml_mixed_model(training_data, test_data, preds, target, pred_option):

    # split the data into the predictors and targets
    x_train, x_test = training_data.loc[:, preds], test_data.loc[:, preds]
    y_train, y_test = training_data[target], test_data[target]

    print(f"Shape of the train-test datasets: {x_train.shape, x_test.shape, y_train.shape, y_test.shape}")

    # We do not need to train any hyperparameters in the linear mixed model analysis (although it would be possible to
    # iterate some settings such as ML-estimation versus REML-estimation

    # Fit and Evaluate the model
    # --------------------------

    # initialize the mixed model
    mixed_model = LinearMixedModel(group_identifier="ID")

    # fit the model with the training data
    mixed_model.fit(x_train, y_train)

    # make predictions on the test dataset
    # choose if only the fixed effect estimates are used for prediction (predicting unknown participants), or if the
    # the fixed and random effect estimates are used for prediction (predicting known participants)
    predictions = mixed_model.predict(x_test, pred_option=pred_option)

    # get the mean absolute error and R²-score as the regression evaluation metrics
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"R²-Prediction Score: {r2}")
    print(f"Mean Absolute Prediction Error: {mae}")

    # Get the Model Parameters from the model
    # ---------------------------------------------------------------------------

    model_parameters = mixed_model.get_model_coeffs()
    # print a summary of the model parameters as provided by the statsmodule package
    print(model_parameters.summary())
    # save the relevant model parameters in a dataframe
    model_param_df = pd.read_html(model_parameters.summary().as_html(), header=0, index_col=0)[1]

    print(f"Analysis are done")
    # return the results of the regression analysis
    results = {"model_params": model_param_df, "scores": {"r2": r2, "mae": mae}}

    return results


# helper function for the machine learning regression analysis (for the nonlinear algorithms)
# to give more control about the results, it would also be possible to return the trained model, save it, and be
# able to get results in a later step without having to wait for the training process over and over again
def ml_nonlinear_regression(training_data, test_data, preds, target, ml_regressor, hyperparameters, pred_option):

    # split the data into the predictors and targets
    x_train, x_test = training_data.loc[:, preds], test_data.loc[:, preds]
    y_train, y_test = training_data[target], test_data[target]

    print(f"Shape of the train-test datasets: {x_train.shape, x_test.shape, y_train.shape, y_test.shape}")

    # The machine learning procedure has three parts:

    # In the first part, model hyperparameters are selected using Randomized Hyperparameter Cross Validation Search
    # see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

    # In the second part, the tuned model is fit on the training dataset and the predicton performance is tested using
    # the test dataset

    # In the third part, the machine learning model is interpreted -> the importance of the input features for the
    # model prediction are evaluated

    # Hyperparameter selection:
    # -------------------------

    # setup Cross Validation
    # 5-fold cross validation is used.
    # if the prediction option is to predict data of unknown participants, use a similar cross validation scheme and
    # split the data by groups
    if pred_option == "generalized":
        # get the group k fold splitter
        cv_generator = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        # create the splits
        cv_splits = [(train, test) for train, test in cv_generator.split(x_train, y_train, groups=x_train["ID"])]
    # else if the prediction option is to predict data of known participants,
    elif pred_option == "individualized":
        # participants are stratified across the groups in order to balance
        # out the number of participants during the cross validation process
        # setup the CV generator
        cv_generator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # create the cross validation splits, which are stratified by the participant -> "ID" column of the dataset
        # The X-Input is a placeholder, because it is not required for splitting:
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold.split
        cv_splits = [(train, test) for train, test in cv_generator.split(
            X=np.zeros(len(training_data)),
            y=training_data["ID"])]

    # run the RandomizedSearchCv with the selected random forest regressor, the random forest hyperparameter grid, and
    # the cross validation splits, select the r2_score as the score for selecting the best hyperparameter pairs,
    # use multiple cores (n_jobs), refit the entire model after having selected the best parameters and specify
    # a random state
    print("Running the Randomized Search Cross Validation")
    ml_model = RandomizedSearchCV(ml_regressor, hyperparameters, cv=cv_splits, scoring=make_scorer(r2_score),
                                  refit=True, n_jobs=8, random_state=42)

    # Evaluating the tuned model
    # --------------------------

    # fit the training dataset to the tuned model
    ml_model.fit(x_train, y_train)

    # get the selected hyperparameters
    selected_hyperparameters = ml_model.best_params_
    print(f"Selected Hyperparameters: {selected_hyperparameters}")
    # It is also possible to get additional infos from the RandomizedSearchCV procedure
    # print(testing.cv_results_)
    # print(testing.best_score_)

    # make predictions on the test set
    predictions = ml_model.predict(x_test)

    # get the mean absolute error and R²-score as the regression evaluation metrics
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"R²-Prediction Score: {r2}")
    print(f"Mean Absolute Prediction Error: {mae}")

    # Get the feature importance scores to "interpret" the machine learning model
    # ---------------------------------------------------------------------------
    print(f"Running the feature importance permutation")
    # use feature importance permutation
    # see: https://scikit-learn.org/stable/modules/permutation_importance.html
    feature_importance_scores = permutation_importance(ml_model, x_test, y_test,
                                                       scoring=make_scorer(r2_score),
                                                       n_jobs=8,
                                                       n_repeats=30,
                                                       random_state=42)

    print(f"Analysis are done")
    # return the results of the regression analysis
    results = {"hyperparams": selected_hyperparameters, "scores": {"r2": r2, "mae": mae},
               "feat_importance": feature_importance_scores}

    return results


# helper function to plot the mixed model coefficients
def plot_mixed_model_coefficients(data, plot_title):

    # remove two variables that will not be plotted
    df_to_plot = data.drop(["const", "Group Var"])
    # set the axis to be a variable
    df_to_plot = df_to_plot.rename_axis("variable").reset_index()
    # rename the variables to create "prettier" plots
    df_to_plot["variable"] = df_to_plot['variable'].replace(pretty_variable_names)

    coeff_plot = (
            p9.ggplot(df_to_plot, p9.aes(x='variable', y='Coef.'))
            # plot a vline at
            + p9.geom_hline(yintercept=0, colour="black", linetype='dashed', size=1.5)
            + p9.geom_pointrange(p9.aes(ymax="[0.025", ymin="0.975]"), position=p9.position_dodge(width=0.8), size=1.75)
            + p9.geom_vline(xintercept=[i + 0.5 for i in range(len(df_to_plot["variable"].unique()))], color="grey",
                            alpha=0.8, linetype='dotted')
            + p9.coord_flip()
            + p9.theme_classic()
            + p9.theme(text=p9.element_text(size=20), figure_size=(26, 16))
            + p9.ggtitle(plot_title)
    )

    # save the plot
    # p9.ggsave(plot=coeff_plot, filename=plot_title)

    # show the plot
    print(coeff_plot)

    return True


# helper function to plot the feature importance scores of the fitted ml models
def plot_feature_importance_scores(importance_results, col_names, plot_title):

    # Get the Importance results scores and convert them to a dictionary with each column representing the feature
    # importance scores for one predictor in the machine learning model
    importance_df = pd.DataFrame(importance_results["importances"]).T
    importance_df.columns = [pretty_variable_names[name] if name in pretty_variable_names else name for name in col_names]
    # sort the dataframe by the mean importance score
    importance_df = importance_df.reindex(importance_df.mean().sort_values().index, axis=1)

    # create a barplot figure to visualize the importance scores
    plt.figure(figsize=(22, 16))
    sns.set(font_scale=2.4)
    sns.set_style("whitegrid")
    g = sns.barplot(data=importance_df, orient="h", palette="deep", errwidth=6).set(title=plot_title)
    # g.set(xlim=(0, importance_df.to_numpy().max()))
    plt.tight_layout()

    # save the figure
    # plt.savefig(plot_title + "_feat_import.png")

    # show the figure
    plt.show()

    return


#%%

# Setup the Machine Learning Algorithms that will be used for the analysis
# ------------------------------------------------------------------------

ml_algorithms = {
    "Linear_Mixed_Model": {},
    "NearestNeighbor": {
        'model': KNeighborsRegressor(),
        'hyperparams': {
            "weights": ['uniform', 'distance'],
            "n_neighbors": [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35]
        }
    },
    "Random_Forest": {
        "model": RandomForestRegressor(random_state=42),
        "hyperparams": {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_features': ['auto', 'sqrt', "log2"],
            'max_depth': [60, 70, 80, 90, 100, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }}
}

#%%

##############################################
# Step 3: Run the Machine Learning Analysis  #
##############################################

# The machine learning analysis will include multiple loops:
#   - Loop all Target Variables (valence & arousal)
#   - Loop all Preprocessed Datasets
#   - Loop all Machine Learning Methods


# helper function to run all analysis for a specified dataset, target, ml algorithm and prediction option
def run_ml_analysis(dataset, all_predictors, target, algorithm, prediction_option, visualize_results=False,
                    plot_title=""):

    # get the train dataset
    train_data, test_data = dataset["train"], dataset["test"]
    # get all preds of the dataset
    preds = [pred for pred in train_data.columns if pred in all_predictors]

    # if the algorithm is the linear mixed model, then run the mixed model analysis pipeline
    if algorithm == "Linear_Mixed_Model":
        ml_result = ml_mixed_model(training_data=train_data, test_data=test_data, preds=preds,
                                   target=target, pred_option=prediction_option)
        if visualize_results:
            plot_mixed_model_coefficients(ml_result["model_params"], plot_title)
    # else, run the nonlinear machine learning pipeline
    else:
        ml_result = ml_nonlinear_regression(training_data=train_data, test_data=test_data, preds=preds,
                                            target=target, ml_regressor=ml_algorithms[algorithm]["model"],
                                            hyperparameters=ml_algorithms[algorithm]["hyperparams"],
                                            pred_option=prediction_option)
        if visualize_results:
            plot_feature_importance_scores(ml_result["feat_importance"], preds, plot_title)

    return ml_result


#%%

# --------------------------------------------------------------------------------------
# Playground to run the machine learning analysis for a single dataset & target variable
# --------------------------------------------------------------------------------------

# First setup a playground to be able to run the machine learning analyis for one case

# Randomly draw if the ML analysis are done for generalized or individualized prediction
playground_pred_option = random.choice(["generalized", "individualized"])
# randomly draw a dataset from the randomly drawn prediction option
if playground_pred_option == "generalized":
    playground_dset_name, playground_dset = random.choice(list(generalized_prediction_datasets.items()))
else:
    playground_dset_name, playground_dset = random.choice(list(individualized_prediction_datasets.items()))
# randomly draw a target variable to predict
playground_target = random.choice(["valence", "arousal"])
# randomly draw a machine learning model
playground_ml_model = random.choice(list(ml_algorithms.items()))[0]

# run the analysis for the randomly selected options
# alternatively you can also run the analysis for all ml algorithms by looping over the algorithms
print(f"Running the playground analysis with all predictors for {playground_pred_option} prediction with dataset:"
      f" {playground_dset_name}, target: {playground_target} and ML model: {playground_ml_model}")

play_ground_ml_results = {"pred_op": playground_pred_option,
                          "dset": playground_dset_name,
                          "target": playground_target,
                          "algorithm": playground_ml_model,
                          "results": run_ml_analysis(dataset=playground_dset, all_predictors=covariates,
                                                     target=playground_target, algorithm=playground_ml_model,
                                                     prediction_option=playground_pred_option,
                                                     visualize_results=False,
                                                     plot_title=playground_pred_option + "_" + playground_dset_name +
                                                                "_" + playground_target + "_" + playground_ml_model)}


#%%

# helper function to run the ml analysis loop
def run_ml_analysis_loop(dset, prediction_option, visualize_single_results):

    ml_results = {}

    # loop both target variables
    for target in targets:
        ml_results[target] = {}
        # loop all datasets
        for dataset in dset:
            ml_results[target][dataset] = {}
            # loop all ML algorithms
            for algorithm in ml_algorithms:
                ml_results[target][dataset][algorithm] = {}
                # run the machine learning analysis and save the results
                print(f"Running the {prediction_option} ML prediction for Target: {target}, dataset: {dataset} and "
                      f"algorithm: {algorithm}")
                print(f"Getting the baseline results")
                baseline_results = run_ml_analysis(
                    dataset=dset[dataset],
                    all_predictors=covariates,
                    target=target, algorithm=algorithm,
                    prediction_option=prediction_option,
                    visualize_results=visualize_single_results,
                    plot_title="Baseline_" + prediction_option + "_" + dataset + "_" + target + "_" + algorithm)
                print("\n")
                print(f"Getting the full model results")
                full_model_results = run_ml_analysis(
                    dataset=dset[dataset],
                    all_predictors=predictors + covariates,
                    target=target, algorithm=algorithm,
                    prediction_option=prediction_option,
                    visualize_results=visualize_single_results,
                    plot_title="Full_Mod" + prediction_option + "_" + dataset + "_" + target + "_" + algorithm
                )

                ml_results[target][dataset][algorithm]["baseline_results"] = baseline_results
                ml_results[target][dataset][algorithm]["full_model_results"] = full_model_results
                ml_results[target][dataset][algorithm]["dset_shapes"] = \
                    {'train': dset[dataset]['train'].shape,
                     'test': dset[dataset]['test'].shape}
                ml_results[target][dataset][algorithm]["selected_features"] = \
                    [pred for pred in dset[dataset]['train'].columns if pred in predictors + covariates]

                print("\n")
            print("\n")

    return ml_results


#%%

# ------------------------------------------------------------------
# Run the machine learning analysis to make a generalized prediction
# ------------------------------------------------------------------
generalized_prediction_results = run_ml_analysis_loop(dset=generalized_prediction_datasets,
                                                      prediction_option="generalized",
                                                      visualize_single_results=False)

# save the results
with open("generalized_prediction_results.p", 'wb') as fp:
    pickle.dump(generalized_prediction_results, fp, protocol=pickle.HIGHEST_PROTOCOL)

#%%

# ----------------------------------------------------------------------
# Run the machine learning analysis to make an individualized prediction
# ----------------------------------------------------------------------

individualized_prediction_results = run_ml_analysis_loop(dset=individualized_prediction_datasets,
                                                         prediction_option="individualized",
                                                         visualize_single_results=False)

# save the results
with open("individualized_prediction_results.p", 'wb') as fp:
    pickle.dump(individualized_prediction_results, fp, protocol=pickle.HIGHEST_PROTOCOL)


#%%

# if the results are already saved, "import" them here
# with open('generalized_prediction_results.p', 'rb') as handle:
#     generalized_prediction_results = pickle.load(handle)
#
# with open('individualized_prediction_results.p', 'rb') as handle:
#     individualized_prediction_results = pickle.load(handle)


#%%

#####################
# Result Processing #
#####################

# simple helper to print the results
def print_results(results):
    for targ in results:
        for dset in results[targ]:
            for alg in results[targ][dset]:
                print(f"Results for {targ}, {dset}, {alg}")
                print(f"Baseline: {results[targ][dset][alg]['baseline_results']['scores']}")
                print(f"Full Model: {results[targ][dset][alg]['full_model_results']['scores']}")


#%%

print("Generalized Prediction Results")
print_results(generalized_prediction_results)

#%%

print("Individualized Prediction Results")
print_results(individualized_prediction_results)


#%%

# Helper Functions to Create Plots of the Results (result visualization)
# ----------------------------------------------------------------------

# helper function to plot the feature permutation scores in one figure
def merged_importance_plot_fig(results, target, plot_title):

    # first, create a merged dataframe that contains all feature permutation scores for each feature, dataset and
    # algorithm
    df_list = []

    # loop all datasets of the results dictionary of the target variable
    for dset in results[target]:
        # loop all ml algorithms
        for alg in results[target][dset]:
            # exclude the linear mixed model results, because we did not calculate feature importance scores for it
            if alg != "Linear_Mixed_Model":
                # extract the importance scores of the target, dataset, algorithm combination
                importance_scores = results[target][dset][alg]["full_model_results"]["feat_importance"]
                # create a dataframe from the importance scores
                importance_df = pd.DataFrame(importance_scores["importances"]).T
                # set the column labels of the dataframe
                importance_df.columns = results[target][dset][alg]["selected_features"]
                # add a column that holds the info about the dataset
                importance_df["dset"] = dset
                # add a column that holds the info about the algorithm
                importance_df["alg"] = alg
                # for plotting, the dataset needs to be converted from the wide format into the long format
                importance_df = pd.melt(importance_df, id_vars=["dset", "alg"],
                                        value_vars=results[target][dset][alg]["selected_features"])
                # finally add the importance scores dataframe to the list of all dataframes
                df_list.append(importance_df)

    # now concat all importance score dataframes together
    df_to_plot = pd.concat(df_list)
    # to make the plot a little bit "prettier", sort the feature columns by the average importance score (across all
    # datasets and algorithms)
    # create a list of the mean importance score feature order
    mean_scores_per_feature = list(df_to_plot.groupby("variable")["value"].mean().sort_values().index)
    # change the datatype of the variable to category and sort the new category column by the calculated order
    df_to_plot["variable"] = df_to_plot["variable"].astype("category")
    df_to_plot["variable"] = df_to_plot["variable"].cat.set_categories(mean_scores_per_feature)
    df_to_plot.sort_values(["variable"])

    # rename some variables to make the plot prettier
    df_to_plot["dset"] = df_to_plot['dset'].replace({'iqr': 'IQR removal', 'selected_thresh': 'Cutoff removal',
                                             'use_all': 'No removal'})
    df_to_plot["alg"] = df_to_plot['alg'].replace({'NearestNeighbor': 'K-Nearest Neighbor',
                                                   'Random_Forest': 'Random Forest'})
    # rename the variables to create "prettier" plots
    df_to_plot["variable"] = df_to_plot['variable'].replace(pretty_variable_names)
    # rename the columns
    df_to_plot = df_to_plot.rename(columns={"dset": "Outlier Removal Procedure"})

    # start to plot the finalized dataframe
    plt.figure(figsize=(22, 16))
    sns.set(font_scale=2.5)
    sns.set_style("white")
    sns.despine()
    # we use a seaborn catplot to plot barcharts of the feature importance scores per dataset and algorithm
    g = sns.catplot(x="value", y="variable", hue="Outlier Removal Procedure", col="alg", data=df_to_plot,
                    kind="bar", height=20, aspect=0.75)
    # do some custom changes to the plot
    (g.set_axis_labels("Pemutation Importance Score", "Features")
     .set_titles("{col_name}"))
    # set a tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(plot_title)
    # save the figure
    plt.savefig("Perm_scores_" + plot_title + '.png')
    # shot the figure
    plt.show()
    plt.close('all')


# helper function to plot the model estimates of the trained linear mixed model across the datasets
# alternative: https://zhiyzuo.github.io/Python-Plot-Regression-Coefficient/
def plot_mixed_model_coefficients(results, target, plot_title):

    # first, create a merged dataframe that contains all feature permutation scores for each feature, dataset and
    # algorithm
    df_list = []

    # loop all datasets of the results dictionary of the target variable
    for dset in results[target]:
        # extract the feature coefficients (that are already stored in a dataframe)
        coeffs_df = results[target][dset]["Linear_Mixed_Model"]["full_model_results"]["model_params"]
        # add the dataset info to the dataframe
        coeffs_df["dset"] = dset
        # add it to the list of all dataframes
        df_list.append(coeffs_df)

    # add all dataframes together
    df_to_plot = pd.concat(df_list)
    # remove two variables that will not be plotted
    df_to_plot = df_to_plot.drop(["const", "Group Var"])
    # set the axis to be a variable
    df_to_plot = df_to_plot.rename_axis("variable").reset_index()

    # do some renaming and changing to make the plot prettier

    # sort the variables column to have an order that first shows the features and then the control variables
    order = covariates + predictors
    # change the datatype of the variable to category and sort the new category column by the calculated order
    df_to_plot["variable"] = df_to_plot["variable"].astype("category")
    df_to_plot["variable"] = df_to_plot["variable"].cat.set_categories(order)
    df_to_plot.sort_values(["variable"])

    # rename some variables to make the plot prettier
    df_to_plot["dset"] = df_to_plot['dset'].replace({'iqr': 'IQR removal', 'selected_thresh': 'Cutoff removal',
                                             'use_all': 'No removal'})
    # rename the variables to create "prettier" plots
    df_to_plot["variable"] = df_to_plot['variable'].replace(pretty_variable_names)
    # rename the columns
    df_to_plot = df_to_plot.rename(columns={"dset": "Outlier Removal Procedure"})

    coefficient_plot = (
            p9.ggplot(df_to_plot, p9.aes(x='variable', y='Coef.', color='Outlier Removal Procedure',
                                         group='Outlier Removal Procedure'))
            # plot a vline at
            + p9.geom_hline(yintercept=0, colour="black", linetype='dashed', size=1.5)
            + p9.geom_pointrange(p9.aes(ymax="[0.025", ymin="0.975]"), position=p9.position_dodge(width=0.8), size=1.75)
            + p9.geom_vline(xintercept=[i + 0.5 for i in range(len(df_to_plot["variable"].unique()))], color="grey",
                            alpha=0.8, linetype='dotted')
            + p9.coord_flip()
            + p9.theme_classic()
            + p9.theme(text=p9.element_text(size=21), figure_size=(30, 16), legend_position=(0.8, 0.8))
            + p9.scale_color_manual(values=['#1b9e77', '#d95f02', '#7570b3'])
            + p9.labs(x="Features", y="Coefficient Estimates with 95% CI")
            + p9.ggtitle(plot_title)
    )

    # save the plot
    p9.ggsave(plot=coefficient_plot, filename="Coefficient-Plots_" + plot_title, limitsize=False)

    # show the plot
    print(coefficient_plot)

    return


#%%

# Create Plots for the Generalized Prediction
# -------------------------------------------

# create the merged permutation plots for both target variables
merged_importance_plot_fig(generalized_prediction_results, "arousal", "Generalized_Pred_Arousal")

#%%

merged_importance_plot_fig(generalized_prediction_results, "valence", "Generalized_Pred_Valence")

#%%

plot_mixed_model_coefficients(generalized_prediction_results, "arousal", "Generalized_Pred_Arousal")

#%%

plot_mixed_model_coefficients(generalized_prediction_results, "valence", "Generalized_Pred_Valence")

#%%

# Create Plots for the Individualized Prediction
# ----------------------------------------------

# create the merged permutation plots for both target variables
merged_importance_plot_fig(individualized_prediction_results, "arousal", "Individualized_Pred_Arousal")

#%%

merged_importance_plot_fig(individualized_prediction_results, "valence", "Individualized_Pred_Valence")

#%%

plot_mixed_model_coefficients(individualized_prediction_results, "arousal", "Individualized_Pred_Arousal")

#%%

plot_mixed_model_coefficients(individualized_prediction_results, "valence", "Individualized_Pred_Valence")


#%%

# Create a Result Table of the analysis
# -------------------------------------


# create a new dataframe with the following structure

#                               Baseline      Full Mod
#                            -------------  -----------
#           Train/Test Shape |  R² | MAE |  | R² | MAE |
# -------------------------------------------------------
#  Dset1 |
#  Dset2 |
#  Dset3 |


# helper function to extract the relevant results from the results data set
def ml_results_table(result_file):

    result_table = {}

    # loop the results file to extract the relevant info that needs to be put together for the table creation
    for target in result_file:
        for dset in result_file[target]:
            for alg in result_file[target][dset]:
                # create the nested rows of the result table
                result_table[(target, alg, dset)] = {}

                # now populate the table
                # ----------------------

                # add the dataset shapes
                result_table[(target, alg, dset)][("Num. Samples", "Train Data")] = \
                    result_file[target][dset][alg]['dset_shapes']["train"][0]
                result_table[(target, alg, dset)][("Num. Samples", "Test Data")] = \
                    result_file[target][dset][alg]['dset_shapes']["test"][0]

                # add the baseline results
                result_table[(target, alg, dset)][("Baseline Model", "Num. Preds.")] = len(covariates)
                result_table[(target, alg, dset)][("Baseline Model", "R²-score")] = \
                    np.round(result_file[target][dset][alg]["baseline_results"]["scores"]["r2"], 4)
                result_table[(target, alg, dset)][("Baseline Model", "MAE")] = \
                    np.round(result_file[target][dset][alg]["baseline_results"]["scores"]["mae"], 4)

                # add the full model results
                result_table[(target, alg, dset)][("Full Model", "Num. Preds.")] = \
                    len(result_file[target][dset][alg]["selected_features"])
                result_table[(target, alg, dset)][("Full Model", "R²-score")] = np.round(
                    result_file[target][dset][alg]["full_model_results"]["scores"]["r2"], 4)
                result_table[(target, alg, dset)][("Full Model", "MAE")] = np.round(
                    result_file[target][dset][alg]["full_model_results"]["scores"]["mae"], 4)

    # convert the result dictionary into a dataframe
    result_df = pd.DataFrame.from_dict(result_table, orient='index')

    # do some renaming to create a prettier table
    result_df = result_df.rename(index={"arousal": "Arousal", "valence": "Valence",
                                        'iqr': 'IQR removal', 'selected_thresh': 'Cutoff removal',
                                        'use_all': 'No removal', 'NearestNeighbor': 'K-Nearest Neighbor',
                                        'Random_Forest': 'Random Forest', "Linear_Mixed_Model": "Linear Mixed Model"})

    return result_df


#%%

# create and save the results table (as excel files, because excel makes it easy to further process the table)
generalized_results_table = ml_results_table(generalized_prediction_results)
generalized_results_table.to_excel("generalized_results_table.xlsx")

#%%

individualized_results_table = ml_results_table(individualized_prediction_results)
individualized_results_table.to_excel("individualized_results_table.xlsx")


#%%

#####################################################################################################################
# Test the mixed model prediction routine and compare the prediction performance of a "badly" and "correctly" specified
# model (a model that predicts data of known population, but only predicts using the fixed effects versus a model
# that also uses the random effects)
######################################################################################################################

# a function that runs the analysis routine with a sample dataset with a hierarchical data structure as provided by
# statsmodels package

# This sample shows that the linear mixed model works well for an individualized prediction and it shows when
# the random forest or knn model fails (the models can not extrapolate beyond the seen data and therefore fails
# for the given dataset/problem, because the weight of the pigs keeps on growing from timepoint to timepoint
# and only the first timepoints are used for training)
def sample_dataset_analysis():

    # get the dataset
    data = smf.datasets.get_rdataset("dietox", "geepack").data

    # create a separate ID column
    data["ID"] = data.groupby("Pig").ngroup()

    # first, split the entire dataset in data subsets for each participant
    split_dataframes = [y for x, y in data.groupby("ID", as_index=True)]

    train_dfs = []
    test_dfs = []
    # iterate the dataframe of each participant
    # measurements
    for df in split_dataframes:
        # sort it by the timestamp column to order the trials chronologically
        df = df.sort_values("Time").reset_index()
        # add an order column
        df["order"] = range(len(df))
        # get the first 80% of the dataset
        df_80 = df.head(int(len(df) * .8))
        # get the remaining 20% of the dataset
        df_20 = df.iloc[max(df_80.index) + 1:]
        # add the participant train and test datasets to the list of training and test dfs
        train_dfs.append(df_80)
        test_dfs.append(df_20)

    # combine the training and test datasets of each participant to create the final training and test dataset with data
    # from all participants
    training_dataset, test_dataset = pd.concat(train_dfs, ignore_index=True), pd.concat(test_dfs, ignore_index=True)

    print("Mixed Model Generalized")
    ml_mixed_model(training_data=training_dataset, test_data=test_dataset, preds=["Time", "ID"],
                   target='Weight', pred_option="generalized")
    print("\n")
    print("Mixed Model Individualized")
    ml_mixed_model(training_data=training_dataset, test_data=test_dataset, preds=["Time", "ID"],
                   target='Weight', pred_option="individualized")
    print("\n")
    print("Random Forest")
    ml_nonlinear_regression(training_data=training_dataset, test_data=test_dataset, preds=["Time", "ID"],
                            target='Weight', ml_regressor=ml_algorithms["Random_Forest"]["model"],
                            hyperparameters=ml_algorithms["Random_Forest"]["hyperparams"], pred_option='individualized')
    print("\n")
    print("KNN")
    ml_nonlinear_regression(training_data=training_dataset, test_data=test_dataset, preds=["Time", "ID"],
                            target='Weight', ml_regressor=ml_algorithms["NearestNeighbor"]["model"],
                            hyperparameters=ml_algorithms["NearestNeighbor"]["hyperparams"],
                            pred_option='individualized')


# run the sample dataset analysis
sample_dataset_analysis()


#%%

# test the analysis procedure with a simulated dataset
# in the dataset we simulated perfectly correlated x-y relationships with different intercepts and slopes for
# different "participants" (the data structure is hierarchical)
def simulation_data_analysis():

    # helper to generate a linear related x_y pair
    def _generate_linear_x_y_vals(array):

        y_vals = random.randint(-500, 500) + random.uniform(-20.0, 20.0) * array

        return pd.DataFrame({"Target": y_vals, "Predictor": array})

    # build a dataframe that has a mixed model structure
    df_list = []

    # create 100 datasets with a perfectly correlated x-y pair, but different intercepts and slopes per dataset
    for i in range(100):
        random_corr_df = _generate_linear_x_y_vals(np.arange(-50, 50))
        random_corr_df["ID"] = i
        df_list.append(random_corr_df)

    total_df = pd.concat(df_list)

    total_df = total_df.rename(columns={0: "Target", 1: "Predictor"})

    sns.scatterplot(total_df["Predictor"], total_df["Target"])
    plt.show()

    # split the data into train and test data
    from sklearn.model_selection import train_test_split
    # first, split the entire dataset in data subsets for each participant
    split_dataframes = [y for x, y in total_df.groupby("ID", as_index=True)]

    train_dfs = []
    test_dfs = []
    # iterate the dataframe of each participant
    # measurements
    for df in split_dataframes:
        # split the sub dataframe
        train_subframe, test_subframe = train_test_split(df, test_size=0.2)
        # add the participant train and test datasets to the list of training and test dfs
        train_dfs.append(train_subframe)
        test_dfs.append(test_subframe)

    # combine the training and test datasets of each participant to create the final training and test dataset with data
    # from all participants
    training_dataset, test_dataset = pd.concat(train_dfs, ignore_index=True), pd.concat(test_dfs, ignore_index=True)

    print("Mixed Model Generalized")
    ml_mixed_model(training_data=training_dataset, test_data=test_dataset, preds=["Predictor", "ID"],
                   target='Target', pred_option="generalized")
    print("\n")
    print("Mixed Model Individualized")
    ml_mixed_model(training_data=training_dataset, test_data=test_dataset, preds=["Predictor", "ID"],
                   target='Target', pred_option="individualized")
    print("\n")
    print("Random Forest")
    ml_nonlinear_regression(training_data=training_dataset, test_data=test_dataset, preds=["Predictor", "ID"],
                            target='Target', ml_regressor=ml_algorithms["Random_Forest"]["model"],
                            hyperparameters=ml_algorithms["Random_Forest"]["hyperparams"], pred_option='individualized')
    print("\n")
    print("KNN")
    ml_nonlinear_regression(training_data=training_dataset, test_data=test_dataset, preds=["Predictor", "ID"],
                            target='Target', ml_regressor=ml_algorithms["NearestNeighbor"]["model"],
                            hyperparameters=ml_algorithms["NearestNeighbor"]["hyperparams"],
                            pred_option='individualized')
    print("\n")

    print("Random Intercept and Slope Model")
    # random intercept and slope model
    import statsmodels.formula.api as sm
    md = sm.mixedlm("Target ~ Predictor", training_dataset, groups=training_dataset["ID"], re_formula="~Predictor")
    mdf = md.fit(method=["lbfgs"])

    pseudo_mod = sm.mixedlm("Target ~ Predictor", test_dataset, groups=test_dataset["ID"], re_formula="~Predictor")
    re = mdf.random_effects
    # Multiply each predicted random effect by the random effects design matrix for one group
    # here, it basically gets the difference between the mean intercept of the model and each group intercept
    rex = [np.dot(pseudo_mod.exog_re_li[j], re[k]) for (j, k) in enumerate(pseudo_mod.group_labels)]
    # Add the fixed and random terms to get the overall prediction
    # it basically adjusts for the difference between the grand mean intercept and each group mean intercept
    # if random slopes had been added, the group specific slope would also be taken into account
    rex = np.concatenate(rex)
    preds = mdf.predict(exog=test_dataset[["Predictor"]]) + rex
    print(r2_score(test_dataset["Target"], preds))
    print(mean_absolute_error(test_dataset["Target"], preds))


# run the simulation data analysis
simulation_data_analysis()
