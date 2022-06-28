'''
Code to get descriptive statistics about:
- Sociodemographics of the sample
- The valence-arousal distribution
- The Typing-Features
The code was run in Pycharm with scientific mode turned on. The #%% symbol separates the code into cells, which
can be run separately from another (similar to a jupyter notebook)
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'''

# import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%

# import the keyboard features dataset
keyboard_data = pd.read_csv("Keyboard_Features.csv")


#%%

keyboard_data["events_per_trial"] = keyboard_data["tot_ev"] / keyboard_data["num_trials"]


#%%

###########################################
# get the Sociodemograhpics of the Sample #
###########################################

# first, calculate the number of data collections per participant after removal of "bad cases"
keyboard_data["cleaned_data_collections"] = keyboard_data.groupby('ID')['ID'].transform('size')

# to "convert" the long keyboard data dataframe into a format that has one row per participant with its sociodem data,
# simply select the sociodem columns and drop duplicates. This will remove all but one row per participant (because
# the sociodem data of each participant is the same across all data collections)
sociodem_items = ["age", "sex", "finished", "nationality", "occupation", "os", "app_version", "num_data_colls",
                  'cleaned_data_collections', "ID"]

sociodem_data = keyboard_data.loc[:, sociodem_items].drop_duplicates().reset_index(drop=True)

# get some sociodemographic stats about the sample

# get basic sociodemographic info about the dataset
print(f"Total number of participants who completed at least one data collection: {len(sociodem_data)}")
print(f"Number of participants who finished the study (saw the study-end page): {sociodem_data['finished'].sum()}")
print("\n")
# age (1 = younger than 30, 2 = 30-39, 3 = 40-49, 4 = 50-59, 5 = 60 or older, -99 = missing)
print(f"Age distribution\n{sociodem_data['age'].value_counts()}")
print("\n")
# sex (0 = female, 1 = male, 2 = other, -99 = missing)
print(f"Sex distribution\n{sociodem_data['sex'].value_counts()}")
print("\n")
print(f"Nationality distribution\n{sociodem_data['nationality'].value_counts()}")
print("\n")
# occupation (0 = working, 1 = student, 2 = other, -99 = missing)
print(f"Occupation distribution\n{sociodem_data['occupation'].value_counts()}")
print("\n")
print(f"OS distribution\n{sociodem_data['os'].value_counts()}")
print("\n")
print(f"App version distribution\n{sociodem_data['app_version'].value_counts()}")
print("\n")
print(f"Number of completed trials:\n{pd.to_numeric(sociodem_data['num_data_colls']).sum()}")
print(f"Descriptive stats about completed trials:\n{pd.to_numeric(sociodem_data['num_data_colls']).describe()}")
print("\n")
print(f"Number of completed trials after cleaning:\n{pd.to_numeric(sociodem_data['cleaned_data_collections']).sum()}")
print(f"Descriptive stats about completed trials after cleaning:\n{pd.to_numeric(sociodem_data['cleaned_data_collections']).describe()}")

#%%

# helper functions

# helper function to plot the valence/arousal distribution of the data
def plot_valence_arousal(data, filename):
    fig, ax = plt.subplots()
    # only show the upper and left spine
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    # draw a scatterplot of the relationship between valence and arousal
    val_arousal_plot = sns.scatterplot(data=data, x='valence', y='arousal', s=8)
    # add vertical and horizontal line
    val_arousal_plot.axhline(50, ls='--', color='black', alpha=0.6)
    val_arousal_plot.axvline(50, ls='--', color='black', alpha=0.6)
    # add custom x and y ticks
    val_arousal_plot.set(xticks=np.arange(0, 101, 50), yticks=np.arange(0, 101, 50))
    # add custom text to the axis
    ax.text(-0.05, 0.25, 'calm',
            horizontalalignment='center',
            verticalalignment='center',
            rotation='vertical',
            transform=ax.transAxes,
            fontsize=11)
    ax.text(-0.05, 0.75, 'excited',
            horizontalalignment='center',
            verticalalignment='center',
            rotation='vertical',
            transform=ax.transAxes,
            fontsize=11)
    ax.text(0.25, -0.05, 'negative',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=11)
    ax.text(0.75, -0.05, 'positive',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=11)

    # plt.show()
    # plt.savefig(filename + '.png')
    plt.show()


#%%

# get some info about the valence and arousal distribution in the dataset
# -----------------------------------------------------------------------

# get descriptive stats about valence and arousal in the total dataset
print(f"Descriptive Info about Valence and Arousal:\n{keyboard_data.loc[:, ['valence', 'arousal']].describe()}")
# plot the valence/arousal distribution
plot_valence_arousal(keyboard_data, "Valence_Arousal_Plot")


#%%

##############################################################################
# Descriptive Stats and Data Visualizations for the Keyboard Typing Features #
##############################################################################

# list all typing features + the target variables valence & arousal
typing_features = ['tot_ev', 'corr_ev', 'numpad_ev', 'shift_cpslck_ev', 'type_time', 'dwelltime_mean',
                   'dwelltime_median', 'dwelltime_sd', 'latency_mean', 'latency_median', 'latency_sd',
                   'down_down_mean', 'down_down_median', 'down_down_sd', 'up_up_mean', 'up_up_median',
                   'up_up_sd', 'no_key_pushed', 'num_trials', 'arousal', 'valence']

# Visualization Helper Functions
# -------------------------------

# helper function to create a kdeplot of selected keyboard usage features
def multi_kde_plot(data, name):

    # set a style
    sns.set_style("white")

    # first rename the columns of the dataframe to give the variables in the plots "prettier" names
    # this is not an ideal coding solution!
    # data.columns = [rename_dict.get(x, x) for x in data.columns]

    # create a plot with an appropriate number of columns and rows (depending of the number of the columns to plot
    num_cols = data.shape[1]

    fig, axes = plt.subplots(nrows=int(np.sqrt(num_cols)) + 1, ncols=int(np.sqrt(num_cols)) + 1,
                             figsize=(30, 30), sharex=False, sharey=False)
    axes = axes.ravel()  # array to 1D
    cols = list(data.columns)  # create a list of dataframe columns to use

    for col, ax in zip(cols, axes):
        sns.set(font_scale=2.25)
        sns.kdeplot(data=data, x=col, shade=True, ax=ax)
        ax.set(title=col, xlabel=None, xticklabels=[], yticklabels=[])

    # delete the empty subplots
    ax_to_del = [i for i in range(num_cols, len(axes))]

    for i in ax_to_del:
        fig.delaxes(axes[i])

    fig.tight_layout()
    # plt.savefig('Plots_in_Paper/' + name + '_kde_plot.png')
    plt.show()


# helper function to plot a correlation heatmap
def correlation_heatmap(data, fig_size, font_scale, name, add_text=True):

    # first rename the columns of the dataframe to give the variables in the plots "prettier" names
    # this is not an ideal coding solution!
    # data.columns = [rename_dict.get(x, x) for x in data.columns]

    # calculate the correlation matrix of the data
    corr = data.corr()

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
    plt.tight_layout()
    # plt.savefig(name + '.png')
    # plot the heatmap
    plt.show()

#%%

# First, get some descriptive stats about the typing features
typing_feature_descriptives = keyboard_data.loc[:, typing_features].describe()

#%%

# Create a KDE Plot of the distributions of each unprocessed typing feature
multi_kde_plot(keyboard_data.loc[:, typing_features], "KDE_Plot_Typing_Features")

#%%

# Create a correlation heatmap of the unprocessed typing features
correlation_heatmap(keyboard_data.loc[:, typing_features], fig_size=(48, 38), font_scale=3.8,
                    name="Corr_Heatmap_Typing_Features", add_text=True)

#%%

########################################################################################
# Careful "conclusions" from looking at the descriptive stats of the keyboard features #
########################################################################################

# outlier handling (-> there are some outliers in the data that might present potential logging problems or
# careless responding)
# -Use all raw data
# -Handpicked outlier removal thresholds
# -Automatic outlier removal procedure using IQR

# data transformation?
# - using a transformer to make data distribution more normal-like?

# collinearity handling
# - remove highly correlated features (some features seem to be almost identical)
