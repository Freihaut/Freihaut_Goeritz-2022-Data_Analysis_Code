'''
Code to transform the raw typing data into a selected typing features dataset that can be used for further analysis
The code was run in Pycharm with scientific mode turned on. The #%% symbol separates the code into cells, which
can be run separately from another (similar to a jupyter notebook)
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'''

#########
# Setup #
#########

import gzip
import json
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%

with gzip.open("data/Long_Keyboard_Raw_Dataset.json.gz", "rb") as f:
    study_data = json.loads(f.read())


#%%

############################################################################################################
# Get the sociodemograhpics of the sample (to merge them into the typing features dataset in a later step) #
############################################################################################################

sociodemographics = {}

for par in study_data:

    sociodemographics[par] = {}

    # check if the participant finished the study (=reached the study-end page)
    if "finishedStudy" in study_data[par]:
        sociodemographics[par]["finished"] = True
    else:
        sociodemographics[par]["finished"] = False

    # grab sociodem info from the dataset and count the number of completed data collections (=trials)
    trials = 0
    for dset in study_data[par]:

        if isinstance(study_data[par][dset], dict):

            trials += 1

            if "Sociodemographics" in study_data[par][dset]:

                # get sociodem data
                sociodemographics[par].update(study_data[par][dset]["Sociodemographics"])
                # get app and os infos
                sociodemographics[par]["os"] = study_data[par][dset]["os"]
                sociodemographics[par]["app_version"] = study_data[par][dset]["appVersion"]

                trials -= 1

    # get the number of data collections that each participant has completed during the study
    sociodemographics[par]["num_data_colls"] = trials

# convert the dictionary to a pandas dataframe
sociodemographics = pd.DataFrame(sociodemographics).T

# change the id index to an id column
sociodemographics = sociodemographics.reset_index()
sociodemographics = sociodemographics.rename({"index": "ID"}, axis=1)

#%%

# get basic sociodem info about the complete (unprocessed dataset)
print(f"Total number of participants: {len(sociodemographics)}")
print(f"Desc. stats about the number of recorded trials:\n{pd.to_numeric(sociodemographics['num_data_colls']).describe()}")
print(f"Number of participants without recorded typing-trials:{sum(sociodemographics['num_data_colls'] == 0)}")

#%%

#################################################
# Process the Keyboard data of the participants #
#################################################


# Helper function to process the keyboard data
# ---------------------------------------------

# process the raw typing data, sort it by the timestamp and remove potential artifacts to get a final list of
# typing events in which each keydown event has a corresponding keyup event
def extract_typing_data(data):
    # Only include Keydown and Keyup events to exclude mouse data
    key, value1, value2 = "eventType", "KeyDown", "KeyUp"

    # The keyboard event listener was attached to the entire Study-App window and not the password input field.
    # This caused some keys to be recorded, which are not part of the actual typing behavior. Those keys
    # should be filtered out (it would have been a better solution to attach the event listener to the input field).
    # The filtering method does not guarantee that all "non-password typing keys" are filtered out
    # the meta key is the windows key (and an equivilant key on mac), HangulMode changes the language setting on the
    # keyboard
    keys_to_filter = ["Meta", "HangulMode", "Unidentified", "AudioVolumeMute", "AudioVolumeDown", "AudioVolumeUp", "Tab"]
    # lock the number of filtered out key presses
    filtered_key_presses = 0

    # sort the data by its timestamp (sorted returns a list with tuples (id, logged_event) --> only the logged event
    # data is relevant --> only get the second entry of the tuple
    sorted_page_data = sorted(data.items(), key=lambda x: x[1]["time"])

    # Keyboard usage data
    keyboard_events = []
    # Keydown events to remove doubled down events
    key_down_events = {}
    # number of potential artifacts
    no_corresponding_event = 0

    # Gets all keyup and keydown events and deletes doubled keydown events which occur if the key is held down
    # for a longer time
    # the data is stored in tuples: ("0", {datapoint})
    # only keypresses are saved which have a keydown and a corresponding keyup event, keydown events with no
    # corresponding up event and keyup events with no corresponding down events are removed from the dataset
    for data_tuple in sorted_page_data:
        # the second tuple is the datapoint, the first tuple the dictionary key
        datapoint = data_tuple[1]
        # filter out the keys that are not part of the password typing
        if datapoint["key"] not in keys_to_filter:
            # if it is a keydown event
            if key in datapoint and datapoint[key] == value1:
                # if its a non existing keydown event, add it to the dict of keydown events without a corresponding up
                if datapoint["code"] not in key_down_events:
                    key_down_events[datapoint["code"]] = [datapoint]
                # if the keydown event already exists
                else:
                    # if the time difference between this keydown event and the last keydown event is smaller than
                    # 1750 ms, it is likely that the key was held down and more than one keydown events were fired
                    if datapoint["time"] - key_down_events[datapoint["code"]][-1]["time"] < 1750:
                        key_down_events[datapoint["code"]].append(datapoint)
                    else:
                        # if there is a greater time difference between the keydown events, every key down event prior
                        # to the current keydown event is likely an artifact that has no corresponding keyup event
                        # (e.g. if participants switched tabs during typing)
                        # those potential artifacts are removed from the dataset (they are overwritten by the latest
                        # keydown event
                        # print("Potential artifact", datapoint["time"] - key_down_events[datapoint["code"]][-1]["time"])
                        # print("Old:", key_down_events[datapoint["keyCode"]])
                        # print("New:", datapoint)
                        no_corresponding_event += 1
                        key_down_events[datapoint["code"]] = [datapoint]
            # if it is a keyup event
            elif key in datapoint and datapoint[key] == value2:
                # search for the corresponding down event
                if datapoint["code"] in key_down_events:
                    # append the keyup event and its corresponding keydown event to the cleaned keyboard event list
                    keyboard_events.append(key_down_events[datapoint["code"]][0])
                    keyboard_events.append(datapoint)
                    # if there are more than one keydown events corresponding the the keyup event, print that information
                    # if len(key_down_events[datapoint["code"]]) > 1:
                    #     print("More than one Keydown Event for the keyUp event")
                    #     print(key_down_events[datapoint["code"]])
                    # delete the keydown event entry of that key (because the corresponding keyup event was found)
                    del key_down_events[datapoint["code"]]
                else:
                    # If they Keyup event has no corresponding keydown event, it is likely an artifact
                    no_corresponding_event += 1
        # if it is one of the filtered out keys, log it
        else:
            filtered_key_presses += 1


    # add remaining keydown events with no keyup events to the no_corresponding_events counter
    no_corresponding_event += len(key_down_events)

    # sort the keyboard events list by timestamps again to get a cleaned list of valid keyboard events of the typing
    # task: The list should contain each valid keydown event with a corresponding keyup event
    keyboard_events = sorted(keyboard_events, key=operator.itemgetter("time"))

    # return the cleaned list of keyboard events and potential artifacts (key-up or key-down events with no
    # corresponding down-up key)
    return keyboard_events, no_corresponding_event, filtered_key_presses


# NOT NEEDED IN THE PRESENT STUDY
# helper function to remove potential keydown/keyup events with no corresponding keyup/keydown events in isolated trials
# the data for this function was already cleaned in a first step, it is repeated if keyup/keydown pairs are split
# between trials
def clean_trial(data):

    cleaned_trial_data = []
    # create a dic for the keydown events
    down_events = {}
    # loop all keyboard datapoints
    for dpoint in data:
        # if its a keydown event, add it to the dict of keydown events
        if dpoint["eventType"] == "KeyDown":
            down_events[dpoint["code"]] = dpoint
        # if its a keyup event
        elif dpoint["eventType"] == "KeyUp":
            # look in the down events dict, if the corresponding key exists, save the pair
            if dpoint["code"] in down_events:
                cleaned_trial_data.append(down_events[dpoint["code"]])
                cleaned_trial_data.append(dpoint)

    # sort the cleaned list by timestamps
    cleaned_trial_data = sorted(cleaned_trial_data, key=operator.itemgetter("time"))

    return cleaned_trial_data


#%%

# Helper Function to calculate keyboard typing features
# -----------------------------------------------------
# a seperate function for each feature (or related features): This was done to increase the readability of the code
# From a Performance point of view, it might be better to not loop the keyboard data separately for the calculation of
# each feature (and might need refinement in cases with large datasets)

# helper function to get information about specific keypresses
def get_pressed_keys_info(keyboard_data):

    # get information about the total number of key events (and special key events)
    total_events = 0
    correction_events = 0
    numpad_events = 0
    shift_capslock_presses = 0

    correction_keys = ["Backspace", "Delete"]

    for dpoint in keyboard_data:
        total_events += 1
        if dpoint["code"] in correction_keys:
            correction_events += 1
        elif "Numpad" in dpoint["code"]:
            numpad_events += 1
        elif "Shift" in dpoint["code"] or "CapsLock" in dpoint["code"]:
            shift_capslock_presses += 1

    # return the number of key presses not the number of total events
    return {"tot_ev": total_events/2, "corr_ev": correction_events/2, "numpad_ev": numpad_events/2,
            "shift_cpslck_ev": shift_capslock_presses/2}


# get the writing time (time difference between last and first keyboard event in the dataset)
def get_writing_time(keyboard_data):

    # get the max and min timestamp and calculate the difference between them (keyboard_data) should already be
    # chronologically ordered in regards to time --> a faster solution, which is tallied to the data structure is
    # writing_time = keyboard_data[-1]["time"] - keyboard_data[0]["time"]
    # get the writing time in seconds = division by 1000
    typing_time = (max(i["time"] for i in keyboard_data) - min(i["time"] for i in keyboard_data)) / 1000

    return {"type_time": typing_time}


# Calculate the time between pressing a key and releasing a key --> Keypress Dwell Time
def get_keypress_dwell_time(keyboard_data):

    key, value1, value2 = "eventType", "KeyDown", "KeyUp"

    # store the individual keypress dwell times
    key_press_time = []
    # Hold information on which key is pressed down
    key_pushed = []

    # loop over all keyboard data
    for datapoint in keyboard_data:
        # if the datapoint is a keydown event
        if datapoint[key] == value1:
            # add the datapoint to the key_pushed list
            key_pushed.append(datapoint)
        # If the datapoint is a keyup event
        elif datapoint[key] == value2:
            # loop the key_pushed list and search for the corresponding keydown event to the keyup event
            for i in range(len(key_pushed)):
                # if the keydown to the corresponding keyup is found
                if datapoint["code"] == key_pushed[i]["code"]:
                    # calculate the time it took between pressing and releasing the key and append it to the
                    # key_press_time list
                    key_press_time.append(datapoint["time"] - key_pushed[i]["time"])
                    # delete the keydown event from the list and break the inner loop
                    del key_pushed[i]
                    break

    # calculate the mean, median and standard deviation of the keypress dwell times
    dwelltime_mean = np.round(np.mean(key_press_time), 3)
    dwelltime_median = np.median(key_press_time)
    dwelltime_sd = np.round(np.std(key_press_time), 3)

    return {"dwelltime_mean": dwelltime_mean, "dwelltime_median": dwelltime_median, "dwelltime_sd": dwelltime_sd}


# Calculate the time between releasing a key and pressing the next key (can contain negative values if a key is pressed
# before the previous key is released)
def get_keypress_latency(keyboard_data):

    key, value1, value2 = "eventType", "KeyDown", "KeyUp"

    # store the individual latency times
    latency_times = []

    # initialize variables
    next_key_down_time = 0
    currentkey_up_time = 0
    down_time_set = False
    up_time_set = False

    # loop over all datapoints: if it is a keydown event, loop over all consecutive datapoints until the time of the
    # corresponding keyup event and the time of the next keydown event is found (latency as the time difference
    # between pressing the next button and releasing the previous button
    for datapoint in keyboard_data:
        # if its a keydown press and not a shift key
        if datapoint[key] == value1:
            # loop over all consecutive datapoints to find the corresponding keyup event
            for i in keyboard_data[(keyboard_data.index(datapoint) + 1):]:
                # if the consecutive datapoint is the corresponding keyup event to the previous datapoint
                if (i[key] == value2) and (i["code"] == datapoint["code"]) and (up_time_set is False):
                    # save the timepoint of the keyup event, tell the algorithm its ready and break the loop
                    currentkey_up_time = i["time"]
                    up_time_set = True
                    break
            # loop over all consecutive datapoints to find the corresponding keydown event
            for i in keyboard_data[(keyboard_data.index(datapoint) + 1):]:
                # if the consecutive datapoint is the next keydown event
                if (i[key] == value1) and (down_time_set is False):
                    # save the timestamp of the keydown event, tell the algorithms its ready and break the loop
                    next_key_down_time = i["time"]
                    down_time_set = True
                    break
            # if there is a new up time and a new down time calculate a latency time and save it in the array
            if down_time_set and up_time_set:
                latency_times.append(next_key_down_time - currentkey_up_time)
            # Reset to False to prevent wrong latency calculations
            down_time_set = False
            up_time_set = False

    # Calculate the mean, median and standard deviation of the keypress latency time
    mean_latency = np.round(np.mean(latency_times), 3)
    median_latency = np.median(latency_times)
    sd_latency = np.round(np.std(latency_times), 3)

    return {"latency_mean": mean_latency, "latency_median": median_latency, "latency_sd": sd_latency}


# calculate the time between pressing a key and pressing the next key (down-down time) and the time between
# releasing a key and releasing the next key (up-up time)
def get_down_down_and_up_up_time(keyboard_data):

    key, value1, value2 = "eventType", "KeyDown", "KeyUp"

    # Save the keydown to keydown times
    down_down_times = []
    last_down_time = 0

    # save the keyup to keyup times
    up_up_times = []
    last_up_time = 0

    # Loop over all datapoints
    for datapoint in keyboard_data:
        # If the event is keydown and its not the first keydown event
        if datapoint[key] == value1:
            if last_down_time != 0:
                # save the time between the keydown event and the previous keydown event in the array
                down_down_times.append(datapoint["time"] - last_down_time)
            # set the last keydown time to the time of the current keydown event
            last_down_time = datapoint["time"]
        # if the event is a keyup and its not the first keyup event
        elif datapoint[key] == value2:
            if last_up_time != 0:
                # save the time between the keyup event and the previous keyup even in the array
                up_up_times.append(datapoint["time"] - last_up_time)
            # set the last keyup time to the time of the current keyup event
            last_up_time = datapoint["time"]

    # Calculate the mean, median and standard deviation of the down-down and up-up times
    mean_down_down = np.round(np.mean(down_down_times), 3)
    median_down_down = np.median(down_down_times)
    sd_down_down = np.round(np.std(down_down_times), 3)

    mean_up_up = np.round(np.mean(up_up_times), 3)
    median_up_up = np.round(np.median(up_up_times), 3)
    sd_up_up = np.round(np.std(up_up_times), 3)

    return {"down_down_mean": mean_down_down, "down_down_median": median_down_down, "down_down_sd": sd_down_down,
            "up_up_mean": mean_up_up, "up_up_median": median_up_up, "up_up_sd": sd_up_up}


# calculate the percentage of time that no key is pressed/that at least one key is pressed
def get_keypress_percentage(keyboard_data):

    key, value1, value2 = "eventType", "KeyDown", "KeyUp"

    # Save the duration of the numbers of pushed keys in a dictionary
    # {"numberOfKeysPushed0": [time1, time2, ..., timen], "numerOfKeysPushed1": [time1...],...}
    keys_pushed_at_same_time = {0:0}
    # log the number of pushed keys
    number_of_keys_pushed = 0
    # log the last timestamp
    last_timestamp = 0

    for datapoint in keyboard_data:
        # if datapoint is keydown and not a shiftkey
        if datapoint[key] == value1:
            # Get the duration between this key event and the last key event and log its time as well as the information
            # about how many keys are pressed down when this event happened
            if last_timestamp != 0:
                # if the entry doesnt exist already make a new entry else add to the existing entry
                if number_of_keys_pushed not in keys_pushed_at_same_time:
                    keys_pushed_at_same_time[number_of_keys_pushed] = datapoint["time"] - last_timestamp
                else:
                    keys_pushed_at_same_time[number_of_keys_pushed] += datapoint["time"] - last_timestamp
            # Save the timestamp of the pushed key
            last_timestamp = datapoint["time"]
            # increase the numbers of keys pushed
            number_of_keys_pushed += 1
            # test.append(len(keyPushed))
        # If datapoint is keyup and not a shiftkey
        elif datapoint[key] == value2:
            # Get the duration between this key event and the last key event and log its time as well as the information
            # about how many keys are pressed down when this event happened
            if number_of_keys_pushed != 0:
                if number_of_keys_pushed not in keys_pushed_at_same_time:
                    keys_pushed_at_same_time[number_of_keys_pushed] = datapoint["time"] - last_timestamp
                else:
                    keys_pushed_at_same_time[number_of_keys_pushed] += datapoint["time"] - last_timestamp
                # save the timestamp of the released key
                last_timestamp = datapoint["time"]
                # decrease the number of keys pushed down
                number_of_keys_pushed -= 1

    # Calculate Writing time (duration of 0,1,2, ..., n keys-pushed at the same time)
    total_time = sum(keys_pushed_at_same_time.values())

    # calculate the proportion of the duration that no key is pushed; the percentage that at least one key is pressed
    # is the reverse number, only one feature needs to be calculated
    no_key_pushed = np.round(keys_pushed_at_same_time[0] / total_time * 100, 3)

    return {"no_key_pushed": no_key_pushed}


# helper function to extract the written text (the password) to check the data quality
def get_text(keyboard_data):

    key, value1, value2 = "eventType", "KeyDown", "KeyUp"

    # text of logged "codes"
    code_text = []
    # text of logged "keys"
    key_text = []
    # try to filter keys that "should not appear" (do not belong to the password)
    target_keys = "BballOorange3829!"
    bad_text = []

    for datapoint in keyboard_data:
        if datapoint[key] == value2:
            code_text.append(datapoint["code"])
            key_text.append(datapoint["key"])
            if datapoint["key"] not in target_keys and "Shift" not in datapoint["key"] and "Backspace" not in \
                    datapoint["key"] and "CapsLock" not in datapoint["key"] and "Delete" not in datapoint["key"] and "Enter"\
                    not in datapoint["key"]:
                bad_text.append(datapoint["key"])

    return {"text_code": code_text, "text_key": key_text, "bad_text": bad_text}


#%%

# helper functions to bring together the calculated keyboard typing features
# --------------------------------------------------------------------------

# short helper to calculate keyboard features if the features are calculated by typing trial (which is not done
# in the present study, because this strongly increases the number of features and features are likely highly
# correlated)
def calc_all_features(key_data, name):

    # calc all keyboard features
    features = {
        **get_text(key_data),
        **get_pressed_keys_info(key_data),
        **get_writing_time(key_data),
        **get_keypress_dwell_time(key_data),
        **get_keypress_latency(key_data),
        **get_down_down_and_up_up_time(key_data),
        **get_keypress_percentage(key_data)
    }

    # add the name of the trial to each feature
    for i in features.copy():
        features[name + i] = features.pop(i)

    return features


# helper function to calculate all keyboard features per task to prevent the dataset creation loop from being
# overcrowded with different calculation procedures
def keyboard_feature_calculation_pipeline(key_data):

    # first, look at the total keyboard data
    all_data_features = calc_all_features(key_data, "")

    # get the number of trials needed to correctly type in the password
    trials = key_data[-1]["cor"] + 1

    # no trial calculations are done in the present study

    # # second, look at the data of the "first" typing trial
    # # extract the data of the first trial
    # first_trial_data = [i for i in key_data if i["cor"] == 0]
    # # clean it
    # first_trial_data = clean_trial(first_trial_data)
    # # calc the features
    # first_trial_features = calc_all_features(first_trial_data, "first_t_")
    #
    # # third, look at the data of the "final" passwort typing trial
    # # extract the data of the last trial
    # last_trial_data = [i for i in key_data if i["cor"] == trials]
    # # clean it
    # last_trial_data = clean_trial(last_trial_data)
    # last_trial_features = calc_all_features(last_trial_data, "last_t_")

    # return all features combined
    return {**all_data_features, "num_trials": trials}


#%%

###############################################################
# Loop the participants to get their keyboard typing features #
###############################################################

typing_features = {}

# Loop all Participants
for par in study_data:

    print(f"Processing Participant: {par}")
    # create an empty dictionary for each participant
    typing_features[par] = {}

    # loop all datasets of the participant
    for dset in study_data[par]:
        # check if the dataset is a dictionary
        if isinstance(study_data[par][dset], dict):
            # check if the dataset is writing data (and not sociodemographic data)
            if "grabbedData" in study_data[par][dset]:
                # create a dictionary for the trial
                typing_features[par][dset] = {}

                # get the cleaned keyboard data
                cleaned_keyboard_data, artifacts, filtered_keys = extract_typing_data(study_data[par][dset]["grabbedData"]["typingData"])

                # add the number of "artifacts" and filtered key presses
                typing_features[par][dset]["artifacts"] = artifacts
                typing_features[par][dset]["filtered_keys"] = filtered_keys

                # calculate the keyboard features and add them
                feats = keyboard_feature_calculation_pipeline(cleaned_keyboard_data)
                typing_features[par][dset].update(feats)

                # add the valence and arousal rating of the trial
                typing_features[par][dset]["arousal"] = study_data[par][dset]["grabbedData"]["selfReportData"]["arousal"]
                typing_features[par][dset]["valence"] = study_data[par][dset]["grabbedData"]["selfReportData"]["valence"]

                # include some additional meta information about the trial
                typing_features[par][dset]["ID"] = par
                typing_features[par][dset]["d_col_ID"] = dset
                typing_features[par][dset]["time"] = study_data[par][dset]["grabbedData"]["time"]


# convert the dictionary into a pandas dataframe
typing_features = pd.concat({k: pd.DataFrame(v).T for k, v in typing_features.items()}, axis=0).round(4).reset_index(drop=True)

#%%

# merge the typing features df and the socicodem df to create a df that contains the sociodem data in every trial row
# per participant
merged_df = typing_features.merge(sociodemographics, on="ID", how="left")


#%%

#####################################################################################################
# Inspect the calculated features to filter potential bad recordings and get a feeling for the data #
#####################################################################################################

# Potential Bad cases are:
# - recording errors (should be removed)
# - careless responding (might actually carry information about the emotional state to some extend, might be removed)

# first, take a look at descriptive stats of the numeric features (min, max, mean, median)
desc_features = merged_df.apply(pd.to_numeric, errors='ignore').select_dtypes(include=np.number).describe()

# identify potential trials in which participants used other keys than the ones needed for typing the password
# to do so, calculate a (rough) marker for "non-target" keyboard events, that is the total number of keyboard events
# minus the number of additional correction events - the number of "perfect" events required for typing the password
# this should highlight cases with different typing behavior than potentially expected (altough it does not
# necessarily highlight cases with actual bad typing behavior
merged_df["corrected_key_events"] = merged_df["tot_ev"] - merged_df["corr_ev"] - 19 * merged_df["num_trials"]

inspect_bad_keys = merged_df.loc[:, ["corrected_key_events", "tot_ev", "num_trials", "bad_text", "text_key"]]

#%%


# visualize the distribution of the calculated features
# -----------------------------------------------------
# helper function to create a kdeplot of selected mouse usage features
def multi_kde_plot(data):

    # set a style
    sns.set_style("white")

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
    # plt.savefig('keyboard_features_kde_plot.png')
    plt.show()


# visualize all numeric columns (this also includes valence, arousal, and some sociodemographics which are not relevant
multi_kde_plot(merged_df.apply(pd.to_numeric, errors='ignore').select_dtypes(include=np.number))

#%%

# the data inspection procedure revealed one case, in which the password was not typed, but copy and pasted via
# strg + v: this trial should be removed.
# There are also some potential other problematic cases:
# - some trials have extreme values for selected keyoard usage features (e.g. a very long duration or a very small
#   dwelltime)
# - in some trials, participants typed in other things than the specified password or pressed additional keys, other
#   the ones required for typing the password (e.g. they pressed the audio volumne button)

# only remove the case without typing data from the dataset in this step
cleaned_features = merged_df.loc[merged_df["tot_ev"] > 5]


#%%

# save the cleaned feature data as a csv file
cleaned_features.to_csv("Keyboard_Features.csv", index=False)


#%%


###################################################
# Code to Test Calculations for Selected Datasets #
###################################################

# time stamps are recorded incorrectly which causes some "problems" with the data
test_dataset = study_data['Zl0J4DeW5cPdQJedH0Pesw3SLXx1']['-MqAhD9VdS_PYMApPW-b']["grabbedData"]["typingData"]
test_keyboard_data, test_artifacts, test_filtered_keys = extract_typing_data(test_dataset)
test_features = keyboard_feature_calculation_pipeline(test_keyboard_data)

