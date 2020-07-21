"""
Plot histograms
"""
from numpy import array
from numpy import dstack
from numpy import unique
from pandas import read_csv
from matplotlib import pyplot


# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


# load a dataset group, such as train or test
def load_dataset(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


# get all data for one subject
def data_for_subject(X, y, sub_map, sub_id):
    # get row indexes for the subject id
    ix = [i for i in range(len(sub_map)) if sub_map[i] == sub_id]
    # return the selected samples
    return X[ix, :, :], y[ix]


# convert a series of windows to a 1D list
def to_series(windows):
    series = list()
    for window in windows:
        # remove the overlap from the window
        half = int(len(window) / 2) - 1
        for value in window[-half:]:
            series.append(value)
    return series


# group data by activity
def data_by_activity(X, y, activities):
    # group windows by activity
    return {a: X[y[:, 0] == a, :, :] for a in activities}


labels = ['oś X', 'oś Y', 'oś Z']
colors = ['c', 'g', 'y']
activities_description = {
    1: 'Walk',
    2: 'Walk up',
    3: 'Walk down',
    4: 'Sitting',
    5: 'Standing',
    6: 'Laying'
}


# plot histograms for each activity for a subject
def plot_activity_histograms1(X, y):
    # get a list of unique activities for the subject
    activity_ids = unique(y[:, 0])
    # group windows by activity
    grouped = data_by_activity(X, y, activity_ids)
    # plot per activity, histograms for each axis
    pyplot.figure()

    xaxis = None
    for k in range(len(activity_ids)):
        act_id = activity_ids[k]
        # total acceleration
        for i in range(3):
            ax = pyplot.subplot(len(activity_ids), 1, k + 1, sharex=xaxis)
            ax.set_xlim(-1, 1)
            if k == 0:
                xaxis = ax
            pyplot.hist(to_series(grouped[act_id][:, :, i]), bins=100, color=colors[i], label=labels[i])
            pyplot.title(activities_description[act_id], y=0, loc='left')
            if (act_id == 1):
                pyplot.title('Accelerometer data + G histogram')
            pyplot.ylabel('Count')
            if (act_id == 6):
                pyplot.xlabel('Amplitude')
    pyplot.show()


def plot_activity_histograms2(X, y):
    # get a list of unique activities for the subject
    activity_ids = unique(y[:, 0])
    # group windows by activity
    grouped = data_by_activity(X, y, activity_ids)
    # plot per activity, histograms for each axis
    pyplot.figure()
    xaxis = None
    for k in range(len(activity_ids)):
        act_id = activity_ids[k]
        # total acceleration
        for i in range(3):
            ax = pyplot.subplot(len(activity_ids), 1, k + 1, sharex=xaxis)
            ax.set_xlim(-1, 1)
            if k == 0:
                xaxis = ax
            pyplot.hist(to_series(grouped[act_id][:, :, 3 + i]), bins=100, color=colors[i], label=labels[i])
            pyplot.title(activities_description[act_id], y=0, loc='left')
            if (act_id == 1):
                pyplot.title('Accelerometer data histogram')
            pyplot.ylabel('Count')
            if (act_id == 6):
                pyplot.xlabel('Amplitude')
    pyplot.show()


def plot_activity_histograms3(X, y):
    # get a list of unique activities for the subject
    activity_ids = unique(y[:, 0])
    # group windows by activity
    grouped = data_by_activity(X, y, activity_ids)
    # plot per activity, histograms for each axis
    pyplot.figure()
    xaxis = None
    for k in range(len(activity_ids)):
        act_id = activity_ids[k]
        # total acceleration
        for i in range(3):
            ax = pyplot.subplot(len(activity_ids), 1, k + 1, sharex=xaxis)
            ax.set_xlim(-1, 1)
            if k == 0:
                xaxis = ax
            pyplot.hist(to_series(grouped[act_id][:, :, 6 + i]), bins=100, color=colors[i], label=labels[i])
            pyplot.title(activities_description[act_id], y=0, loc='left')
            if (act_id == 1):
                pyplot.title('Gyro data histogram')
            pyplot.ylabel('Count')
            if (act_id == 6):
                pyplot.xlabel('Amplitude')
    pyplot.show()


# load data
trainX, trainy = load_dataset('train', '/home/marcin/Pobrane/HARDataset/')
# load mapping of rows to subjects
sub_map = load_file('/home/marcin/Pobrane/HARDataset/train/subject_train.txt')
train_subjects = unique(sub_map)
# get the data for one subject
sub_id = train_subjects[0]
subX, suby = data_for_subject(trainX, trainy, sub_map, sub_id)
# plot data for subject
plot_activity_histograms1(subX, suby)
plot_activity_histograms2(subX, suby)
plot_activity_histograms3(subX, suby)
