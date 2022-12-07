# imports
import numpy as np
import csv
import os
import pathlib
import matplotlib.pyplot as plt

# get file path
file_path = pathlib.Path(os.getcwd()).parent.parent.joinpath("data2019.csv")

# read in the data from csv file
with open(file_path) as data:
    csv = csv.reader(data, delimiter=';')
    row_count = 0
    consumption_values = []
    for row in csv:
        if row_count % 2 == 1:
            if row_count == 35041:
                pass
            else:
                consumption_values.append(row[4])
        row_count += 1

# Check dimension
if len(consumption_values) != 17520:
    raise ValueError('Not enough values')

# Obtain minimum and maximum values
# print(f'min: {min(consumption_values)} \nmax: {max(consumption_values)}')

# Winter from January to March and from Oktober to December
winter_to_summer = 48 * (31 + 28 + 31)
summer_to_winter = 48 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30)
jan = 48 * (31)
feb = 48 * (31 + 28)
mar = 48 * (31 + 28 + 31)
apr = 48 * (31 + 28 + 31 + 30)
may = 48 * (31 + 28 + 31 + 30 + 31)
jun = 48 * (31 + 28 + 31 + 30 + 31 + 30)
jul = 48 * (31 + 28 + 31 + 30 + 31 + 30 + 31)
aug = 48 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31)
sep = 48 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30)
oct = 48 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31)
nov = 48 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30)
dec = 48 * (31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31)

if dec != 17520:
    raise ValueError("Wrong number of data points!")

# Obtaining data sets for winter and summer and for each month
winter_values = consumption_values[:winter_to_summer] + consumption_values[summer_to_winter:]
summer_values = consumption_values[winter_to_summer:summer_to_winter]
year_values = consumption_values

jan_values = consumption_values[:jan]
feb_values = consumption_values[jan:feb]
mar_values = consumption_values[feb:mar]
apr_values = consumption_values[mar:apr]
may_values = consumption_values[apr:may]
jun_values = consumption_values[may:jun]
jul_values = consumption_values[jun:jul]
aug_values = consumption_values[jul:aug]
sep_values = consumption_values[aug:sep]
oct_values = consumption_values[sep:oct]
nov_values = consumption_values[oct:nov]
dec_values = consumption_values[nov:dec]


def create_avg_day(value_list):
    """

    :param value_list: list with values for 1:n days with 48 values per day
    :return: list with avg value for time periods
    """
    n_days = int(len(value_list) / 48)
    avg_day_values = np.zeros((48,))
    for time_point in range(48):
        value_sum = 0
        for day in range(n_days):
            value_sum += int(value_list[time_point + day * 48])
        avg_value = value_sum / n_days
        avg_day_values[time_point] = avg_value
    return avg_day_values


avg_day_year = create_avg_day(year_values)
avg_day_winter = create_avg_day(winter_values)
avg_day_summer = create_avg_day(summer_values)
avg_day_jan = create_avg_day(jan_values)
avg_day_feb = create_avg_day(feb_values)
avg_day_mar = create_avg_day(mar_values)
avg_day_apr = create_avg_day(apr_values)
avg_day_may = create_avg_day(may_values)
avg_day_jun = create_avg_day(jun_values)
avg_day_jul = create_avg_day(jul_values)
avg_day_aug = create_avg_day(aug_values)
avg_day_sep = create_avg_day(sep_values)
avg_day_oct = create_avg_day(oct_values)
avg_day_nov = create_avg_day(nov_values)
avg_day_dec = create_avg_day(dec_values)

avg_day_dict = {
    'avg_day_year': avg_day_year,
    'avg_day_winter': avg_day_winter,
    'avg_day_summer': avg_day_summer,
    'avg_day_jan': avg_day_jan,
    'avg_day_feb': avg_day_feb,
    'avg_day_mar': avg_day_mar,
    'avg_day_apr': avg_day_apr,
    'avg_day_may': avg_day_may,
    'avg_day_jun': avg_day_jun,
    'avg_day_jul': avg_day_jul,
    'avg_day_aug': avg_day_aug,
    'avg_day_sep': avg_day_sep,
    'avg_day_oct': avg_day_oct,
    'avg_day_nov': avg_day_nov,
    'avg_day_dec': avg_day_dec,
}

time_points = np.linspace(0.5, 24, 48)
fig, ax = plt.subplots()
summer_line, = ax.plot(time_points, avg_day_summer, label='Summer')
winter_line, = ax.plot(time_points, avg_day_winter, label='Winter')
jan_line, = ax.plot(time_points, avg_day_jan, label='January')
# feb_line, = ax.plot(time_points, avg_day_feb, label='February')
# mar_line, = ax.plot(time_points, avg_day_mar, label='March')
# apr_line, = ax.plot(time_points, avg_day_apr, label='April')
# may_line, = ax.plot(time_points, avg_day_may, label='May')
# jun_line, = ax.plot(time_points, avg_day_jun, label='June')
# jul_line, = ax.plot(time_points, avg_day_jul, label='July')
aug_line, = ax.plot(time_points, avg_day_aug, label='August')
# sep_line, = ax.plot(time_points, avg_day_sep, label='September')
# oct_line, = ax.plot(time_points, avg_day_oct, label='October')
# nov_line, = ax.plot(time_points, avg_day_nov, label='November')
# dec_line, = ax.plot(time_points, avg_day_dec, label='December')
ax.legend(handles=[
    jan_line,
    # feb_line,
    # mar_line,
    # apr_line,
    # may_line,
    # jun_line,
    # jul_line,
    aug_line,
    # sep_line,
    # oct_line,
    # nov_line,
    # dec_line,
    winter_line,
    summer_line,
])

ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
ax.set_yticks([20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000])
ax.set_xlabel('Time')
ax.set_ylabel('MW')
# plt.show()

# print(f'January: {avg_day_jan} \n'
#       f'August: {avg_day_aug} \n'
#       f'Winter: {avg_day_winter} \n'
#       f'Summer: {avg_day_summer}')
