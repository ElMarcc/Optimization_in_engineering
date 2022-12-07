import numpy as np
import math

np.random.seed(1)

jan_sunrise = 8.5
feb_sunrise = 8
mar_sunrise = 7
apr_sunrise = 7
may_sunrise = 6.25
jun_sunrise = 6
jul_sunrise = 6.25
aug_sunrise = 6.75
sep_sunrise = 7.5
oct_sunrise = 8.25
nov_sunrise = 8
dec_sunrise = 8.5
summer_sunrise = 6.75
winter_sunrise = 8

jan_sunset = 17.5
feb_sunset = 18.25
mar_sunset = 19
apr_sunset = 20.75
may_sunset = 21.5
jun_sunset = 21.75
jul_sunset = 21.5
aug_sunset = 21
sep_sunset = 20
oct_sunset = 19.25
nov_sunset = 17.25
dec_sunset = 17
summer_sunset = 21
winter_sunset = 18

# defines how much average power income [Wh] you have per
# peak power capacity [W] you have installed every day
# unit [Wh / W]
jan_power_income = 2.9
feb_power_income = 3.8
mar_power_income = 4.7
apr_power_income = 4.5
may_power_income = 4.6
jun_power_income = 4.8
jul_power_income = 4.9
aug_power_income = 4.9
sep_power_income = 4.6
oct_power_income = 3.8
nov_power_income = 3
dec_power_income = 2.7
summer_power_income = 4.7
winter_power_income = 3.5

# avg wind power production each month per max capacity in [MW / MW]
max_cap = 17484
jan_wind = 5524.6 / max_cap
feb_wind = 8341.9 / max_cap
mar_wind = 6035 / max_cap
apr_wind = 3158.4 / max_cap
may_wind = 3458.5 / max_cap
jun_wind = 2812.5 / max_cap
jul_wind = 2801.6 / max_cap
aug_wind = 2713.4 / max_cap
sep_wind = 2949.6 / max_cap
oct_wind = 5690.6 / max_cap
nov_wind = 4368.3 / max_cap
dec_wind = 5883.8 / max_cap
summer_wind = 2982.3 / max_cap
winter_wind = 5974 / max_cap

jan_wind_series = np.random.normal(jan_wind, jan_wind / 3, 48).clip(min=0)
feb_wind_series = np.random.normal(feb_wind, feb_wind / 3, 48).clip(min=0)
mar_wind_series = np.random.normal(mar_wind, mar_wind / 3, 48).clip(min=0)
apr_wind_series = np.random.normal(apr_wind, apr_wind / 3, 48).clip(min=0)
may_wind_series = np.random.normal(may_wind, may_wind / 3, 48).clip(min=0)
jun_wind_series = np.random.normal(jun_wind, jun_wind / 3, 48).clip(min=0)
jul_wind_series = np.random.normal(jul_wind, jul_wind / 3, 48).clip(min=0)
aug_wind_series = np.random.normal(aug_wind, aug_wind / 3, 48).clip(min=0)
sep_wind_series = np.random.normal(sep_wind, sep_wind / 3, 48).clip(min=0)
oct_wind_series = np.random.normal(oct_wind, oct_wind / 3, 48).clip(min=0)
nov_wind_series = np.random.normal(nov_wind, nov_wind / 3, 48).clip(min=0)
dec_wind_series = np.random.normal(dec_wind, dec_wind / 3, 48).clip(min=0)
summer_wind_series = np.random.normal(summer_wind, summer_wind / 3, 48).clip(min=0)
winter_wind_series = np.random.normal(winter_wind, winter_wind / 3, 48).clip(min=0)


wind_dict = {
    'jan_wind': jan_wind_series,
    'feb_wind': feb_wind_series,
    'mar_wind': mar_wind_series,
    'apr_wind': apr_wind_series,
    'may_wind': may_wind_series,
    'jun_wind': jun_wind_series,
    'jul_wind': jul_wind_series,
    'aug_wind': aug_wind_series,
    'sep_wind': sep_wind_series,
    'oct_wind': oct_wind_series,
    'nov_wind': nov_wind_series,
    'dec_wind': dec_wind_series,
    'summer_wind': summer_wind_series,
    'winter_wind': winter_wind_series,
}


def get_solar_time_series(sunrise, sunset, power_income):
    """
    Calculates the the average power income from solar panels
    every half an hour for an average day.
    """
    time_series = np.zeros((48,))
    pi = math.pi
    a = power_income * pi / (2 * (sunset - sunrise))
    b = sunset - sunrise
    time_points = np.arange(0, 24, 0.5)
    for idx, time in enumerate(time_points):
        if time + 0.5 > sunrise and time < sunset:
            starting_time = max(sunrise, time)
            ending_time = min(sunset, time + 0.5)
            power_production = -a * b / pi * math.cos(pi / b * (ending_time - sunrise)) + \
                               a * b / pi * math.cos(pi / b * (starting_time - sunrise))
            time_series[idx] = np.random.normal(power_production, power_production / 2, 1).clip(min=0)

    w = sum(time_series)
    # print(w)
    return time_series


jan_solar_time_series = get_solar_time_series(sunrise=jan_sunrise, sunset=jan_sunset, power_income=jan_power_income)
aug_solar_time_series = get_solar_time_series(sunrise=aug_sunrise, sunset=aug_sunset, power_income=aug_power_income)
summer_solar_time_series = get_solar_time_series(sunrise=summer_sunrise, sunset=summer_sunset,
                                                 power_income=summer_power_income)
winter_solar_time_series = get_solar_time_series(sunrise=winter_sunrise, sunset=winter_sunset,
                                                 power_income=winter_power_income)

solar_time_series_dict = {
    'jan_solar_time_series': jan_solar_time_series,
    'aug_solar_time_series': aug_solar_time_series,
    'summer_solar_time_series': summer_solar_time_series,
    'winter_solar_time_series': winter_solar_time_series,
}

# print(jan_solar_time_series)
# print(aug_solar_time_series)
# print(summer_solar_time_series)
# print(winter_solar_time_series)
