import numpy as np
import math

np.random.seed(1)
uncertainty = 0

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
year_sunrise = 7.5

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
year_sunset = 19.5

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
year_power_income = 4.1

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
year_wind = 4478.2 / max_cap

if uncertainty:
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
    year_wind_series = np.random.normal(year_wind, year_wind / 3, 48).clip(min=0)
else:
    jan_wind_series = np.ones((48,)) * jan_wind
    feb_wind_series = np.ones((48,)) * feb_wind
    mar_wind_series = np.ones((48,)) * mar_wind
    apr_wind_series = np.ones((48,)) * apr_wind
    may_wind_series = np.ones((48,)) * may_wind
    jun_wind_series = np.ones((48,)) * jun_wind
    jul_wind_series = np.ones((48,)) * jul_wind
    aug_wind_series = np.ones((48,)) * aug_wind
    sep_wind_series = np.ones((48,)) * sep_wind
    oct_wind_series = np.ones((48,)) * oct_wind
    nov_wind_series = np.ones((48,)) * nov_wind
    dec_wind_series = np.ones((48,)) * dec_wind
    summer_wind_series = np.ones((48,)) * summer_wind
    winter_wind_series = np.ones((48,)) * winter_wind
    year_wind_series = np.ones((48,)) * year_wind

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
    'year_wind': year_wind_series,
}

winter_offshore = 15550000 / 365 / 48 / 7500
year_offshore = 26000000 / 365 / 48 / 7500
summer_offshore = 10450000 / 365 / 48 / 7500
jan_offshore = 2550000 / 365 / 48 / 7500
feb_offshore = 2200000 / 365 / 48 / 7500
mar_offshore = 2800000 / 365 / 48 / 7500
apr_offshore = 1800000 / 365 / 48 / 7500
may_offshore = 1900000 / 365 / 48 / 7500
jun_offshore = 1500000 / 365 / 48 / 7500
jul_offshore = 1700000 / 365 / 48 / 7500
aug_offshore = 1400000 / 365 / 48 / 7500
sep_offshore = 2150000 / 365 / 48 / 7500
oct_offshore = 2400000 / 365 / 48 / 7500
nov_offshore = 2300000 / 365 / 48 / 7500
dec_offshore = 3300000 / 365 / 48 / 7500

if uncertainty:
    jan_offshore_wind_series = np.random.normal(jan_offshore, jan_offshore / 2, 48).clip(min=0)
    feb_offshore_wind_series = np.random.normal(feb_offshore, feb_offshore / 2, 48).clip(min=0)
    mar_offshore_wind_series = np.random.normal(mar_offshore, mar_offshore / 2, 48).clip(min=0)
    apr_offshore_wind_series = np.random.normal(apr_offshore, apr_offshore / 2, 48).clip(min=0)
    may_offshore_wind_series = np.random.normal(may_offshore, may_offshore / 2, 48).clip(min=0)
    jun_offshore_wind_series = np.random.normal(jun_offshore, jun_offshore / 2, 48).clip(min=0)
    jul_offshore_wind_series = np.random.normal(jul_offshore, jul_offshore / 2, 48).clip(min=0)
    aug_offshore_wind_series = np.random.normal(aug_offshore, aug_offshore / 2, 48).clip(min=0)
    sep_offshore_wind_series = np.random.normal(sep_offshore, sep_offshore / 2, 48).clip(min=0)
    oct_offshore_wind_series = np.random.normal(oct_offshore, oct_offshore / 2, 48).clip(min=0)
    nov_offshore_wind_series = np.random.normal(nov_offshore, nov_offshore / 2, 48).clip(min=0)
    dec_offshore_wind_series = np.random.normal(dec_offshore, dec_offshore / 2, 48).clip(min=0)
    summer_offshore_wind_series = np.random.normal(winter_offshore, winter_offshore / 2, 48).clip(min=0)
    winter_offshore_wind_series = np.random.normal(winter_offshore, winter_offshore / 2, 48).clip(min=0)
    year_offshore_wind_series = np.random.normal(year_offshore, year_offshore / 2, 48).clip(min=0)
else:
    jan_offshore_wind_series = np.ones((48,)) * jan_offshore
    feb_offshore_wind_series = np.ones((48,)) * feb_offshore
    mar_offshore_wind_series = np.ones((48,)) * mar_offshore
    apr_offshore_wind_series = np.ones((48,)) * apr_offshore
    may_offshore_wind_series = np.ones((48,)) * may_offshore
    jun_offshore_wind_series = np.ones((48,)) * jun_offshore
    jul_offshore_wind_series = np.ones((48,)) * jul_offshore
    aug_offshore_wind_series = np.ones((48,)) * aug_offshore
    sep_offshore_wind_series = np.ones((48,)) * sep_offshore
    oct_offshore_wind_series = np.ones((48,)) * oct_offshore
    nov_offshore_wind_series = np.ones((48,)) * nov_offshore
    dec_offshore_wind_series = np.ones((48,)) * dec_offshore
    summer_offshore_wind_series = np.ones((48,)) * winter_offshore
    winter_offshore_wind_series = np.ones((48,)) * winter_offshore
    year_offshore_wind_series = np.ones((48,)) * year_offshore

offshore_dict = {
    'jan_offshore': jan_offshore_wind_series,
    'feb_offshore': feb_offshore_wind_series,
    'mar_offshore': mar_offshore_wind_series,
    'apr_offshore': apr_offshore_wind_series,
    'may_offshore': may_offshore_wind_series,
    'jun_offshore': jun_offshore_wind_series,
    'jul_offshore': jul_offshore_wind_series,
    'aug_offshore': aug_offshore_wind_series,
    'sep_offshore': sep_offshore_wind_series,
    'oct_offshore': oct_offshore_wind_series,
    'nov_offshore': nov_offshore_wind_series,
    'dec_offshore': dec_offshore_wind_series,
    'summer_offshore': summer_offshore_wind_series,
    'winter_offshore': winter_offshore_wind_series,
    'year_offshore': year_offshore_wind_series,
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
            if uncertainty:
                time_series[idx] = np.random.normal(power_production, power_production / 2, 1).clip(min=0)
            else:
                time_series[idx] = power_production

    w = sum(time_series)
    # print(w)
    return time_series


jan_solar_time_series = get_solar_time_series(sunrise=jan_sunrise, sunset=jan_sunset, power_income=jan_power_income)
feb_solar_time_series = get_solar_time_series(sunrise=feb_sunrise, sunset=feb_sunset, power_income=feb_power_income)
mar_solar_time_series = get_solar_time_series(sunrise=mar_sunrise, sunset=mar_sunset, power_income=mar_power_income)
apr_solar_time_series = get_solar_time_series(sunrise=apr_sunrise, sunset=apr_sunset, power_income=apr_power_income)
may_solar_time_series = get_solar_time_series(sunrise=may_sunrise, sunset=may_sunset, power_income=may_power_income)
jun_solar_time_series = get_solar_time_series(sunrise=jun_sunrise, sunset=jun_sunset, power_income=jun_power_income)
jul_solar_time_series = get_solar_time_series(sunrise=jul_sunrise, sunset=jul_sunset, power_income=jul_power_income)
aug_solar_time_series = get_solar_time_series(sunrise=aug_sunrise, sunset=aug_sunset, power_income=aug_power_income)
sep_solar_time_series = get_solar_time_series(sunrise=sep_sunrise, sunset=sep_sunset, power_income=sep_power_income)
oct_solar_time_series = get_solar_time_series(sunrise=oct_sunrise, sunset=oct_sunset, power_income=oct_power_income)
nov_solar_time_series = get_solar_time_series(sunrise=nov_sunrise, sunset=nov_sunset, power_income=nov_power_income)
dec_solar_time_series = get_solar_time_series(sunrise=dec_sunrise, sunset=dec_sunset, power_income=dec_power_income)
summer_solar_time_series = get_solar_time_series(sunrise=summer_sunrise, sunset=summer_sunset,
                                                 power_income=summer_power_income)
winter_solar_time_series = get_solar_time_series(sunrise=winter_sunrise, sunset=winter_sunset,
                                                 power_income=winter_power_income)
year_solar_time_series = get_solar_time_series(sunrise=year_sunrise, sunset=year_sunset, power_income=year_power_income)

solar_time_series_dict = {
    'jan_solar_time_series': jan_solar_time_series,
    'feb_solar_time_series': feb_solar_time_series,
    'mar_solar_time_series': mar_solar_time_series,
    'apr_solar_time_series': apr_solar_time_series,
    'may_solar_time_series': may_solar_time_series,
    'jun_solar_time_series': jun_solar_time_series,
    'jul_solar_time_series': jul_solar_time_series,
    'aug_solar_time_series': aug_solar_time_series,
    'sep_solar_time_series': sep_solar_time_series,
    'oct_solar_time_series': oct_solar_time_series,
    'nov_solar_time_series': nov_solar_time_series,
    'dec_solar_time_series': dec_solar_time_series,
    'summer_solar_time_series': summer_solar_time_series,
    'winter_solar_time_series': winter_solar_time_series,
    'year_solar_time_series': year_solar_time_series,
}

# print(jan_solar_time_series)
# print(aug_solar_time_series)
# print(summer_solar_time_series)
# print(winter_solar_time_series)
