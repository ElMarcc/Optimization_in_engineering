import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from read_in import avg_day_dict
from sun_data_series import solar_time_series_dict,\
    wind_dict,\
    offshore_dict,\
    uct_solar_time_series_dict,\
    uct_wind_dict, \
    uct_offshore_dict, \
    dict_solar_power, \
    sunset_dict, \
    sunrise_dict, \
    get_solar_time_series, \
    get_onshore_time_series, \
    onshore_wind_dict, \
    offshore_wind_dict, \
    get_offshore_time_series
from typing import Dict, Any, Union, List


# define problem
class OptProblem(Problem):
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(n_var=3,
                         n_obj=1,
                         n_ieq_constr=3,
                         xl=[0, 0, 0],
                         xu=[364000, 80000, 140000]
                         )
        self.parameters = parameters

    def _evaluate(self, x, out, *args, **kwargs):
        pars = self.parameters
        f = np.zeros((x.shape[0],))
        g1 = np.zeros((x.shape[0],))
        g2 = np.zeros((x.shape[0],))
        g3 = np.zeros((x.shape[0],))
        p_grid = self.get_p_grid(x=x, d=pars['d'], p_solar=pars['p_solar'], p_wind=pars['p_wind'],
                                 p_offshore=pars['p_offshore'],
                                 time_step=pars['time_step'])
        for i in range(x.shape[0]):
            wind_cost = x[i, 1] * pars['c_w']
            solar_cost = x[i, 0] * pars['c_s']
            offshore_cost = x[i, 2] * pars['c_o']
            grid_power = p_grid[i, :]
            wind_power_generation = np.sum(x[i, 1] * pars['p_wind'] * pars['time_step'])
            solar_power_generation = np.sum(x[i, 0] * pars['p_solar'])
            offshore_power_generation = np.sum(x[i, 2] * pars['p_offshore'])
            percentage = (wind_power_generation + solar_power_generation + offshore_power_generation) / np.sum(
                pars['d'] * pars['time_step'] / 2)
            if wind_power_generation != 0:
                wind_eff = wind_cost / wind_power_generation
            else:
                wind_eff = np.inf
            solar_eff = solar_cost / solar_power_generation
            if offshore_power_generation != 0:
                offshore_eff = offshore_cost / offshore_power_generation
            else:
                offshore_eff = np.inf
            grid_power_cost = np.sum(p_grid[i, :]) * pars['c_grid']
            vars = x[i, :]

            f[i] = solar_cost + wind_cost + offshore_cost + grid_power_cost

            g1[i] = np.sum(p_grid[i, :]) - np.sum(pars['d'] * pars['time_step'] / 2)
            g2[i] = 0.25 * (x[i, 1] + x[i, 2]) - x[i, 0]
            g3[i] = 0.25 * x[i, 0] - (x[i, 1] + x[i, 2])

        out["F"] = np.column_stack([f])
        out["G"] = np.column_stack([g1, g2, g3])

    def get_p_grid(self, x, d, p_solar, p_wind, p_offshore, time_step) -> np.ndarray:
        """
        Calculates p_grid for the given x values, so that the demand is always satisfied and
        p_grid is always positive, since we assume that that we do not supply energy for the grid.
        Returns a numpy array with the following dimensions: [pop_size, 48]
        """
        pgrid = np.zeros((x.shape[0], 48))
        for j in range(x.shape[0]):
            for i in range(48):
                wind_energy_generation = x[j, 1] * p_wind[i] * time_step
                solar_energy_generation = x[j, 0] * p_solar[i]
                offshore_energy_generation = x[j, 2] * p_offshore[i]
                demand = d[i] * time_step
                pgrid[j, i] = max(0, demand - (
                        solar_energy_generation + wind_energy_generation + offshore_energy_generation))
        return pgrid


def minimize_OptProblem(parameters: Dict[str, Any]):
    # initialize algorithm
    algorithm = GA(
        pop_size=100,
        eliminate_duplicates=True)

    # define termination criteria
    termination = get_termination("n_gen", 100)

    problem = OptProblem(parameters)

    # minimize
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        eliminate_duplicates=True,
    )
    return result


def get_parameters(time_period: str, grid_cost: Union[float, int]):
    time_period_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'summer',
                         'winter', 'year']
    if time_period not in time_period_names:
        raise ValueError(f'{time_period} is no valid time period name.')
    parameters = {
        'd': avg_day_dict[f'avg_day_{time_period}'],  # avg day demand
        'c_s': 105,  # costs solar energy [€/MWpeak/day]
        'c_w': 332,  # costs wind energy [€/MWpeak/day]
        'c_o': 682,  # costs offshore wind energy [€/MWpeak/day]
        'c_grid': grid_cost,  # costs of grid power [€/MWh]
        'time_step': 0.5,  # time_step [h]
        'p_solar': solar_time_series_dict[f'{time_period}_solar_time_series'],  # solar power production [MWh/MWpeak]
        'p_wind': wind_dict[f'{time_period}_wind'],  # wind power production [MW/MWpeak]
        'p_offshore': offshore_dict[f'{time_period}_offshore'],
        'uct_p_solar': uct_solar_time_series_dict[f'uct_{time_period}_solar_time_series'],
        'uct_p_wind': uct_wind_dict[f'uct_{time_period}_wind'],
        'uct_p_offshore': uct_offshore_dict[f'uct_{time_period}_offshore'],
    }
    return parameters


def create_df(time_periods: List[str], grid_costs: List[Union[float, int]]):
    data_strs = []
    col_strs = ['F[€/day]', 'x_1[MW]', 'x_2[MW]', 'x_3[MW]', 'g_c[%]', 'r_c[%]', 'r_p[%]', 'P[g1>0][%]', 'P[g2>0][%]',
                'P[g3>0][%]']
    for time_period in time_periods:
        for grid_cost in grid_costs:
            data_strs.append(f'{time_period}_{grid_cost}')
    df = pd.DataFrame(index=data_strs, columns=col_strs)
    return df


def evaluate(time_periods: List[str], grid_costs: List[Union[float, int]]):
    df = create_df(time_periods, grid_costs)
    for idx_tp, time_period in enumerate(time_periods):
        for idx_gc, grid_cost in enumerate(grid_costs):
            pars = get_parameters(time_period, grid_cost)
            result = minimize_OptProblem(pars)
            x = result.X
            # calculating the grid power consumed
            p_grid = np.zeros((48,))
            for i in range(48):
                p_grid[i] = max(0, pars['d'][i] * pars['time_step'] - (
                        x[0] * pars['p_solar'][i] +
                        x[1] * pars['p_wind'][i] * pars['time_step'] +
                        x[2] * pars['p_offshore'][i]))
            # calculating percentages of whole consumption
            part_of_grid = np.sum(p_grid) / np.sum(pars['d'] * pars['time_step'])
            prod_of_renewables = (np.sum(x[0] * pars['p_solar']) + np.sum(
                x[1] * pars['p_wind'] * pars['time_step']) + np.sum(x[2] * pars['p_offshore'])) / np.sum(
                pars['d'] * pars['time_step'])
            # calculating failure probabilities
            p_f = calc_cons_violations(x, pars, time_period) * 100
            row = np.concatenate((
                result.F, x, np.array([part_of_grid * 100, (1 - part_of_grid) * 100, prod_of_renewables * 100]), p_f))
            for idx in range(row.shape[0]):
                if idx <= 3:
                    row[idx] = round(row[idx], 0)
                else:
                    row[idx] = round(row[idx], 2)
            df.iloc[idx_tp * len(grid_costs) + idx_gc] = row
    return df


def calc_cons_violations(x: np.ndarray, pars: Dict[str, Any], time_period: str):
    n = 1
    g1_uct = np.zeros((n,))
    g2_uct = np.zeros((n,))
    g3_uct = np.zeros((n,))
    # collect random distributed data points
    solar_data = np.zeros((48, n))
    onshore_data = np.zeros((48, n))
    offshore_data = np.zeros((48, n))
    # get solar data
    sunrise = sunrise_dict[f'{time_period}_sunrise']
    sunset = sunset_dict[f'{time_period}_sunset']
    solar_power_income = dict_solar_power[f'{time_period}_power_income']
    # get onshore wind data
    onshore_wind_eff = onshore_wind_dict[f'{time_period}_wind']
    # get offshore wind data
    offshore_wind_eff = offshore_wind_dict[f'{time_period}_offshore']
    for i in range(n):
        _, solar_data[:, i] = get_solar_time_series(sunrise, sunset, solar_power_income)
        onshore_data[:, i] = get_onshore_time_series(onshore_wind_eff)
        offshore_data[:, i] = get_offshore_time_series(offshore_wind_eff)
    p_grid = np.zeros((48, n))
    for j in range(48):
        p_grid[j, :] = pars['d'][j] * pars['time_step'] - (
                x[0] * solar_data[j, :] +
                x[1] * onshore_data[j, :] * pars['time_step'] +
                x[2] * offshore_data[j, :])
    for k in range(n):
        g1_uct[k] = np.sum(p_grid[:, k]) - np.sum(pars['d'] * pars['time_step'] / 2)
    g2_uct[:] = 0.25 * (x[1] + x[2]) - x[0]
    g3_uct[:] = 0.25 * x[0] - (x[1] + x[2])
    p_f1 = (g1_uct > 0).sum() / n
    p_f2 = (g2_uct > 0).sum() / n
    p_f3 = (g3_uct > 0).sum() / n
    return np.array([p_f1, p_f2, p_f3])



time_periods = [
    'year',
    'summer',
    'winter',
    'jan',
    'aug',
]
grid_costs = [
    50,
    100,
    150,
    200,
    250,
]

df = evaluate(time_periods, grid_costs)

print(df.to_string())

pars = get_parameters('year', 178)

# result = minimize_OptProblem(pars)

# x = result.X
# p_grid = np.zeros((48,))
# for i in range(48):
#     p_grid[i] = max(0, pars['d'][i] * pars['time_step'] - (
#             x[0] * pars['p_solar'][i] + x[1] * pars['p_wind'][i] * pars['time_step'] + x[2] * pars['p_offshore'][i]))
# part_of_grid = np.sum(p_grid) / np.sum(pars['d'] * pars['time_step'])
# part_of_renewables = (np.sum(x[0] * pars['p_solar']) + np.sum(x[1] * pars['p_wind'] * pars['time_step']) + np.sum(
#     x[2] * pars['p_offshore'])) / np.sum(pars['d'] * pars['time_step'])
# print(f'The percentage of renewables of the whole daily demand is: {100 - part_of_grid * 100}')
# print(f'The percentage of the grid of the whole daily demand is: {part_of_grid * 100}')
# print(f'The total potential production of renewables is {part_of_renewables * 100}%.')

cost_per_energy_solar = pars['c_s'] / np.sum(pars['p_solar'])
cost_per_energy_wind = pars['c_w'] / np.sum(pars['p_wind'] * pars['time_step'])
cost_per_energy_offshore = pars['c_o'] / np.sum(pars['p_offshore'])
print(f'The cost [€] per potential energy [MWh] for solar energy is: {cost_per_energy_solar} €/MWh')
print(f'The cost [€] per potential energy [MWh] for wind energy is: {cost_per_energy_wind} €/MWh')
print(f'The cost [€] per potential energy [MWh] for offshore energy is: {cost_per_energy_offshore} €/MWh')

# print(result.X)
# print(result.F)

save = 0

if save:
    with pd.ExcelWriter('result_save.xlsx') as writer:
        df.to_excel(writer, sheet_name='results')
