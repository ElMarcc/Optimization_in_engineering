import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from read_in import avg_day_dict
from sun_data_series import solar_time_series_dict, wind_dict
from typing import Dict, Any


# define problem
class OptProblem(Problem):
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_ieq_constr=3,
                         xl=[0, 0],
                         xu=[364000, 80000]
                         )
        self.parameters = parameters

    def _evaluate(self, x, out, *args, **kwargs):
        pars = self.parameters
        f = np.zeros((x.shape[0],))
        g1 = np.zeros((x.shape[0],))
        g2 = np.zeros((x.shape[0],))
        g3 = np.zeros((x.shape[0],))
        p_grid = self.get_p_grid(x=x, d=pars['d'], p_solar=pars['p_solar'], p_wind=pars['p_wind'],
                                 time_step=pars['time_step'])
        for i in range(x.shape[0]):
            wind_cost = x[i, 1] * pars['c_w']
            solar_cost = x[i, 0] * pars['c_s']
            grid_power = p_grid[i, :]
            for j in range(48):
                wind_power_generation = np.sum(x[i, 1] * pars['p_wind'][j] * pars['time_step'])
                solar_power_generation = np.sum(x[i, 0] * pars['p_solar'][j])
                percentage = (wind_power_generation + solar_power_generation) / np.sum(
                    pars['d'] * pars['time_step'] / 2)
            grid_power_cost = np.sum(p_grid[i, :]) * pars['c_grid']

            f[i] = solar_cost + wind_cost + grid_power_cost

            g1[i] = np.sum(p_grid[i, :]) - np.sum(pars['d'] * pars['time_step'] / 2)
            # g1[i] = np.sum(pars['d'] * pars['time_step'] / 2) - np.sum(x[i, 0] * pars['p_solar']) - \
            #         np.sum(x[i, 1] * pars['p_wind'] * pars['time_step'])
            g2[i] = 0.25 * x[i, 1] - x[i, 0]
            g3[i] = 0.25 * x[i, 0] - x[i, 1]

        out["F"] = np.column_stack([f])
        out["G"] = np.column_stack([g1, g2, g3])

    def get_p_grid(self, x, d, p_solar, p_wind, time_step) -> np.ndarray:
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
                demand = d[i] * time_step
                pgrid[j, i] = max(0, demand - (solar_energy_generation + wind_energy_generation))
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


def get_parameters(time_period: str):
    time_period_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'summer',
                         'winter']
    if time_period not in time_period_names:
        raise ValueError(f'{time_period} is no valid time period name.')
    parameters = {
        'd': avg_day_dict[f'avg_day_{time_period}'],  # avg day demand
        'c_s': 105,  # costs solar energy [€/MWpeak/day]
        'c_w': 332,  # costs wind energy [€/MWpeak/day]
        'c_grid': 178,  # costs of grid power [€/MWh]
        'time_step': 0.5,  # time_step [h]
        'p_solar': solar_time_series_dict[f'{time_period}_solar_time_series'],  # solar power production [MWh/MWpeak]
        'p_wind': wind_dict[f'{time_period}_wind'],  # wind power production [MW/MWpeak]
    }
    return parameters


pars = get_parameters('aug')

result = minimize_OptProblem(pars)

x = result.X
p_grid = np.zeros((48,))
for i in range(48):
    p_grid[i] = max(0, pars['d'][i] * pars['time_step'] - (
            x[0] * pars['p_solar'][i] + x[1] * pars['p_wind'][i] * pars['time_step']))
part_of_grid = np.sum(p_grid) / np.sum(pars['d'] * pars['time_step'])
part_of_renewables = (np.sum(x[0] * pars['p_solar']) + np.sum(x[1] * pars['p_wind'] * pars['time_step'])) / np.sum(
    pars['d'] * pars['time_step'])
print(f'The percentage of renewables of the whole daily demand is: {100 - part_of_grid * 100}')
print(f'The percentage of the grid of the whole daily demand is: {part_of_grid * 100}')
print(f'The total potential production of renewables is {part_of_renewables * 100}%.')

cost_per_energy_solar = x[0] * pars['c_s'] / np.sum(x[0] * pars['p_solar'])
cost_per_energy_wind = x[1] * pars['c_w'] / np.sum(x[1] * pars['p_wind'] * pars['time_step'])
print(f'The cost [€] per potential energy [MWh] for solar energy is: {cost_per_energy_solar} €/MWh')
print(f'The cost [€] per potential energy [MWh] for wind energy is: {cost_per_energy_wind} €/MWh')

print(result.X)
print(result.F)
