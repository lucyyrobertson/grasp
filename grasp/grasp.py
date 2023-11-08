from pathlib import Path
from typing import Union
import random

import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


class Grasp:

    def __init__(self, actual: list,
                 perceived: list,
                 subject_id: Union[str, None] = None,
                 condition: Union[str, None] = None,
                 fig_path: Union[str, None] = None,
                 show_plot: bool = False):
        self.actual = actual
        self.perceived = perceived
        self.r_squared = None
        self.mean_abs_error = None
        self.intercept = None
        self.slope = None
        self.subject_id = subject_id
        self.condition = condition
        if fig_path:
            self.fig_path = Path(fig_path)
        else:
            self.fig_path = fig_path
        self.show_plot = show_plot

        self._regression()
        self._mean_abs_error()
        self._plot()

    def _regression(self):
        actual = sm.add_constant(self.actual)
        model = sm.OLS(self.perceived, actual).fit()
        self.r_squared = model.rsquared
        self.intercept, self.slope = model.params

    def _mean_abs_error(self):
        abs_error = list()
        for actual_value, perceived_value in zip(self.actual, self.perceived):
            abs_error.append(abs(perceived_value - actual_value))
        self.mean_abs_error = np.mean(abs_error)

    def _plot(self):
        plt.figure(figsize=(4, 4))

        for actual_value, perceived_value in zip(self.actual, self.perceived):
            jit = (random.randrange(-20, 20, 1)) / 100
            plt.plot(actual_value+jit, perceived_value, 'ok', alpha=0.5)
        x1 = np.min(self.actual)
        x2 = np.max(self.actual)
        y1 = (self.slope * x1) + self.intercept
        y2 = (self.slope * x2) + self.intercept
        plt.plot([x1, x2], [y1, y2], '-k')
        plt.ylabel('Perceived width (cm)')
        plt.xlabel('Actual width (cm)')
        plt.grid()
        plt.plot([x1, x2], [x1, x2], '--', color='lightgrey')
        plt.tight_layout()
        plt.title(f'{self.subject_id},: {self.condition}')
        if self.fig_path:
            figure_path = self.fig_path / (self.subject_id + '_' + self.condition + '.png')
            plt.savefig(figure_path, dpi=180)
            plt.close()
        if self.show_plot:
            plt.show()
