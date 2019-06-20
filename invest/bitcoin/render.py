import numpy as np
import matplotlib 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from datetime import datetime

class BitcoinTradingGraph:
    def __init__(self, df, title=None):
        self.df = df
        self.net_worths = np.zeros(len(df['Timestamp']))
        self.fig = plt.figure()
        # self.fig.suptitle(title)

        self.net_worth_ax = plt.subplot2grid((6, 1), (0, 0), rowspan=2, colspan=1)

        self.price_ax = plt.subplot2grid((6, 1), (2, 0), rowspan=8, colspan=1, sharex=self.net_worth_ax)

        self.volume_ax = self.price_ax.twinx()

        plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top = 0.90, wspace=0.2, hspace=0)
        plt.show(block=False)

    def render(self, current_step, net_worths, trades, window_size=200):
        net_worth = round(net_worths[-1], 2)
        initial_net_worth = round(net_worths[0], 2)
        profit_percent = round(
            (net_worth - initial_net_worth) / initial_net_worth * 100, 2)
        self.fig.suptitle(
            'Net worth: $' + str(net_worth) + ' | Profit: ' + str(profit_percent) + '%')
        window_start = max(current_step - window_size, 0)
        step_range = slice(window_start, current_step + 1)
        dates = self.df['Time'].values[step_range]

        data_labels = np.array([datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M') 
                               for x in self.df['Timestamp'].values[step_range]])

