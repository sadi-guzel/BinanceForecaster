import random
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import os
import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from LazyProphet import LazyProphet as lp

from credentials import api_key, api_secret

mpl.rcParams['figure.figsize'] = 30, 10

if not os.path.exists('out/'):
    os.mkdir('out/')

client = Client(api_key, api_secret)
# currencies = [{'asset': 'BTC'}]
currencies = client.get_account(timestamp=datetime.now().timestamp())['balances']
for currency in currencies:
    print(currency)
    start = time.time()
    try:
        bars = client.get_historical_klines(f'{currency["asset"]}USDT', '1m',
                                            start_str=str(datetime.today() + timedelta(days=-7)),
                                            end_str=str(datetime.today()))
        for line in bars:
            del line[5:]
        df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close'])
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        # df['date'] = pd.to_datetime(df['date'].astype(str)) + pd.DateOffset(hours=3)
        df['date'] = pd.to_datetime(df['date'], format='YYYY-MM-DD HH:MM:SS')
        df['close'] = df['close'].astype(float)
        # df['close'] = df['close'].rolling(window=4, min_periods=1).mean()

        df.sort_values(by=['date'], inplace=True, ascending=True)
        df.reset_index(drop=True, inplace=True)

        y = df['close'].values
        n_future = 60 * 3   # next 3 hours
        plt.plot(y[-n_future:])
        legends = [currency['asset']]

        model = lp.LazyProphet(n_basis=10,  # weighted piecewise basis functions
                               seasonal_period=[24 * 60, 24*60*7],
                               boosting_params={'verbosity': 2, 'force_col_wise': True},
                               fourier_order=500,
                               decay=.99
                               # the 'penalized' in penalized weighted piecewise linear basis functions
                               )
        fitted = model.fit(y)
        predicted = model.predict(n_future)
        plt.plot(np.append(fitted[-n_future:], predicted))
        legends.append(f'PRED')
        plt.plot(fitted[-n_future:])
        legends.append(f'FIT')

        # for alpha in [0.05, 0.1, 0.25, 0.75, 0.9, 0.95]:
        # for alpha in [0.75, 0.95]:
        #     model = lp.LazyProphet(n_basis=4,  # weighted piecewise basis functions
        #                            seasonal_period=[60, 12 * 60, 24 * 60],
        #                            boosting_params={'verbosity': 2, 'objective': 'quantile', 'alpha': alpha, 'force_col_wise': True},
        #                            fourier_order=500,
        #                            decay=.99
        #                            # the 'penalized' in penalized weighted piecewise linear basis functions
        #                            )
        #     fitted = model.fit(y)
        #     predicted = model.predict(n_future)
        #     with mpl.rc_context({'lines.linewidth': 2}):
        #         plt.plot(np.append(fitted[-n_future:], predicted))
        #     legends.append(f'PRED_{alpha}')
        #     with mpl.rc_context({'lines.linewidth': 2}):
        #         plt.plot(fitted[-n_future:])
        #     legends.append(f'FIT_{alpha}')
        plt.legend(legends)
        plt.grid()
        plt.axvline(n_future)
        plt.tight_layout()
        plt.savefig(f"out/{currency['asset']}.png", dpi=100)
        plt.close()

        print(f'elapsed time: {time.time() - start} seconds')

    except Exception as e:
        print(str(e))
