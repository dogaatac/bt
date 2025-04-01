# plotter.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import timedelta
import random

def plot_trades(df, trades, config):
    # Rastgele 10 işlem seç
    if len(trades) > config.PLOT_NUM_TRADES:
        selected_trades = random.sample(trades, config.PLOT_NUM_TRADES)
    else:
        selected_trades = trades

    # Grafik çizimi
    for idx, trade in enumerate(selected_trades):
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        sweep_time = trade['sweep_time']
        trade_type = trade['type']
        pivot_price = trade['pivot_price']
        sweep_level = trade['sweep_low'] if trade_type == 'long' else trade['sweep_high']
        manip_low = trade.get('manip_low', sweep_level)
        manip_high = trade.get('manip_high', sweep_level)
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        sl = trade['sl']
        tp = trade['tp']

        # Grafik için zaman aralığı
        start_time = sweep_time - timedelta(minutes=15 * config.PLOT_CANDLES_BEFORE)
        end_time = exit_time + timedelta(minutes=15 * config.PLOT_CANDLES_AFTER)
        df_plot = df.loc[start_time:end_time].copy()

        # Ek çizimler
        apds = []
        # Pivot çizgisi
        pivot_series = pd.Series(pivot_price, index=df_plot.index)
        apds.append(mpf.make_addplot(pivot_series, linestyle='--', color='blue', label=f'Pivot: {pivot_price:.2f}'))
        # Sweep noktası
        sweep_series = pd.Series(np.nan, index=df_plot.index)
        sweep_series.loc[sweep_time] = sweep_level
        apds.append(mpf.make_addplot(sweep_series, scatter=True, marker='o', color='purple', label=f'Sweep: {sweep_level:.2f}'))
        # Giriş noktası
        entry_series = pd.Series(np.nan, index=df_plot.index)
        entry_series.loc[entry_time] = entry_price
        apds.append(mpf.make_addplot(entry_series, scatter=True, marker='^', color='green', label=f'Giriş: {entry_price:.2f}'))
        # Çıkış noktası
        exit_series = pd.Series(np.nan, index=df_plot.index)
        exit_series.loc[exit_time] = exit_price
        apds.append(mpf.make_addplot(exit_series, scatter=True, marker='v', color='red' if trade['profit'] < 0 else 'green', label=f'Çıkış: {exit_price:.2f}'))
        # SL çizgisi
        sl_series = pd.Series(sl, index=df_plot.index)
        apds.append(mpf.make_addplot(sl_series, linestyle='--', color='darkred', label=f'SL: {sl:.2f}'))
        # TP çizgisi
        tp_series = pd.Series(tp, index=df_plot.index)
        apds.append(mpf.make_addplot(tp_series, linestyle='--', color='darkgreen', label=f'TP: {tp:.2f}'))
        # Manipülasyon sürecindeki en uç/en dip noktayı göster
        manip_extreme_series = pd.Series(np.nan, index=df_plot.index)
        manip_extreme = manip_low if trade_type == 'long' else manip_high
        manip_extreme_series.loc[sweep_time] = manip_extreme
        apds.append(mpf.make_addplot(manip_extreme_series, scatter=True, marker='x', color='orange', label=f'Manip Extreme: {manip_extreme:.2f}'))

        # Grafiği çiz
        mpf.plot(df_plot, type='candle', style='yahoo', title=f'İşlem {idx + 1}: {trade_type.capitalize()} (Kâr/Zarar: {trade["profit"]:.2f} USD)',
                 ylabel='Fiyat (USDT)', addplot=apds, figscale=config.PLOT_FIGSCALE, savefig=f'trade_{idx + 1}_chart.png')