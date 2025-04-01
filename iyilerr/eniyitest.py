import pandas as pd
import numpy as np
from colorama import init, Fore, Style
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import timedelta
import random

# Colorama’yı başlat
init()

# Veriyi yükle
df = pd.read_csv("btc_usdt_15m.csv")
df['open_time'] = pd.to_datetime(df['open_time'])
df.set_index('open_time', inplace=True)

# Pivot high ve low hesaplama fonksiyonları
def pivot_high(series, left, right):
    pivots = []
    for i in range(left, len(series) - right):
        if all(series.iloc[i] > series.iloc[i - left:i]) and all(series.iloc[i] > series.iloc[i + 1:i + right + 1]):
            pivots.append((i, series.iloc[i]))
    return pivots

def pivot_low(series, left, right):
    pivots = []
    for i in range(left, len(series) - right):
        if all(series.iloc[i] < series.iloc[i - left:i]) and all(series.iloc[i] < series.iloc[i + 1:i + right + 1]):
            pivots.append((i, series.iloc[i]))
    return pivots

# Parametreler
left = 5  # Sol periyot
right = 5  # Sağ periyot
manipulation_threshold = 0.003  # %0.5 manipülasyon eşiği
max_candles = 30  # Maksimum 15 mum
consecutive_candles = 4  # En az 2 ardışık mum
min_candles_for_second_condition = 10  # İkinci koşul için minimum 5 mum
max_candles_for_second_condition = 25  # İkinci koşul için maksimum 15 mum
risk_reward_ratio = 1.5  # Risk-Kazanç oranı (örneğin 1:1.5)

# Pivot high ve low’ları hesapla
ph = pivot_high(df['high'], left, right)
pl = pivot_low(df['low'], left, right)

# Başlangıç parametreleri
balance = 10000  # USD
max_risk = 100  # USD
positions = []
trades = []
used_pivots = set()  # Kullanılan pivotları takip et

# Pivotları sözlüğe çevir
ph_dict = {idx: price for idx, price in ph}
pl_dict = {idx: price for idx, price in pl}

# Sweep ve manipülasyon takibi için listeler
sweeps_pl = []  # Buy side sweep’ler: (pivot_idx, pivot_price, sweep_low, sweep_idx, manip_low, manip_high)
sweeps_ph = []  # Sell side sweep’ler: (pivot_idx, pivot_price, sweep_high, sweep_idx, manip_low, manip_high)

# Her bar için döngü
for i in range((left + right), len(df)):
    current_high = df['high'].iloc[i]
    current_low = df['low'].iloc[i]
    current_close = df['close'].iloc[i]

    # Aktif pivotları kontrol et (son 200 bar)
    active_ph = {k: v for k, v in ph_dict.items() if k > i - 200 and k < i}
    active_pl = {k: v for k, v in pl_dict.items() if k > i - 200 and k < i}

    # Sell side sweep kontrolü (fiyat pivot high’ı aşarsa)
    for ph_idx, ph_price in active_ph.items():
        if ph_idx in used_pivots:
            continue
        if current_high > ph_price:
            manipulation_ratio = (current_high - ph_price) / ph_price
            if manipulation_ratio >= manipulation_threshold:
                sweeps_ph.append((ph_idx, ph_price, current_high, i, current_low, current_high))
                used_pivots.add(ph_idx)

    # Buy side sweep kontrolü (fiyat pivot low’un altına inerse)
    for pl_idx, pl_price in active_pl.items():
        if pl_idx in used_pivots:
            continue
        if current_low < pl_price:
            manipulation_ratio = (pl_price - current_low) / pl_price
            if manipulation_ratio >= manipulation_threshold:
                sweeps_pl.append((pl_idx, pl_price, current_low, i, current_low, current_high))
                used_pivots.add(pl_idx)

    # Long işlem için geri dönüş kontrolü (buy side sweep sonrası)
    for sweep in sweeps_pl[:]:
        pl_idx, pl_price, sweep_low, sweep_idx, manip_low, manip_high = sweep
        bars_since_sweep = i - sweep_idx
        if bars_since_sweep > max_candles:
            sweeps_pl.remove(sweep)
            continue

        # Fiyat pivot low’un üstüne dönene kadar manipülasyon devam eder
        if current_close <= pl_price:
            # Manipülasyon sürecindeki en düşük ve en yüksek seviyeleri güncelle
            manip_low = min(manip_low, current_low)
            manip_high = max(manip_high, current_high)
            sweeps_pl[sweeps_pl.index(sweep)] = (pl_idx, pl_price, sweep_low, sweep_idx, manip_low, manip_high)

        # Birinci Koşul: En az 2 ardışık mum pivot low’un üstünde kapanırsa
        if bars_since_sweep >= consecutive_candles:
            closes_above = all(df['close'].iloc[i - j] > pl_price for j in range(consecutive_candles))
            if closes_above:
                if i + 1 < len(df):
                    entry_price = df['open'].iloc[i + 1]
                    sl_price = manip_low  # SL, manipülasyon sürecindeki en düşük seviyeye konur
                    sl_distance = entry_price - sl_price
                    if sl_distance > 0:
                        position_size = max_risk / sl_distance
                        tp_price = entry_price + sl_distance * risk_reward_ratio  # TP, RR oranına göre hesaplanır
                        positions.append({
                            'type': 'long',
                            'entry_time': df.index[i + 1],
                            'entry_price': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'size': position_size,
                            'pivot_price': pl_price,
                            'sweep_low': sweep_low,
                            'sweep_time': df.index[sweep_idx],
                            'manip_low': manip_low,
                            'manip_high': manip_high
                        })
                        sweeps_pl.remove(sweep)
                        break

        # İkinci Koşul: Fiyat pivot low’un altında 5-15 mum kapanış yapıp geri dönerse
        if bars_since_sweep >= min_candles_for_second_condition:
            closes_below = all(df['close'].iloc[i - j] < pl_price for j in range(min_candles_for_second_condition, min(bars_since_sweep + 1, max_candles_for_second_condition + 1)))
            if closes_below and current_close > pl_price:
                if i + 1 < len(df):
                    entry_price = df['open'].iloc[i + 1]
                    sl_price = manip_low  # SL, manipülasyon sürecindeki en düşük seviyeye konur
                    sl_distance = entry_price - sl_price
                    if sl_distance > 0:
                        position_size = max_risk / sl_distance
                        tp_price = entry_price + sl_distance * risk_reward_ratio  # TP, RR oranına göre hesaplanır
                        positions.append({
                            'type': 'long',
                            'entry_time': df.index[i + 1],
                            'entry_price': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'size': position_size,
                            'pivot_price': pl_price,
                            'sweep_low': sweep_low,
                            'sweep_time': df.index[sweep_idx],
                            'manip_low': manip_low,
                            'manip_high': manip_high
                        })
                        sweeps_pl.remove(sweep)
                        break

    # Short işlem için geri dönüş kontrolü (sell side sweep sonrası)
    for sweep in sweeps_ph[:]:
        ph_idx, ph_price, sweep_high, sweep_idx, manip_low, manip_high = sweep
        bars_since_sweep = i - sweep_idx
        if bars_since_sweep > max_candles:
            sweeps_ph.remove(sweep)
            continue

        # Fiyat pivot high’ın altına dönene kadar manipülasyon devam eder
        if current_close >= ph_price:
            # Manipülasyon sürecindeki en düşük ve en yüksek seviyeleri güncelle
            manip_low = min(manip_low, current_low)
            manip_high = max(manip_high, current_high)
            sweeps_ph[sweeps_ph.index(sweep)] = (ph_idx, ph_price, sweep_high, sweep_idx, manip_low, manip_high)

        # Birinci Koşul: En az 2 ardışık mum pivot high’ın altında kapanırsa
        if bars_since_sweep >= consecutive_candles:
            closes_below = all(df['close'].iloc[i - j] < ph_price for j in range(consecutive_candles))
            if closes_below:
                if i + 1 < len(df):
                    entry_price = df['open'].iloc[i + 1]
                    sl_price = manip_high  # SL, manipülasyon sürecindeki en yüksek seviyeye konur
                    sl_distance = sl_price - entry_price
                    if sl_distance > 0:
                        position_size = max_risk / sl_distance
                        tp_price = entry_price - sl_distance * risk_reward_ratio  # TP, RR oranına göre hesaplanır
                        positions.append({
                            'type': 'short',
                            'entry_time': df.index[i + 1],
                            'entry_price': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'size': position_size,
                            'pivot_price': ph_price,
                            'sweep_high': sweep_high,
                            'sweep_time': df.index[sweep_idx],
                            'manip_low': manip_low,
                            'manip_high': manip_high
                        })
                        sweeps_ph.remove(sweep)
                        break

        # İkinci Koşul: Fiyat pivot high’ın üstünde 5-15 mum kapanış yapıp geri dönerse
        if bars_since_sweep >= min_candles_for_second_condition:
            closes_above = all(df['close'].iloc[i - j] > ph_price for j in range(min_candles_for_second_condition, min(bars_since_sweep + 1, max_candles_for_second_condition + 1)))
            if closes_above and current_close < ph_price:
                if i + 1 < len(df):
                    entry_price = df['open'].iloc[i + 1]
                    sl_price = manip_high  # SL, manipülasyon sürecindeki en yüksek seviyeye konur
                    sl_distance = sl_price - entry_price
                    if sl_distance > 0:
                        position_size = max_risk / sl_distance
                        tp_price = entry_price - sl_distance * risk_reward_ratio  # TP, RR oranına göre hesaplanır
                        positions.append({
                            'type': 'short',
                            'entry_time': df.index[i + 1],
                            'entry_price': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'size': position_size,
                            'pivot_price': ph_price,
                            'sweep_high': sweep_high,
                            'sweep_time': df.index[sweep_idx],
                            'manip_low': manip_low,
                            'manip_high': manip_high
                        })
                        sweeps_ph.remove(sweep)
                        break

    # Açık pozisyonları kontrol et
    for pos in positions[:]:
        if pos['type'] == 'long':
            if current_low <= pos['sl']:
                profit = -max_risk
                balance += profit
                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': df.index[i],
                    'type': pos['type'],
                    'entry_price': pos['entry_price'],
                    'exit_price': pos['sl'],
                    'size': pos['size'],
                    'sl': pos['sl'],
                    'tp': pos['tp'],
                    'profit': profit,
                    'pivot_price': pos['pivot_price'],
                    'sweep_low': pos['sweep_low'],
                    'sweep_time': pos['sweep_time'],
                    'manip_low': pos['manip_low'],
                    'manip_high': pos['manip_high']
                })
                positions.remove(pos)
            elif current_high >= pos['tp']:
                profit = max_risk * risk_reward_ratio
                balance += profit
                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': df.index[i],
                    'type': pos['type'],
                    'entry_price': pos['entry_price'],
                    'exit_price': pos['tp'],
                    'size': pos['size'],
                    'sl': pos['sl'],
                    'tp': pos['tp'],
                    'profit': profit,
                    'pivot_price': pos['pivot_price'],
                    'sweep_low': pos['sweep_low'],
                    'sweep_time': pos['sweep_time'],
                    'manip_low': pos['manip_low'],
                    'manip_high': pos['manip_high']
                })
                positions.remove(pos)
        elif pos['type'] == 'short':
            if current_high >= pos['sl']:
                profit = -max_risk
                balance += profit
                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': df.index[i],
                    'type': pos['type'],
                    'entry_price': pos['entry_price'],
                    'exit_price': pos['sl'],
                    'size': pos['size'],
                    'sl': pos['sl'],
                    'tp': pos['tp'],
                    'profit': profit,
                    'pivot_price': pos['pivot_price'],
                    'sweep_high': pos['sweep_high'],
                    'sweep_time': pos['sweep_time'],
                    'manip_low': pos['manip_low'],
                    'manip_high': pos['manip_high']
                })
                positions.remove(pos)
            elif current_low <= pos['tp']:
                profit = max_risk * risk_reward_ratio
                balance += profit
                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': df.index[i],
                    'type': pos['type'],
                    'entry_price': pos['entry_price'],
                    'exit_price': pos['tp'],
                    'size': pos['size'],
                    'sl': pos['sl'],
                    'tp': pos['tp'],
                    'profit': profit,
                    'pivot_price': pos['pivot_price'],
                    'sweep_high': pos['sweep_high'],
                    'sweep_time': pos['sweep_time'],
                    'manip_low': pos['manip_low'],
                    'manip_high': pos['manip_high']
                })
                positions.remove(pos)

# İşlem detaylarını yazdır
print("\n" + Fore.CYAN + "İşlem Detayları:" + Style.RESET_ALL)
for trade in trades:
    if trade['type'] == 'long':
        print(f"{Fore.GREEN}Long İşlem{Style.RESET_ALL} - "
              f"Entry Time: {trade['entry_time']}, "
              f"Exit Time: {trade['exit_time']}, "
              f"Pivot Low: {trade['pivot_price']:.2f}, "
              f"Sweep Low: {trade['sweep_low']:.2f}, "
              f"Manip Low: {trade['manip_low']:.2f}, "
              f"Entry: {trade['entry_price']:.2f}, "
              f"SL: {trade['sl']:.2f}, "
              f"TP: {trade['tp']:.2f}, "
              f"Profit: {trade['profit']:.2f} USD")
    elif trade['type'] == 'short':
        print(f"{Fore.RED}Short İşlem{Style.RESET_ALL} - "
              f"Entry Time: {trade['entry_time']}, "
              f"Exit Time: {trade['exit_time']}, "
              f"Pivot High: {trade['pivot_price']:.2f}, "
              f"Sweep High: {trade['sweep_high']:.2f}, "
              f"Manip High: {trade['manip_high']:.2f}, "
              f"Entry: {trade['entry_price']:.2f}, "
              f"SL: {trade['sl']:.2f}, "
              f"TP: {trade['tp']:.2f}, "
              f"Profit: {trade['profit']:.2f} USD")

# Sonuçları yazdır
print(f"\n{Fore.CYAN}Başlangıç Bakiyesi: 10000 USD{Style.RESET_ALL}")
print(f"{Fore.CYAN}Son Bakiye: {balance:.2f} USD{Style.RESET_ALL}")
print(f"{Fore.CYAN}Toplam İşlem Sayısı: {len(trades)}{Style.RESET_ALL}")
profitable_trades = sum(1 for trade in trades if trade['profit'] > 0)
losing_trades = sum(1 for trade in trades if trade['profit'] < 0)
total_profit = sum(trade['profit'] for trade in trades)
print(f"{Fore.CYAN}Karlı İşlem Sayısı: {profitable_trades}{Style.RESET_ALL}")
print(f"{Fore.CYAN}Zararlı İşlem Sayısı: {losing_trades}{Style.RESET_ALL}")
print(f"{Fore.CYAN}Toplam Kâr/Zarar: {total_profit:.2f} USD{Style.RESET_ALL}")

# Rastgele 10 işlem seç
if len(trades) > 10:
    selected_trades = random.sample(trades, 10)
else:
    selected_trades = trades

# Grafik çizimi (rastgele 10 işlem için)
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
    start_time = sweep_time - timedelta(minutes=15 * 10)
    end_time = exit_time + timedelta(minutes=15 * 5)
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
    manip_extreme_series.loc[sweep_time] = manip_extreme  # Görselleştirme için sweep zamanında gösteriyoruz
    apds.append(mpf.make_addplot(manip_extreme_series, scatter=True, marker='x', color='orange', label=f'Manip Extreme: {manip_extreme:.2f}'))

    # Grafiği çiz
    mpf.plot(df_plot, type='candle', style='yahoo', title=f'İşlem {idx + 1}: {trade_type.capitalize()} (Kâr/Zarar: {trade["profit"]:.2f} USD)',
             ylabel='Fiyat (USDT)', addplot=apds, figscale=1.5, savefig=f'trade_{idx + 1}_chart.png')