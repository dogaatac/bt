import pandas as pd
import numpy as np
from colorama import init, Fore, Style

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
max_candles = 30  # Maksimum 10 mum
consecutive_candles = 4  # En az 2 ardışık mum
min_candles_for_second_condition = 10  # İkinci koşul için minimum 5 mum
max_candles_for_second_condition = 25  # İkinci koşul için maksimum 10 mum
risk_reward_ratio = 1.5  # Risk-Kazanç oranı (örneğin 1:2)

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

# Sweep ve geri dönüş takibi için listeler
sweeps_pl = []  # Buy side sweep’ler: (pivot_idx, pivot_price, sweep_low, sweep_idx)
sweeps_ph = []  # Sell side sweep’ler: (pivot_idx, pivot_price, sweep_high, sweep_idx)

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
                sweeps_ph.append((ph_idx, ph_price, current_high, i))
                used_pivots.add(ph_idx)

    # Buy side sweep kontrolü (fiyat pivot low’un altına inerse)
    for pl_idx, pl_price in active_pl.items():
        if pl_idx in used_pivots:
            continue
        if current_low < pl_price:
            manipulation_ratio = (pl_price - current_low) / pl_price
            if manipulation_ratio >= manipulation_threshold:
                sweeps_pl.append((pl_idx, pl_price, current_low, i))
                used_pivots.add(pl_idx)

    # Long işlem için geri dönüş kontrolü (buy side sweep sonrası)
    for sweep in sweeps_pl[:]:
        pl_idx, pl_price, sweep_low, sweep_idx = sweep
        bars_since_sweep = i - sweep_idx
        if bars_since_sweep > max_candles:
            sweeps_pl.remove(sweep)
            continue

        # Birinci Koşul: En az 2 ardışık mum pivot low’un üstünde kapanırsa
        if bars_since_sweep >= consecutive_candles:
            closes_above = all(df['close'].iloc[i - j] > pl_price for j in range(consecutive_candles))
            if closes_above:
                if i + 1 < len(df):
                    entry_price = df['open'].iloc[i + 1]
                    sl_price = sweep_low
                    sl_distance = entry_price - sl_price
                    if sl_distance > 0:
                        position_size = max_risk / sl_distance
                        tp_price = entry_price + sl_distance * risk_reward_ratio
                        positions.append({
                            'type': 'long',
                            'entry_time': df.index[i + 1],
                            'entry_price': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'size': position_size,
                            'pivot_price': pl_price,
                            'sweep_low': sweep_low
                        })
                        sweeps_pl.remove(sweep)
                        break

        # İkinci Koşul: Fiyat pivot low’un altında 5-10 mum kapanış yapıp geri dönerse
        if bars_since_sweep >= min_candles_for_second_condition:
            closes_below = all(df['close'].iloc[i - j] < pl_price for j in range(min_candles_for_second_condition, min(bars_since_sweep + 1, max_candles_for_second_condition + 1)))
            if closes_below and current_close > pl_price:
                if i + 1 < len(df):
                    entry_price = df['open'].iloc[i + 1]
                    sl_price = sweep_low
                    sl_distance = entry_price - sl_price
                    if sl_distance > 0:
                        position_size = max_risk / sl_distance
                        tp_price = entry_price + sl_distance * risk_reward_ratio
                        positions.append({
                            'type': 'long',
                            'entry_time': df.index[i + 1],
                            'entry_price': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'size': position_size,
                            'pivot_price': pl_price,
                            'sweep_low': sweep_low
                        })
                        sweeps_pl.remove(sweep)
                        break

    # Short işlem için geri dönüş kontrolü (sell side sweep sonrası)
    for sweep in sweeps_ph[:]:
        ph_idx, ph_price, sweep_high, sweep_idx = sweep
        bars_since_sweep = i - sweep_idx
        if bars_since_sweep > max_candles:
            sweeps_ph.remove(sweep)
            continue

        # Birinci Koşul: En az 2 ardışık mum pivot high’ın altında kapanırsa
        if bars_since_sweep >= consecutive_candles:
            closes_below = all(df['close'].iloc[i - j] < ph_price for j in range(consecutive_candles))
            if closes_below:
                if i + 1 < len(df):
                    entry_price = df['open'].iloc[i + 1]
                    sl_price = sweep_high
                    sl_distance = sl_price - entry_price
                    if sl_distance > 0:
                        position_size = max_risk / sl_distance
                        tp_price = entry_price - sl_distance * risk_reward_ratio
                        positions.append({
                            'type': 'short',
                            'entry_time': df.index[i + 1],
                            'entry_price': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'size': position_size,
                            'pivot_price': ph_price,
                            'sweep_high': sweep_high
                        })
                        sweeps_ph.remove(sweep)
                        break

        # İkinci Koşul: Fiyat pivot high’ın üstünde 5-10 mum kapanış yapıp geri dönerse
        if bars_since_sweep >= min_candles_for_second_condition:
            closes_above = all(df['close'].iloc[i - j] > ph_price for j in range(min_candles_for_second_condition, min(bars_since_sweep + 1, max_candles_for_second_condition + 1)))
            if closes_above and current_close < ph_price:
                if i + 1 < len(df):
                    entry_price = df['open'].iloc[i + 1]
                    sl_price = sweep_high
                    sl_distance = sl_price - entry_price
                    if sl_distance > 0:
                        position_size = max_risk / sl_distance
                        tp_price = entry_price - sl_distance * risk_reward_ratio
                        positions.append({
                            'type': 'short',
                            'entry_time': df.index[i + 1],
                            'entry_price': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'size': position_size,
                            'pivot_price': ph_price,
                            'sweep_high': sweep_high
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
                    'sweep_low': pos['sweep_low']
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
                    'sweep_low': pos['sweep_low']
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
                    'sweep_high': pos['sweep_high']
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
                    'sweep_high': pos['sweep_high']
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