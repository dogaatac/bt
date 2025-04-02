# engine.py
import pandas as pd
import numpy as np
from colorama import init, Fore, Style

# Colorama’yı başlat
init()

def pivot_high(series, left, right):
    series = np.array(series)  # NumPy dizisine çevir
    pivots = []
    for i in range(left, len(series) - right):
        left_slice = series[i - left:i]
        right_slice = series[i + 1:i + right + 1]
        if np.all(series[i] > left_slice) and np.all(series[i] > right_slice):
            pivots.append((i, float(series[i])))
    return pivots

def pivot_low(series, left, right):
    series = np.array(series)  # NumPy dizisine çevir
    pivots = []
    for i in range(left, len(series) - right):
        left_slice = series[i - left:i]
        right_slice = series[i + 1:i + right + 1]
        if np.all(series[i] < left_slice) and np.all(series[i] < right_slice):
            pivots.append((i, float(series[i])))
    return pivots

def run_engine(config, data_cache=None):
    # Veriyi yükle: data_cache varsa kullan, yoksa dosyadan oku
    if data_cache is not None and config.DATA_FILE in data_cache:
        df = data_cache[config.DATA_FILE]
    else:
        df = pd.read_csv(config.DATA_FILE)
        df['open_time'] = pd.to_datetime(df['open_time'])
        df.set_index('open_time', inplace=True)

    # Veriyi dizilere çevir
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    index = df.index

    # Pivot high ve low’ları hesapla
    ph = pivot_high(high, config.LEFT, config.RIGHT)
    pl = pivot_low(low, config.LEFT, config.RIGHT)

    # Başlangıç parametreleri
    balance = config.INITIAL_BALANCE
    positions = []
    trades = []
    used_pivots = set()

    # Pivotları sözlüğe çevir
    ph_dict = {idx: price for idx, price in ph}
    pl_dict = {idx: price for idx, price in pl}

    # Sweep ve manipülasyon takibi için listeler
    sweeps_pl = []  # Buy side sweep’ler: (pivot_idx, pivot_price, sweep_low, sweep_idx, manip_low, manip_high)
    sweeps_ph = []  # Sell side sweep’ler: (pivot_idx, pivot_price, sweep_high, sweep_idx, manip_low, manip_high)

    # Her bar için döngü
    for i in range((config.LEFT + config.RIGHT), len(df)):
        current_high = float(high[i])
        current_low = float(low[i])
        current_close = float(close[i])

        # Aktif pivotları kontrol et (son 200 bar)
        active_ph = {k: v for k, v in ph_dict.items() if k > i - 200 and k < i}
        active_pl = {k: v for k, v in pl_dict.items() if k > i - 200 and k < i}

        # Sell side sweep kontrolü (fiyat pivot high’ı aşarsa)
        for ph_idx, ph_price in active_ph.items():
            if ph_idx in used_pivots:
                continue
            if current_high > ph_price:
                manipulation_ratio = (current_high - ph_price) / ph_price
                if manipulation_ratio >= config.MANIPULATION_THRESHOLD:
                    sweeps_ph.append((ph_idx, ph_price, current_high, i, current_low, current_high))
                    used_pivots.add(ph_idx)

        # Buy side sweep kontrolü (fiyat pivot low’un altına inerse)
        for pl_idx, pl_price in active_pl.items():
            if pl_idx in used_pivots:
                continue
            if current_low < pl_price:
                manipulation_ratio = (pl_price - current_low) / pl_price
                if manipulation_ratio >= config.MANIPULATION_THRESHOLD:
                    sweeps_pl.append((pl_idx, pl_price, current_low, i, current_low, current_high))
                    used_pivots.add(pl_idx)

        # Long işlem için geri dönüş kontrolü (buy side sweep sonrası)
        for sweep in sweeps_pl[:]:
            pl_idx, pl_price, sweep_low, sweep_idx, manip_low, manip_high = sweep
            bars_since_sweep = i - sweep_idx
            if bars_since_sweep > config.MAX_CANDLES:
                sweeps_pl.remove(sweep)
                continue

            # Fiyat pivot low’un üstüne dönene kadar manipülasyon devam eder
            if current_close <= pl_price:
                manip_low = min(manip_low, current_low)
                manip_high = max(manip_high, current_high)
                sweeps_pl[sweeps_pl.index(sweep)] = (pl_idx, pl_price, sweep_low, sweep_idx, manip_low, manip_high)

            # Birinci Koşul: En az 4 ardışık mum pivot low’un üstünde kapanırsa
            if bars_since_sweep >= config.CONSECUTIVE_CANDLES:
                closes_above = all(float(close[i - j]) > pl_price for j in range(config.CONSECUTIVE_CANDLES))
                if closes_above:
                    if i + 1 < len(df):
                        entry_price = float(open_[i + 1])
                        sl_price = manip_low
                        sl_distance = entry_price - sl_price
                        if sl_distance > 0:
                            # Risk miktarını güncel bakiyeye göre hesapla
                            if config.RISK_TYPE == 'fixed':
                                risk_amount = config.INITIAL_BALANCE * config.MAX_RISK
                            else:  # config.RISK_TYPE == 'percentage'
                                risk_amount = balance * config.RISK_PERCENTAGE
                            position_size = risk_amount / sl_distance
                            tp_price = entry_price + sl_distance * config.RISK_REWARD_RATIO
                            positions.append({
                                'type': 'long',
                                'entry_time': index[i + 1],
                                'entry_price': entry_price,
                                'sl': sl_price,
                                'tp': tp_price,
                                'size': position_size,
                                'pivot_price': pl_price,
                                'sweep_low': sweep_low,
                                'sweep_time': index[sweep_idx],
                                'manip_low': manip_low,
                                'manip_high': manip_high,
                                'risk_amount': risk_amount
                            })
                            sweeps_pl.remove(sweep)
                            break

            # İkinci Koşul: Fiyat pivot low’un altında 5-20 mum kapanış yapıp geri dönerse
            if bars_since_sweep >= config.MIN_CANDLES_FOR_SECOND_CONDITION:
                closes_below = all(float(close[i - j]) < pl_price for j in range(config.MIN_CANDLES_FOR_SECOND_CONDITION, min(bars_since_sweep + 1, config.MAX_CANDLES_FOR_SECOND_CONDITION + 1)))
                if closes_below and current_close > pl_price:
                    if i + 1 < len(df):
                        entry_price = float(open_[i + 1])
                        sl_price = manip_low
                        sl_distance = entry_price - sl_price
                        if sl_distance > 0:
                            # Risk miktarını güncel bakiyeye göre hesapla
                            if config.RISK_TYPE == 'fixed':
                                risk_amount = config.INITIAL_BALANCE * config.MAX_RISK
                            else:  # config.RISK_TYPE == 'percentage'
                                risk_amount = balance * config.RISK_PERCENTAGE
                            position_size = risk_amount / sl_distance
                            tp_price = entry_price + sl_distance * config.RISK_REWARD_RATIO
                            positions.append({
                                'type': 'long',
                                'entry_time': index[i + 1],
                                'entry_price': entry_price,
                                'sl': sl_price,
                                'tp': tp_price,
                                'size': position_size,
                                'pivot_price': pl_price,
                                'sweep_low': sweep_low,
                                'sweep_time': index[sweep_idx],
                                'manip_low': manip_low,
                                'manip_high': manip_high,
                                'risk_amount': risk_amount
                            })
                            sweeps_pl.remove(sweep)
                            break

        # Short işlem için geri dönüş kontrolü (sell side sweep sonrası)
        for sweep in sweeps_ph[:]:
            ph_idx, ph_price, sweep_high, sweep_idx, manip_low, manip_high = sweep
            bars_since_sweep = i - sweep_idx
            if bars_since_sweep > config.MAX_CANDLES:
                sweeps_ph.remove(sweep)
                continue

            # Fiyat pivot high’ın altına dönene kadar manipülasyon devam eder
            if current_close >= ph_price:
                manip_low = min(manip_low, current_low)
                manip_high = max(manip_high, current_high)
                sweeps_ph[sweeps_ph.index(sweep)] = (ph_idx, ph_price, sweep_high, sweep_idx, manip_low, manip_high)

            # Birinci Koşul: En az 4 ardışık mum pivot high’ın altında kapanırsa
            if bars_since_sweep >= config.CONSECUTIVE_CANDLES:
                closes_below = all(float(close[i - j]) < ph_price for j in range(config.CONSECUTIVE_CANDLES))
                if closes_below:
                    if i + 1 < len(df):
                        entry_price = float(open_[i + 1])
                        sl_price = manip_high
                        sl_distance = sl_price - entry_price
                        if sl_distance > 0:
                            # Risk miktarını güncel bakiyeye göre hesapla
                            if config.RISK_TYPE == 'fixed':
                                risk_amount = config.INITIAL_BALANCE * config.MAX_RISK
                            else:  # config.RISK_TYPE == 'percentage'
                                risk_amount = balance * config.RISK_PERCENTAGE
                            position_size = risk_amount / sl_distance
                            tp_price = entry_price - sl_distance * config.RISK_REWARD_RATIO
                            positions.append({
                                'type': 'short',
                                'entry_time': index[i + 1],
                                'entry_price': entry_price,
                                'sl': sl_price,
                                'tp': tp_price,
                                'size': position_size,
                                'pivot_price': ph_price,
                                'sweep_high': sweep_high,
                                'sweep_time': index[sweep_idx],
                                'manip_low': manip_low,
                                'manip_high': manip_high,
                                'risk_amount': risk_amount
                            })
                            sweeps_ph.remove(sweep)
                            break

            # İkinci Koşul: Fiyat pivot high’ın üstünde 5-20 mum kapanış yapıp geri dönerse
            if bars_since_sweep >= config.MIN_CANDLES_FOR_SECOND_CONDITION:
                closes_above = all(float(close[i - j]) > ph_price for j in range(config.MIN_CANDLES_FOR_SECOND_CONDITION, min(bars_since_sweep + 1, config.MAX_CANDLES_FOR_SECOND_CONDITION + 1)))
                if closes_above and current_close < ph_price:
                    if i + 1 < len(df):
                        entry_price = float(open_[i + 1])
                        sl_price = manip_high
                        sl_distance = sl_price - entry_price
                        if sl_distance > 0:
                            # Risk miktarını güncel bakiyeye göre hesapla
                            if config.RISK_TYPE == 'fixed':
                                risk_amount = config.INITIAL_BALANCE * config.MAX_RISK
                            else:  # config.RISK_TYPE == 'percentage'
                                risk_amount = balance * config.RISK_PERCENTAGE
                            position_size = risk_amount / sl_distance
                            tp_price = entry_price - sl_distance * config.RISK_REWARD_RATIO
                            positions.append({
                                'type': 'short',
                                'entry_time': index[i + 1],
                                'entry_price': entry_price,
                                'sl': sl_price,
                                'tp': tp_price,
                                'size': position_size,
                                'pivot_price': ph_price,
                                'sweep_high': sweep_high,
                                'sweep_time': index[sweep_idx],
                                'manip_low': manip_low,
                                'manip_high': manip_high,
                                'risk_amount': risk_amount
                            })
                            sweeps_ph.remove(sweep)
                            break

        # Açık pozisyonları kontrol et
        for pos in positions[:]:
            if pos['type'] == 'long':
                if current_low <= pos['sl']:
                    profit = -pos['risk_amount']
                    balance += profit
                    trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': index[i],
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
                        'manip_high': pos['manip_high'],
                        'risk_amount': pos['risk_amount']
                    })
                    positions.remove(pos)
                elif current_high >= pos['tp']:
                    profit = pos['risk_amount'] * config.RISK_REWARD_RATIO
                    balance += profit
                    trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': index[i],
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
                        'manip_high': pos['manip_high'],
                        'risk_amount': pos['risk_amount']
                    })
                    positions.remove(pos)
            elif pos['type'] == 'short':
                if current_high >= pos['sl']:
                    profit = -pos['risk_amount']
                    balance += profit
                    trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': index[i],
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
                        'manip_high': pos['manip_high'],
                        'risk_amount': pos['risk_amount']
                    })
                    positions.remove(pos)
                elif current_low <= pos['tp']:
                    profit = pos['risk_amount'] * config.RISK_REWARD_RATIO
                    balance += profit
                    trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': index[i],
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
                        'manip_high': pos['manip_high'],
                        'risk_amount': pos['risk_amount']
                    })
                    positions.remove(pos)

    return df, trades, balance

def print_trades(trades, print_format):
    print("\n" + Fore.CYAN + "İşlem Detayları:" + Style.RESET_ALL)
    for trade in trades:
        if trade['type'] == 'long':
            if print_format == 'detailed':
                print(f"{Fore.GREEN}Long İşlem{Style.RESET_ALL} - "
                      f"Entry Time: {trade['entry_time']}, "
                      f"Exit Time: {trade['exit_time']}, "
                      f"Pivot Low: {trade['pivot_price']:.2f}, "
                      f"Sweep Low: {trade['sweep_low']:.2f}, "
                      f"Manip Low: {trade['manip_low']:.2f}, "
                      f"Entry: {trade['entry_price']:.2f}, "
                      f"SL: {trade['sl']:.2f}, "
                      f"TP: {trade['tp']:.2f}, "
                      f"Profit: {trade['profit']:.2f} USD, "
                      f"Risk Amount: {trade['risk_amount']:.2f} USD")
            elif print_format == 'simple':
                print(f"{Fore.GREEN}Long İşlem{Style.RESET_ALL} - "
                      f"Entry: {trade['entry_price']:.2f}, "
                      f"Profit: {trade['profit']:.2f} USD, "
                      f"Risk: {trade['risk_amount']:.2f} USD")
        elif trade['type'] == 'short':
            if print_format == 'detailed':
                print(f"{Fore.RED}Short İşlem{Style.RESET_ALL} - "
                      f"Entry Time: {trade['entry_time']}, "
                      f"Exit Time: {trade['exit_time']}, "
                      f"Pivot High: {trade['pivot_price']:.2f}, "
                      f"Sweep High: {trade['sweep_high']:.2f}, "
                      f"Manip High: {trade['manip_high']:.2f}, "
                      f"Entry: {trade['entry_price']:.2f}, "
                      f"SL: {trade['sl']:.2f}, "
                      f"TP: {trade['tp']:.2f}, "
                      f"Profit: {trade['profit']:.2f} USD, "
                      f"Risk Amount: {trade['risk_amount']:.2f} USD")
            elif print_format == 'simple':
                print(f"{Fore.RED}Short İşlem{Style.RESET_ALL} - "
                      f"Entry: {trade['entry_price']:.2f}, "
                      f"Profit: {trade['profit']:.2f} USD, "
                      f"Risk: {trade['risk_amount']:.2f} USD")

def print_summary(initial_balance, final_balance, trades):
    print(f"\n{Fore.CYAN}Başlangıç Bakiyesi: {initial_balance} USD{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Son Bakiye: {final_balance:.2f} USD{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Toplam İşlem Sayısı: {len(trades)}{Style.RESET_ALL}")
    profitable_trades = sum(1 for trade in trades if trade['profit'] > 0)
    losing_trades = sum(1 for trade in trades if trade['profit'] < 0)
    total_profit = sum(trade['profit'] for trade in trades)
    print(f"{Fore.CYAN}Karlı İşlem Sayısı: {profitable_trades}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Zararlı İşlem Sayısı: {losing_trades}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Toplam Kâr/Zarar: {total_profit:.2f} USD{Style.RESET_ALL}")