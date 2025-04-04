# optimizer.py
import itertools
import pandas as pd
from engine import run_engine
import config
import time
from multiprocessing import Pool
import random
import numpy as np
import multiprocessing

# Parametre aralıkları
LEFT_VALUES = list(range(10, 30, 5))  # [10, 15, 20, 25]
RIGHT_VALUES = list(range(10, 30, 5))  # [10, 15, 20, 25]
MANIPULATION_THRESHOLD_VALUES = [x / 1000 for x in range(2, 14, 2)]  # [0.002, 0.004, 0.006, 0.008, 0.010, 0.012]
MAX_CANDLES_VALUES = [15, 30]
CONSECUTIVE_CANDLES_VALUES = [2, 3, 4]
MIN_CANDLES_FOR_SECOND_CONDITION_VALUES = [5, 10, 15, 20]
MAX_CANDLES_FOR_SECOND_CONDITION_VALUES = [15, 20, 25, 30]
RISK_REWARD_RATIO_VALUES = [1.5]

# Veri dosyaları
DATA_FILES = ["1yil.csv"]

# Ağırlıklar (veri dosyalarına göre)
WEIGHTS = {"1yil.csv": 1}

# Verileri bir kez yükle ve önbelleğe al, aynı zamanda toplam gün sayısını hesapla
DATA_CACHE = {}
TOTAL_DAYS = {}
for data_file in DATA_FILES:
    df = pd.read_csv(data_file)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)
    DATA_CACHE[data_file] = df
    # Toplam gün sayısını hesapla (ilk ve son tarih arasındaki fark)
    total_days = (df.index.max() - df.index.min()).days + 1  # +1 çünkü son gün dahil
    TOTAL_DAYS[data_file] = max(total_days, 1)  # 0 olmaması için minimum 1 gün

def test_combination(combo):
    """Belirli bir kombinasyonu üç dosyada test eder ve metrikleri hesaplar."""
    try:
        left, right, manip_threshold, max_candles, consec_candles, min_candles_2nd, max_candles_2nd, rr_ratio = combo

        # Config değerlerini güncelle
        config.LEFT = left
        config.RIGHT = right
        config.MANIPULATION_THRESHOLD = manip_threshold
        config.MAX_CANDLES = max_candles
        config.CONSECUTIVE_CANDLES = consec_candles
        config.MIN_CANDLES_FOR_SECOND_CONDITION = min_candles_2nd
        config.MAX_CANDLES_FOR_SECOND_CONDITION = max_candles_2nd
        config.RISK_REWARD_RATIO = rr_ratio

        # Her veri dosyası için test yap
        all_trades = []
        file_results = {}
        for data_file in DATA_FILES:
            config.DATA_FILE = data_file
            try:
                _, trades, _ = run_engine(config, data_cache=DATA_CACHE)
                total_profit = sum(trade['profit'] for trade in trades)
                file_results[data_file] = total_profit
                all_trades.extend(trades)
            except Exception as e:
                print(f"Hata: {data_file} için kombinasyon {combo} çalıştırılamadı. Hata: {e}")
                file_results[data_file] = float('-inf')

        # Ağırlıklı kâr hesapla
        weighted_profit = sum(file_results[file] * WEIGHTS[file] for file in file_results) / len(DATA_FILES)

        # Toplam başarı oranı
        total_success_rate = len([t for t in all_trades if t['profit'] > 0]) / len(all_trades) * 100 if all_trades else 0

        # Profit Factor
        trade_df = pd.DataFrame(all_trades)
        gross_profit = trade_df[trade_df['profit'] > 0]['profit'].sum()
        gross_loss = abs(trade_df[trade_df['profit'] < 0]['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

        # Sharpe Ratio ve Maksimum Düşüş
        if 'exit_time' in trade_df.columns and not trade_df['exit_time'].isna().all():
            trade_df['exit_time'] = pd.to_datetime(trade_df['exit_time']).dt.tz_localize(None)
            monthly_profits = trade_df.groupby(trade_df['exit_time'].dt.to_period('M'))['profit'].sum()
            mean_monthly_profit = monthly_profits.mean()
            std_monthly_profit = monthly_profits.std()
            sharpe_ratio = mean_monthly_profit / std_monthly_profit if std_monthly_profit != 0 else 0
            trade_df['cumulative_profit'] = trade_df['profit'].cumsum()
            max_drawdown = (trade_df['cumulative_profit'].cummax() - trade_df['cumulative_profit']).max()
        else:
            sharpe_ratio = 0
            max_drawdown = float('inf') if not all_trades else max(0, -min(0, min([t['profit'] for t in all_trades])))

        # Ortalama kâr
        avg_profit = np.mean([t['profit'] for t in all_trades]) if all_trades else 0

        # İşlem sayısı
        trade_count = len(all_trades)

        # Gün başına işlem sayısını hesapla (tüm veri dosyaları için ortalama)
        trades_per_day = 0
        for data_file in DATA_FILES:
            trades_per_day += trade_count / TOTAL_DAYS[data_file]
        trades_per_day /= len(DATA_FILES)  # Ortalama al

        return {
            'left': left,
            'right': right,
            'manip_threshold': manip_threshold,
            'max_candles': max_candles,
            'consec_candles': consec_candles,
            'min_candles_2nd': min_candles_2nd,
            'max_candles_2nd': max_candles_2nd,
            'rr_ratio': rr_ratio,
            'weighted_profit': weighted_profit,
            'total_success_rate': total_success_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_profit': avg_profit,
            'trades_per_day': trades_per_day,  # Gün başına işlem sayısı
            'file_results': str(file_results)
        }
    except Exception as e:
        print(f"Kombinasyon {combo} için hata: {e}")
        return {
            'left': combo[0],
            'right': combo[1],
            'manip_threshold': combo[2],
            'max_candles': combo[3],
            'consec_candles': combo[4],
            'min_candles_2nd': combo[5],
            'max_candles_2nd': combo[6],
            'rr_ratio': combo[7],
            'weighted_profit': float('-inf'),
            'total_success_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': float('inf'),
            'avg_profit': 0,
            'trades_per_day': 0,  # Gün başına işlem sayısı
            'file_results': str({file: float('-inf') for file in DATA_FILES})
        }

def calculate_rank_percentage(rank, total_count):
    """Sıralamayı yüzdeye çevirir (%100 en iyisi, %0 en kötüsü)."""
    return ((total_count - rank) / (total_count - 1)) * 100 if total_count > 1 else 100

def run_optimization():
    """Optimizasyon sürecini çalıştırır, skorları hesaplar ve en iyi 25 kombinasyonu ekrana yazdırır ve CSV'ye kaydeder."""
    while True:
        try:
            max_trials = int(input("Kaç farklı kombinasyon denensin? (Örnek: 1000): "))
            if max_trials <= 0:
                print("Lütfen pozitif bir sayı girin.")
                continue
            break
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")

    # Tüm kombinasyonları oluştur
    all_combinations = list(itertools.product(
        LEFT_VALUES,
        RIGHT_VALUES,
        MANIPULATION_THRESHOLD_VALUES,
        MAX_CANDLES_VALUES,
        CONSECUTIVE_CANDLES_VALUES,
        MIN_CANDLES_FOR_SECOND_CONDITION_VALUES,
        MAX_CANDLES_FOR_SECOND_CONDITION_VALUES,
        RISK_REWARD_RATIO_VALUES
    ))

    total_combinations = len(all_combinations)
    print(f"Toplam mümkün kombinasyon sayısı: {total_combinations}")

    # Rastgele kombinasyon seç
    if max_trials >= total_combinations:
        combinations = all_combinations
    else:
        combinations = random.sample(all_combinations, max_trials)

    print(f"Test edilecek kombinasyon sayısı: {len(combinations)}")
    start_time = time.time()

    # Paralel işlem
    num_processes = min(28, multiprocessing.cpu_count())
    print(f"Kullanılan işlemci çekirdek sayısı: {num_processes}")

    with Pool(processes=num_processes) as pool:
        results = pool.map(test_combination, combinations)

    # Sonuçları DataFrame'e çevir
    df = pd.DataFrame(results)

    # Skor hesaplama için normalize etme
    df['norm_profit'] = (df['weighted_profit'] - df['weighted_profit'].min()) / (df['weighted_profit'].max() - df['weighted_profit'].min())
    df['norm_trade_count'] = (df['trades_per_day'] - df['trades_per_day'].min()) / (df['trades_per_day'].max() - df['trades_per_day'].min())  # Daha fazla işlem iyi
    df['norm_success'] = (df['total_success_rate'] - df['total_success_rate'].min()) / (df['total_success_rate'].max() - df['total_success_rate'].min())
    df['norm_pf'] = (df['profit_factor'] - df['profit_factor'].min()) / (df['profit_factor'].max() - df['profit_factor'].min())
    df['norm_sharpe'] = (df['sharpe_ratio'] - df['sharpe_ratio'].min()) / (df['sharpe_ratio'].max() - df['sharpe_ratio'].min())
    df['norm_drawdown'] = (df['max_drawdown'].max() - df['max_drawdown']) / (df['max_drawdown'].max() - df['max_drawdown'].min())  # Düşük olması iyi
    df['norm_avg_profit'] = (df['avg_profit'] - df['avg_profit'].min()) / (df['avg_profit'].max() - df['avg_profit'].min())

    # NaN değerlerini 0 ile doldur
    df.fillna(0, inplace=True)

    # Toplam skor hesaplama (mevcut ağırlıklar korunuyor)
    df['total_score'] = (0.20 * df['norm_profit'] + 
                         0.15 * df['norm_trade_count'] + 
                         0.20 * df['norm_success'] + 
                         0.15 * df['norm_pf'] + 
                         0.15 * df['norm_sharpe'] + 
                         0.15 * df['norm_drawdown'] + 
                         0.15 * df['norm_avg_profit']) * 100

    # Her metrik için sıralama
    df['rank_profit'] = df['weighted_profit'].rank(ascending=False, method='min')
    df['rank_trade_count'] = df['trades_per_day'].rank(ascending=False, method='min')  # Daha fazla işlem iyi
    df['rank_success'] = df['total_success_rate'].rank(ascending=False, method='min')
    df['rank_pf'] = df['profit_factor'].rank(ascending=False, method='min')
    df['rank_sharpe'] = df['sharpe_ratio'].rank(ascending=False, method='min')
    df['rank_drawdown'] = df['max_drawdown'].rank(ascending=True, method='min')  # Düşük olması iyi
    df['rank_avg_profit'] = df['avg_profit'].rank(ascending=False, method='min')

    # Sonuçları skoruna göre sırala (en iyiden en kötüye)
    df = df.sort_values(by='total_score', ascending=False)

    # Tüm sonuçları CSV'ye kaydet
    df.to_csv('optimization_results.csv', index=False)
    print("\nTüm sonuçlar 'optimization_results.csv' dosyasına kaydedildi.")

    # En iyi 25 kombinasyonu seç
    top_25 = df.head(25).copy()

    # Terminalde her kombinasyonu ayrı blok halinde yazdır
    print("\n=== En İyi 25 Kombinasyon ===")
    for index, row in top_25.iterrows():
        rank = index + 1
        total_combinations = len(df)
        profit_rank_percent = calculate_rank_percentage(row['rank_profit'], total_combinations)
        trade_count_rank_percent = calculate_rank_percentage(row['rank_trade_count'], total_combinations)
        success_rank_percent = calculate_rank_percentage(row['rank_success'], total_combinations)
        pf_rank_percent = calculate_rank_percentage(row['rank_pf'], total_combinations)
        sharpe_rank_percent = calculate_rank_percentage(row['rank_sharpe'], total_combinations)
        drawdown_rank_percent = calculate_rank_percentage(row['rank_drawdown'], total_combinations)
        avg_profit_rank_percent = calculate_rank_percentage(row['rank_avg_profit'], total_combinations)

        print(f"\n{rank}. Kombinasyon (Toplam Skor: {row['total_score']:.2f}):")
        print(f"Config: LEFT={row['left']}, RIGHT={row['right']}, MANIPULATION_THRESHOLD={row['manip_threshold']}, "
              f"MAX_CANDLES={row['max_candles']}, CONSECUTIVE_CANDLES={row['consec_candles']}, "
              f"MIN_CANDLES_2ND={row['min_candles_2nd']}, MAX_CANDLES_2ND={row['max_candles_2nd']}, RR_RATIO={row['rr_ratio']}")
        print(f"Toplam Kâr: {row['weighted_profit']:.2f} USD (Sıra: {profit_rank_percent:.2f}%)")
        print(f"Ortalama Kâr: {row['avg_profit']:.2f} USD (Sıra: {avg_profit_rank_percent:.2f}%)")
        print(f"Gün Başına İşlem Sayısı: {row['trades_per_day']:.2f} (Sıra: {trade_count_rank_percent:.2f}%)")
        print(f"Toplam Başarı Oranı: {row['total_success_rate']:.2f}% (Sıra: {success_rank_percent:.2f}%)")
        print(f"Profit Factor: {row['profit_factor']:.2f} (Sıra: {pf_rank_percent:.2f}%)")
        print(f"Sharpe Ratio: {row['sharpe_ratio']:.2f} (Sıra: {sharpe_rank_percent:.2f}%)")
        print(f"Maksimum Düşüş: {row['max_drawdown']:.2f} USD (Sıra: {drawdown_rank_percent:.2f}%)")

    # CSV dosyasına kaydet
    output_df = pd.DataFrame({
        'Sıra': range(1, len(top_25) + 1),
        'Toplam Skor': top_25['total_score'].round(2),
        'LEFT': top_25['left'],
        'RIGHT': top_25['right'],
        'MANIPULATION_THRESHOLD': top_25['manip_threshold'],
        'MAX_CANDLES': top_25['max_candles'],
        'CONSECUTIVE_CANDLES': top_25['consec_candles'],
        'MIN_CANDLES_2ND': top_25['min_candles_2nd'],
        'MAX_CANDLES_2ND': top_25['max_candles_2nd'],
        'RR_RATIO': top_25['rr_ratio'],
        'Toplam Kâr (USD)': top_25['weighted_profit'].round(2),
        'Toplam Kâr Sıra %': [calculate_rank_percentage(r, len(df)) for r in top_25['rank_profit']],
        'Ortalama Kâr (USD)': top_25['avg_profit'].round(2),
        'Ortalama Kâr Sıra %': [calculate_rank_percentage(r, len(df)) for r in top_25['rank_avg_profit']],
        'Gün Başına İşlem Sayısı': top_25['trades_per_day'].round(2),
        'Gün Başına İşlem Sayısı Sıra %': [calculate_rank_percentage(r, len(df)) for r in top_25['rank_trade_count']],
        'Başarı Oranı (%)': top_25['total_success_rate'].round(2),
        'Başarı Oranı Sıra %': [calculate_rank_percentage(r, len(df)) for r in top_25['rank_success']],
        'Profit Factor': top_25['profit_factor'].round(2),
        'Profit Factor Sıra %': [calculate_rank_percentage(r, len(df)) for r in top_25['rank_pf']],
        'Sharpe Ratio': top_25['sharpe_ratio'].round(2),
        'Sharpe Ratio Sıra %': [calculate_rank_percentage(r, len(df)) for r in top_25['rank_sharpe']],
        'Maksimum Düşüş (USD)': top_25['max_drawdown'].round(2),
        'Maksimum Düşüş Sıra %': [calculate_rank_percentage(r, len(df)) for r in top_25['rank_drawdown']]
    })

    output_file = 'top_25_results.csv'
    output_df.to_csv(output_file, index=False)
    print(f"\nEn iyi 25 kombinasyonun temel verileri '{output_file}' dosyasına kaydedildi.")

    # Süreyi yazdır
    total_time = time.time() - start_time
    print(f"\nToplam optimizasyon süresi: {total_time:.2f} saniye")

if __name__ == "__main__":
    run_optimization()