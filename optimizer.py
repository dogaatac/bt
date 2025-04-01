# optimizer.py
import itertools
import pandas as pd
from engine import run_engine
import config
import time
from multiprocessing import Pool
import random

# Ayar değerleri (en üstte)
LEFT_VALUES = [15, 20, 25, 30]
RIGHT_VALUES = [15, 20, 25, 30]
MANIPULATION_THRESHOLD_VALUES = [0.005, 0.01, 0.015, 0.02]
MAX_CANDLES_VALUES = [20]
CONSECUTIVE_CANDLES_VALUES = [2, 3, 4]
MIN_CANDLES_FOR_SECOND_CONDITION_VALUES =  [15, 20,25]
MAX_CANDLES_FOR_SECOND_CONDITION_VALUES = [20 ,25, 30]
RISK_REWARD_RATIO_VALUES = [1.5]

# Veri dosyaları
DATA_FILES = [
    "btc_usdt_15m-1yil.csv"
]

# Ağırlıklar (BTC-USDT chartlarına daha fazla ağırlık)
WEIGHTS = {
    "btc_usdt_15m.csv": 0.6,
    "eth_usdt_15m.csv": 0.4,
    "sol_usdt_15m.csv": 0.4,
    "btc_usdt_15m-1yil.csv": 1,
    "btc_usdt_15m-3yil.csv": 0.6
}

# Verileri bir kez yükle ve önbelleğe al
DATA_CACHE = {}
for data_file in DATA_FILES:
    df = pd.read_csv(data_file)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)
    DATA_CACHE[data_file] = df

def test_combination(combo):
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
        file_results = {}
        all_monthly_rates = []
        for data_file in DATA_FILES:
            config.DATA_FILE = data_file
            try:
                _, trades, _ = run_engine(config, data_cache=DATA_CACHE)
                total_profit = sum(trade['profit'] for trade in trades)
                file_results[data_file] = total_profit

                # Aylık başarı oranlarını hesapla
                if trades:
                    trade_data = pd.DataFrame(trades)
                    trade_data['exit_time'] = pd.to_datetime(trade_data['exit_time'])
                    trade_data['month'] = trade_data['exit_time'].dt.tz_localize(None).dt.to_period('M')
                    monthly_stats = trade_data.groupby('month').apply(
                        lambda x: len(x[x['profit'] > 0]) / len(x) * 100 if len(x) > 0 else 0
                    )
                    all_monthly_rates.extend(monthly_stats.values)
            except Exception as e:
                print(f"Hata: {data_file} için kombinasyon {combo} çalıştırılamadı. Hata: {e}")
                file_results[data_file] = float('-inf')

        # Ağırlıklı ortalama kâr hesapla
        weighted_profit = sum(file_results[file] * WEIGHTS.get(file, 0.6) for file in file_results) / sum(WEIGHTS.values())

        # Ortalama ve minimum başarı oranını hesapla
        avg_success_rate = sum(all_monthly_rates) / len(all_monthly_rates) if all_monthly_rates else 0
        min_success_rate = min(all_monthly_rates) if all_monthly_rates else 0

        return {
            'combination': combo,
            'file_results': file_results,
            'weighted_profit': weighted_profit,
            'avg_success_rate': avg_success_rate,
            'min_success_rate': min_success_rate
        }
    except Exception as e:
        print(f"Kombinasyon {combo} için genel hata: {e}")
        return {
            'combination': combo,
            'file_results': {file: float('-inf') for file in DATA_FILES},
            'weighted_profit': float('-inf'),
            'avg_success_rate': 0,
            'min_success_rate': 0
        }

def run_optimization():
    # Kullanıcıdan maksimum deneme adedini al
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

    # Toplam kombinasyon sayısını yazdır
    total_combinations = len(all_combinations)
    print(f"Toplam mümkün kombinasyon sayısı: {total_combinations}")

    # Maksimum deneme adedine göre rastgele kombinasyon seç
    if max_trials >= total_combinations:
        combinations = all_combinations
    else:
        combinations = random.sample(all_combinations, max_trials)

    print(f"Test edilecek kombinasyon sayısı: {len(combinations)}")
    start_time = time.time()

    # Paralel işlem havuzu oluştur (sabit 23 çekirdek)
    num_processes = 23
    print(f"Kullanılan işlemci çekirdek sayısı: {num_processes}")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(test_combination, combinations)

    # Ortalama başarı oranı %45 ve üstü olanları filtrele
    valid_results = [r for r in results if r['avg_success_rate'] >= 45]
    if not valid_results:
        print("\nOrtalama %45 ve üstü başarı oranı sağlayan kombinasyon bulunamadı.")
        return

    # Sonuçları ortalama başarı oranına göre sırala
    valid_results.sort(key=lambda x: x['avg_success_rate'], reverse=True)

    # En başarılı 5 kombinasyonu yazdır
    print(f"\n=== En Yüksek Ortalama Başarı Oranına Sahip 5 Setup (Toplam {len(valid_results)} adet) ===")
    for i, result in enumerate(valid_results[:5]):
        combo = result['combination']
        print(f"\n{i + 1}. Kombinasyon:")
        print(f"LEFT: {combo[0]}, RIGHT: {combo[1]}, MANIPULATION_THRESHOLD: {combo[2]}, "
              f"MAX_CANDLES: {combo[3]}, CONSECUTIVE_CANDLES: {combo[4]}, "
              f"MIN_CANDLES_FOR_SECOND_CONDITION: {combo[5]}, MAX_CANDLES_FOR_SECOND_CONDITION: {combo[6]}, "
              f"RISK_REWARD_RATIO: {combo[7]}")
        print(f"Ortalama Başarı Oranı: {result['avg_success_rate']:.2f}%")
        print(f"Minimum Başarı Oranı: {result['min_success_rate']:.2f}%")
        print(f"Ağırlıklı Kâr: {result['weighted_profit']:.2f} USD")

    # Toplam süreyi yazdır
    total_time = time.time() - start_time
    print(f"\nToplam optimizasyon süresi: {total_time:.2f} saniye")

if __name__ == "__main__":
    run_optimization()