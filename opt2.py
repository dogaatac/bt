# optimizer.py
import itertools
import pandas as pd
from engine import run_engine
import config
import time
from multiprocessing import Pool
import random

# Ayar değerleri (en üstte)
LEFT_VALUES = [5,15,20,25,30]
RIGHT_VALUES = [5,15,15,20, 25,30]
MANIPULATION_THRESHOLD_VALUES = [0.0015,0.003,0.005,0.008 ,0.01]
MAX_CANDLES_VALUES = [15,30]
CONSECUTIVE_CANDLES_VALUES = [2, 3, 4]
MIN_CANDLES_FOR_SECOND_CONDITION_VALUES = [5, 10, 15,20]
MAX_CANDLES_FOR_SECOND_CONDITION_VALUES = [15,20, 25,30]
RISK_REWARD_RATIO_VALUES = [1.25,1.5]

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
    "btc_usdt_15m-3yil.csv": 1
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
        all_trades = []
        all_monthly_rates = []
        for data_file in DATA_FILES:
            config.DATA_FILE = data_file
            try:
                _, trades, _ = run_engine(config, data_cache=DATA_CACHE)
                total_profit = sum(trade['profit'] for trade in trades)
                file_results[data_file] = total_profit
                all_trades.extend(trades)

                # Aylık başarı oranlarını hesapla (tüm aylar dahil)
                if trades:
                    trade_data = pd.DataFrame(trades)
                    trade_data['exit_time'] = pd.to_datetime(trade_data['exit_time']).dt.tz_localize(None)  # Saat dilimini kaldır
                    trade_data['month'] = trade_data['exit_time'].dt.to_period('M')
                    
                    monthly_success = trade_data.groupby('month').apply(
                        lambda x: len(x[x['profit'] > 0]) / len(x) * 100 if len(x) > 0 else 0
                    )
                    all_monthly_rates.extend(monthly_success.values)
            except Exception as e:
                print(f"Hata: {data_file} için kombinasyon {combo} çalıştırılamadı. Hata: {e}")
                file_results[data_file] = float('-inf')

        # Ağırlıklı ortalama kâr hesapla
        weighted_profit = sum(file_results[file] * WEIGHTS.get(file, 0.6) for file in file_results) / sum(WEIGHTS.values())

        # Toplam başarı oranını hesapla (tüm işlemler üzerinden)
        total_success_rate = len([t for t in all_trades if t['profit'] > 0]) / len(all_trades) * 100 if all_trades else 0
        
        # Ortalama ve minimum aylık başarı oranını hesapla (tüm aylar)
        avg_success_rate = sum(all_monthly_rates) / len(all_monthly_rates) if all_monthly_rates else 0
        min_success_rate = min(all_monthly_rates) if all_monthly_rates else 0

        # Sharpe Ratio hesaplama
        trade_df = pd.DataFrame(all_trades)
        trade_df['exit_time'] = pd.to_datetime(trade_df['exit_time']).dt.tz_localize(None)  # Saat dilimini kaldır
        monthly_profits = trade_df.groupby(trade_df['exit_time'].dt.to_period('M'))['profit'].sum()
        mean_monthly_profit = monthly_profits.mean()
        std_monthly_profit = monthly_profits.std()
        sharpe_ratio = mean_monthly_profit / std_monthly_profit if std_monthly_profit != 0 else 0

        # Profit Factor hesaplama
        gross_profit = trade_df[trade_df['profit'] > 0]['profit'].sum()
        gross_loss = abs(trade_df[trade_df['profit'] < 0]['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 10  # Üst sınır 10

        return {
            'combination': combo,
            'file_results': file_results,
            'weighted_profit': weighted_profit,
            'total_success_rate': total_success_rate,
            'avg_success_rate': avg_success_rate,
            'min_success_rate': min_success_rate,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor
        }
    except Exception as e:
        print(f"Kombinasyon {combo} için genel hata: {e}")
        return {
            'combination': combo,
            'file_results': {file: float('-inf') for file in DATA_FILES},
            'weighted_profit': float('-inf'),
            'total_success_rate': 0,
            'avg_success_rate': 0,
            'min_success_rate': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0
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

    # Paralel işlem havuzu oluştur (sabit 28 çekirdek)
    num_processes = 28
    print(f"Kullanılan işlemci çekirdek sayısı: {num_processes}")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(test_combination, combinations)

    # Toplam başarı oranı %40 ve üstü olanları filtrele
    valid_results = [r for r in results if r['total_success_rate'] >= 40]
    if not valid_results:
        print("\nToplam başarı oranı %40 ve üstü sağlayan kombinasyon bulunamadı.")
        return

    # Normalize etme için min ve max değerleri hesapla
    profits = [r['weighted_profit'] for r in valid_results]
    sharpe_ratios = [max(0, r['sharpe_ratio']) for r in valid_results]  # Negatifleri 0'a çek
    profit_factors = [min(10, r['profit_factor']) for r in valid_results]  # Üst sınır 10

    min_profit = min(profits)
    max_profit = max(profits)
    profit_range = max_profit - min_profit if max_profit != min_profit else 1

    min_sharpe = min(sharpe_ratios)
    max_sharpe = max(sharpe_ratios)
    sharpe_range = max_sharpe - min_sharpe if max_sharpe != min_sharpe else 1

    min_pf = min(profit_factors)
    max_pf = max(profit_factors)
    pf_range = max_pf - min_pf if max_pf != min_pf else 1

    # Kombinasyon skoru hesaplama
    for result in valid_results:
        normalized_profit = ((result['weighted_profit'] - min_profit) / profit_range) * 100
        normalized_sharpe = ((max(0, result['sharpe_ratio']) - min_sharpe) / sharpe_range) * 100 if sharpe_range != 0 else 0
        normalized_pf = ((min(10, result['profit_factor']) - min_pf) / pf_range) * 100
        # Skor: Her metrik %25 ağırlık
        result['combined_score'] = (0.20 * result['total_success_rate'] + 
                                    0.50 * normalized_profit + 
                                    0.15 * normalized_sharpe + 
                                    0.15 * normalized_pf)


    # Sonuçları kombinasyon skoruna göre sırala
    valid_results.sort(key=lambda x: x['combined_score'], reverse=True)

    # En iyi 20 kombinasyonu yazdır
    print(f"\n=== Başarı, Kâr, Sharpe ve Profit Factor Kombinasyonuna Göre En İyi 20 Setup (Toplam {len(valid_results)} adet) ===")
    for i, result in enumerate(valid_results[:50]):
        combo = result['combination']
        print(f"\n{i + 1}. Kombinasyon (Skor: {result['combined_score']:.2f}):")
        print(f"LEFT: {combo[0]}, RIGHT: {combo[1]}, MANIPULATION_THRESHOLD: {combo[2]}, "
              f"MAX_CANDLES: {combo[3]}, CONSECUTIVE_CANDLES: {combo[4]}, "
              f"MIN_CANDLES_FOR_SECOND_CONDITION: {combo[5]}, MAX_CANDLES_FOR_SECOND_CONDITION: {combo[6]}, "
              f"RISK_REWARD_RATIO: {combo[7]}")
        print(f"Toplam Başarı Oranı: {result['total_success_rate']:.2f}%")
        print(f"Ortalama Aylık Başarı Oranı: {result['avg_success_rate']:.2f}%")
        print(f"Minimum Aylık Başarı Oranı: {result['min_success_rate']:.2f}%")
        print(f"Ağırlıklı Kâr: {result['weighted_profit']:.2f} USD")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Profit Factor: {result['profit_factor']:.2f}")

    # Toplam süreyi yazdır
    total_time = time.time() - start_time
    print(f"\nToplam optimizasyon süresi: {total_time:.2f} saniye")

if __name__ == "__main__":
    run_optimization()