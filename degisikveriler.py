from binance import Client
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Binance client'ı başlat (API anahtarı gerekmiyorsa boş bırakılabilir)
client = Client()

# İstanbul saat dilimini ayarla (UTC+3)
istanbul_tz = pytz.timezone('Europe/Istanbul')
now_istanbul = datetime.now(istanbul_tz)

# 1. Veri: Bugünden 1 yıl geriye (04 Nisan 2024 - 03 Nisan 2025)
start_time_1 = now_istanbul - timedelta(days=365)
end_time_1 = now_istanbul

# 2. Veri: 1 yıl öncesinden 2 yıl geriye (04 Nisan 2023 - 03 Nisan 2024)
start_time_2 = now_istanbul - timedelta(days=365 * 2)
end_time_2 = now_istanbul - timedelta(days=365)

# 3. Veri: 2 yıl öncesinden 3 yıl geriye (04 Nisan 2022 - 03 Nisan 2023)
start_time_3 = now_istanbul - timedelta(days=365 * 3)
end_time_3 = now_istanbul - timedelta(days=365 * 2)

# Zaman damgalarını milisaniye cinsinden UTC'ye çevir
time_periods = [
    (start_time_1, end_time_1, "btc_usdt_15m_1yil_geri.csv"),
    (start_time_2, end_time_2, "btc_usdt_15m_2yil_geri.csv"),
    (start_time_3, end_time_3, "btc_usdt_15m_3yil_geri.csv")
]

for start_time, end_time, filename in time_periods:
    start_timestamp = int(start_time.astimezone(pytz.UTC).timestamp() * 1000)
    end_timestamp = int(end_time.astimezone(pytz.UTC).timestamp() * 1000)

    # BTCUSDT Perpetual Futures için 15 dakikalık klines verilerini indir
    klines = client.futures_historical_klines(
        symbol="BTCUSDT",           # BTC/USDT Perpetual kontrat sembolü
        interval="15m",             # 15 dakikalık zaman dilimi
        start_str=start_timestamp,  # Başlangıç zamanı (UTC)
        end_str=end_timestamp       # Bitiş zamanı (UTC)
    )

    # Ver* Veriyi DataFrame'e dönüştür
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    # Zaman damgalarını İstanbul saat dilimine çevir
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(istanbul_tz)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(istanbul_tz)

    # Veriyi CSV dosyasına kaydet
    df.to_csv(filename, index=False)
    print(f"Veriler '{filename}' dosyasına {now_istanbul.strftime('%Y-%m-%d %H:%M:%S')} İstanbul saati baz alınarak kaydedildi.")

print("Tüm veriler başarıyla indirildi ve kaydedildi.")