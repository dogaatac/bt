from binance import Client
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Binance client'ı başlat (API anahtarı gerekmiyorsa boş bırakılabilir)
client = Client()

# İstanbul saat dilimini ayarla (UTC+3)
istanbul_tz = pytz.timezone('Europe/Istanbul')
now_istanbul = datetime.now(istanbul_tz)

# Başlangıç ve bitiş zaman damgalarını hesapla (son 3 ay)
start_time = now_istanbul - timedelta(days=90)
end_time = now_istanbul

# Zaman damgalarını milisaniye cinsinden UTC'ye çevir
start_timestamp = int(start_time.astimezone(pytz.UTC).timestamp() * 1000)
end_timestamp = int(end_time.astimezone(pytz.UTC).timestamp() * 1000)

# BTCUSDT Perpetual Futures için 15 dakikalık klines verilerini indir
klines = client.futures_historical_klines(

    symbol="BTCUSDT",           # BTC/USDT Perpetual kontrat sembolü
    interval="15m",             # 15 dakikalık zaman dilimi
    start_str=start_timestamp,  # Başlangıç zamanı (UTC)
    end_str=end_timestamp       # Bitiş zamanı (UTC)
)

# Veriyi DataFrame'e dönüştür
df = pd.DataFrame(klines, columns=[
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
])

# Zaman damgalarını İstanbul saat dilimine çevir
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(istanbul_tz)
df['close_time'] = pd.to_datetime(df['close_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(istanbul_tz)

# Veriyi CSV dosyasına kaydet
df.to_csv("sol_usdt_15.csv", index=False)
print(f"Veriler 'btc_usdt_15m-3yil.csv' dosyasına {now_istanbul.strftime('%Y-%m-%d %H:%M:%S')} İstanbul saati baz alınarak kaydedildi.")