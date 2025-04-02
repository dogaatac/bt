# config.py

# Veri dosyası ayarları
DATA_FILE = "btc_usdt_15m-3yil.csv"

# Pivot hesaplama parametreleri
LEFT = 30  # Sol periyot
RIGHT = 20  # Sağ periyot

# Manipülasyon ve işlem parametreleri
MANIPULATION_THRESHOLD = 0.003  # %0.3 manipülasyon eşiği
MAX_CANDLES = 30  # Maksimum 30 mum
CONSECUTIVE_CANDLES = 4  # En az 4 ardışık mum
MIN_CANDLES_FOR_SECOND_CONDITION = 5  # İkinci koşul için minimum 5 mum
MAX_CANDLES_FOR_SECOND_CONDITION = 20  # İkinci koşul için maksimum 20 mum
RISK_REWARD_RATIO = 1.5  # Risk-Kazanç oranı (örneğin 1:1.5)

# Risk yönetimi
INITIAL_BALANCE = 10000  # USD
MAX_RISK = 0.01  # Sabit miktar için varsayılan %1 risk (eski 100 USD yerine %1)
RISK_TYPE = 'fixed'  # Varsayılan olarak sabit miktar
RISK_PERCENTAGE = 0.01  # Kasa yüzdesi için varsayılan %1 (kullanıcı değiştirebilir)

# Grafik ayarları
PLOT_CANDLES_BEFORE = 10  # Sweep’ten önceki mum sayısı
PLOT_CANDLES_AFTER = 5  # Çıkıştan sonraki mum sayısı
PLOT_FIGSCALE = 1.5  # Grafik boyutu
PLOT_NUM_TRADES = 10  # Grafiklenecek işlem sayısı