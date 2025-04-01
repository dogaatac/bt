# main.py
import config
from engine import run_engine, print_trades, print_summary
from plotter import plot_trades
from tabulate import tabulate
import pandas as pd

def get_user_input():
    # Grafik basılıp basılmayacağı
    while True:
        plot_choice = input("Grafikler basılsın mı? (e/h): ").lower()
        if plot_choice in ['e', 'h']:
            plot_enabled = (plot_choice == 'e')
            break
        print("Lütfen 'e' (evet) veya 'h' (hayır) girin.")

    # Tablo yapılsın mı
    while True:
        table_choice = input("Aylık sonuçlar tablo şeklinde gösterilsin mi? (e/h): ").lower()
        if table_choice in ['e', 'h']:
            table_enabled = (table_choice == 'e')
            break
        print("Lütfen 'e' (evet) veya 'h' (hayır) girin.")

    # Temel ayarlar
    print("\nTemel ayarları yapılandırmak ister misiniz? (Varsayılan ayarlar config.py'dan alınır)")
    while True:
        config_choice = input("Ayarları değiştirmek ister misiniz? (e/h): ").lower()
        if config_choice in ['e', 'h']:
            config_enabled = (config_choice == 'e')
            break
        print("Lütfen 'e' (evet) veya 'h' (hayır) girin.")

    if config_enabled:
        config.LEFT = int(input(f"Pivot sol periyot (varsayılan: {config.LEFT}): ") or config.LEFT)
        config.RIGHT = int(input(f"Pivot sağ periyot (varsayılan: {config.RIGHT}): ") or config.RIGHT)
        config.MANIPULATION_THRESHOLD = float(input(f"Manipülasyon eşiği (varsayılan: {config.MANIPULATION_THRESHOLD}): ") or config.MANIPULATION_THRESHOLD)
        config.MAX_CANDLES = int(input(f"Maksimum mum sayısı (varsayılan: {config.MAX_CANDLES}): ") or config.MAX_CANDLES)
        config.CONSECUTIVE_CANDLES = int(input(f"Ardışık mum sayısı (varsayılan: {config.CONSECUTIVE_CANDLES}): ") or config.CONSECUTIVE_CANDLES)
        config.MIN_CANDLES_FOR_SECOND_CONDITION = int(input(f"İkinci koşul için minimum mum (varsayılan: {config.MIN_CANDLES_FOR_SECOND_CONDITION}): ") or config.MIN_CANDLES_FOR_SECOND_CONDITION)
        config.MAX_CANDLES_FOR_SECOND_CONDITION = int(input(f"İkinci koşul için maksimum mum (varsayılan: {config.MAX_CANDLES_FOR_SECOND_CONDITION}): ") or config.MAX_CANDLES_FOR_SECOND_CONDITION)
        config.RISK_REWARD_RATIO = float(input(f"Risk-Kazanç oranı (varsayılan: {config.RISK_REWARD_RATIO}): ") or config.RISK_REWARD_RATIO)
        config.INITIAL_BALANCE = float(input(f"Başlangıç bakiyesi (varsayılan: {config.INITIAL_BALANCE}): ") or config.INITIAL_BALANCE)
        config.MAX_RISK = float(input(f"Maksimum risk (varsayılan: {config.MAX_RISK}): ") or config.MAX_RISK)

    # Print biçimi
    while True:
        print_format = input("Print biçimi (detailed/simple): ").lower()
        if print_format in ['detailed', 'simple']:
            break
        print("Lütfen 'detailed' veya 'simple' girin.")

    return plot_enabled, table_enabled, print_format

def create_monthly_table(trades):
    if not trades:
        print("\nHiç işlem bulunamadı, tablo oluşturulamıyor.")
        return

    # İşlemleri DataFrame'e çevir
    trade_data = pd.DataFrame(trades)
    trade_data['exit_time'] = pd.to_datetime(trade_data['exit_time'])
    
    # Saat dilimi bilgisini kaldırarak ay bazında gruplama
    trade_data['month'] = trade_data['exit_time'].dt.tz_localize(None).dt.to_period('M')
    monthly_stats = trade_data.groupby('month').agg({
        'profit': ['sum', 'count'],
    }).reset_index()
    
    # Sütun isimlerini düzenle
    monthly_stats.columns = ['Ay', 'Kâr/Zarar (USD)', 'Trade Sayısı']
    
    # Başarı oranı, TP ve SL sayıları için ek hesaplamalar
    monthly_stats['Başarı Oranı (%)'] = 0.0
    monthly_stats['TP İşlem'] = 0
    monthly_stats['SL İşlem'] = 0
    
    for month in monthly_stats['Ay']:
        month_trades = trade_data[trade_data['month'] == month]
        total_trades = len(month_trades)
        profitable_trades = len(month_trades[month_trades['profit'] > 0])
        tp_trades = len(month_trades[month_trades['exit_price'] == month_trades['tp']])
        sl_trades = len(month_trades[month_trades['exit_price'] == month_trades['sl']])
        
        # loc ile güvenli atama
        monthly_stats.loc[monthly_stats['Ay'] == month, 'Başarı Oranı (%)'] = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        monthly_stats.loc[monthly_stats['Ay'] == month, 'TP İşlem'] = tp_trades
        monthly_stats.loc[monthly_stats['Ay'] == month, 'SL İşlem'] = sl_trades
    
    # Tabloyu formatla
    monthly_stats['Kâr/Zarar (USD)'] = monthly_stats['Kâr/Zarar (USD)'].round(2)
    monthly_stats['Başarı Oranı (%)'] = monthly_stats['Başarı Oranı (%)'].round(2)
    monthly_stats['Ay'] = monthly_stats['Ay'].astype(str)  # Dönemleri string'e çevir
    
    # Tabulate ile tabloyu ekrana yazdır
    table = tabulate(
        monthly_stats,
        headers=['Ay', 'Trade Sayısı', 'Kâr/Zarar (USD)', 'Başarı Oranı (%)', 'TP İşlem', 'SL İşlem'],
        tablefmt='pretty',
        floatfmt=".2f",
        numalign="right",
        stralign="center"
    )
    print("\nAylık Backtest Sonuçları:")
    print(table)
    
    # Tabloyu CSV dosyasına kaydet
    monthly_stats.to_csv('monthly_backtest_results.csv', index=False)
    print("\nTablo 'monthly_backtest_results.csv' dosyasına kaydedildi.")

def main():
    # Kullanıcıdan giriş al
    plot_enabled, table_enabled, print_format = get_user_input()

    # Algoritmayı çalıştır
    df, trades, final_balance = run_engine(config)

    # İşlem detaylarını yazdır
    print_trades(trades, print_format)

    # Özet bilgileri yazdır
    print_summary(config.INITIAL_BALANCE, final_balance, trades)

    # Aylık tabloyu oluştur ve CSV'ye kaydet (eğer kullanıcı isterse)
    if table_enabled:
        create_monthly_table(trades)

    # Grafikleri çiz (eğer kullanıcı isterse)
    if plot_enabled:
        print("\nGrafikler çiziliyor...")
        plot_trades(df, trades, config)
        print("Grafikler tamamlandı!")

if __name__ == "__main__":
    main()