# main.py
import config
from engine import run_engine, print_trades, print_summary
from plotter import plot_trades
from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Varsayılan başlangıç kasası (config'den alınır, yoksa sabit bir değer)
INITIAL_CAPITAL = getattr(config, 'INITIAL_BALANCE', 10000)  # config.py'dan alır, yoksa 10,000 USD

def plot_equity_curve(trades, initial_capital):
    """Kasa eğrisini grafik olarak çizer."""
    if not trades:
        print("Grafik çizilecek trade verisi yok.")
        return

    # Trade verilerini DataFrame'e çevir
    trade_df = pd.DataFrame(trades)
    trade_df['exit_time'] = pd.to_datetime(trade_df['exit_time']).dt.tz_localize(None)

    # Kümülatif kâr/zarar hesaplama
    trade_df['cumulative_profit'] = trade_df['profit'].cumsum()
    trade_df['equity'] = initial_capital + trade_df['cumulative_profit']

    # Grafik çizimi
    plt.figure(figsize=(12, 6))
    plt.plot(trade_df['exit_time'], trade_df['equity'], label='Kasa (Equity)', color='blue')
    plt.axhline(y=initial_capital, color='gray', linestyle='--', label='Başlangıç Kasası')
    plt.title('Kasa Zaman İçindeki Değişimi')
    plt.xlabel('Zaman')
    plt.ylabel('Kasa (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def get_user_input():
    # Grafik basılıp basılmayacağı
    while True:
        plot_choice = input("Grafikler basılsın mı? (e/h): ").lower()
        if plot_choice in ['e', 'h']:
            plot_enabled = (plot_choice == 'e')
            break
        print("Lütfen 'e' (evet) veya 'h' (hayır) girin.")

    # Kasa eğrisi grafiği
    while True:
        equity_plot_choice = input("Kasa eğrisi grafiği basılsın mı? (e/h): ").lower()
        if equity_plot_choice in ['e', 'h']:
            equity_plot_enabled = (equity_plot_choice == 'e')
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

    # Sabit miktar mı, kasa yüzdesi mi?
    while True:
        risk_type = input("Her trade'de sabit miktar mı yoksa kasa yüzdesi mi kullanılsın? (sabit/yüzde): ").lower()
        if risk_type in ['sabit', 'yüzde']:
            break
        print("Lütfen 'sabit' veya 'yüzde' girin.")

    if risk_type == 'yüzde':
        while True:
            try:
                risk_percentage = float(input("Kasa yüzdesi olarak risk oranı (%): "))
                if 0 < risk_percentage <= 100:
                    config.RISK_TYPE = 'percentage'
                    config.RISK_PERCENTAGE = risk_percentage / 100  # Yüzdeyi ondalık sayıya çevir
                    break
                print("Lütfen 0 ile 100 arasında bir değer girin.")
            except ValueError:
                print("Lütfen geçerli bir sayı girin.")
    else:
        config.RISK_TYPE = 'fixed'

    # Print biçimi
    while True:
        print_format = input("Print biçimi (detailed/simple/grafik): ").lower()
        if print_format in ['detailed', 'simple', 'grafik']:
            break
        print("Lütfen 'detailed', 'simple' veya 'grafik' girin.")

    return plot_enabled, equity_plot_enabled, table_enabled, print_format

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
        
        monthly_stats.loc[monthly_stats['Ay'] == month, 'Başarı Oranı (%)'] = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        monthly_stats.loc[monthly_stats['Ay'] == month, 'TP İşlem'] = tp_trades
        monthly_stats.loc[monthly_stats['Ay'] == month, 'SL İşlem'] = sl_trades
    
    # Tabloyu formatla
    monthly_stats['Kâr/Zarar (USD)'] = monthly_stats['Kâr/Zarar (USD)'].round(2)
    monthly_stats['Başarı Oranı (%)'] = monthly_stats['Başarı Oranı (%)'].round(2)
    monthly_stats['Ay'] = monthly_stats['Ay'].astype(str)  # Dönemleri string'e çevir
    
    # Tüm ayların toplam ve ortalamalarını hesapla
    total_trades = monthly_stats['Trade Sayısı'].sum()
    total_profit = monthly_stats['Kâr/Zarar (USD)'].sum()
    total_success_rate = (len(trade_data[trade_data['profit'] > 0]) / len(trade_data) * 100) if len(trade_data) > 0 else 0
    total_tp = monthly_stats['TP İşlem'].sum()
    total_sl = monthly_stats['SL İşlem'].sum()

    # Sharpe Ratio hesaplama
    monthly_profits = trade_data.groupby('month')['profit'].sum()
    mean_monthly_profit = monthly_profits.mean()
    std_monthly_profit = monthly_profits.std()
    sharpe_ratio = mean_monthly_profit / std_monthly_profit if std_monthly_profit != 0 else 0

    # Profit Factor hesaplama
    gross_profit = trade_data[trade_data['profit'] > 0]['profit'].sum()
    gross_loss = abs(trade_data[trade_data['profit'] < 0]['profit'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

    # Özet satırını ekle
    summary_row = pd.DataFrame({
        'Ay': ['Toplam/Ort'],
        'Trade Sayısı': [total_trades],
        'Kâr/Zarar (USD)': [total_profit],
        'Başarı Oranı (%)': [total_success_rate],
        'TP İşlem': [total_tp],
        'SL İşlem': [total_sl]
    })
    monthly_stats = pd.concat([monthly_stats, summary_row], ignore_index=True)
    
    # Tabulate ile tabloyu ekrana yazdır
    table = tabulate(
        monthly_stats,
        headers=['Ay', 'Trade Sayısı', 'Kâr/Zarar (USD)', 'Başarı Oranı (%)', 'TP İşlem', 'SL İşlem'],
        tablefmt='simple',
        floatfmt=".2f",
        numalign="right",
        stralign="left"
    )
    print("\nAylık Backtest Sonuçları:")
    print(table)
    
    # Sharpe Ratio ve Profit Factor'ü tablodan sonra yazdır
    print(f"\nSharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    # Tabloyu CSV dosyasına kaydet
    monthly_stats.to_csv('monthly_backtest_results.csv', index=False)
    print("\nTablo 'monthly_backtest_results.csv' dosyasına kaydedildi.")

def main():
    # Kullanıcıdan giriş al
    plot_enabled, equity_plot_enabled, table_enabled, print_format = get_user_input()

    # Algoritmayı çalıştır
    df, trades, final_balance = run_engine(config)

    # İşlem detaylarını yazdır
    if print_format == 'grafik':
        plot_equity_curve(trades, config.INITIAL_BALANCE)
    else:
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

    # Kasa eğrisi grafiğini çiz (eğer kullanıcı isterse)
    if equity_plot_enabled:
        print("\nKasa eğrisi grafiği çiziliyor...")
        plot_equity_curve(trades, config.INITIAL_BALANCE)
        print("Kasa eğrisi grafiği tamamlandı!")

if __name__ == "__main__":
    main()