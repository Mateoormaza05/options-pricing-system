import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from black_scholes import bs_pricing, implied_volatility
from monte_carlo import mc_pricing
from greeks import bs_greeks

def option_delta_visualization():
    stock_price = 232.50
    option_dates = ["2025-10-18", "2025-11-15", "2025-12-20", "2026-01-17", "2026-03-20"]
    times = []
    strikes = []
    deltas = []
    
    # Market option prices
    option_prices = {
        (200, 30): 33.2, (210, 30): 24.8, (220, 30): 17.1, (230, 30): 11.2, (240, 30): 6.8, (250, 30): 3.9, (260, 30): 2.1,
        (200, 45): 35.6, (210, 45): 27.9, (220, 45): 20.8, (230, 45): 15.1, (240, 45): 10.7, (250, 45): 7.4, (260, 45): 5.0,
        (200, 60): 37.8, (210, 60): 30.5, (220, 60): 24.0, (230, 60): 18.6, (240, 60): 14.2, (250, 60): 10.8, (260, 60): 8.1,
        (200, 90): 40.9, (210, 90): 34.2, (220, 90): 28.1, (230, 90): 22.8, (240, 90): 18.4, (250, 90): 14.9, (260, 90): 12.0,
        (200, 180): 46.5, (210, 180): 40.8, (220, 180): 35.7, (230, 180): 31.2, (240, 180): 27.1, (250, 180): 23.6, (260, 180): 20.5
    }
    
    today = datetime.today().date()
    for i, date in enumerate(option_dates):
        exp = datetime.strptime(date, "%Y-%m-%d").date()
        T = (exp - today).days / 252
        
        for strike in [200, 210, 220, 230, 240, 250, 260]:
            days_key = [30, 45, 60, 90, 180][i]
            if (strike, days_key) in option_prices:
                market_price = option_prices[(strike, days_key)]
                iv = implied_volatility(market_price, stock_price, strike, T, 0.0422, "call")
                if iv:
                    delta = bs_greeks(stock_price, strike, T, 0.0422, iv, "call", q=0)["delta"]
                    strikes.append(strike)
                    deltas.append(delta)
                    times.append(T)

    plt.figure(figsize=(9,6))
    scat_plot = plt.scatter(strikes, times, c=deltas, cmap="viridis", s=[d*100 for d in deltas], edgecolor="k", alpha=0.8)
    plt.colorbar(scat_plot, label="Delta")
    plt.xlabel("Strike Price ($)")
    plt.ylabel("Time to Expiration (Years)")
    plt.title("AMZN Call Option Delta Scatter Heatmap")
    plt.tight_layout()
    plt.show()

def iv_visualization():
    stock_price = 232.50
    option_dates = ["2025-10-18", "2025-11-15", "2025-12-20", "2026-01-17", "2026-03-20"]
    times = []
    strikes = []
    ivs = []

    # Market option prices for IV calculation
    option_prices = {
        (200, 30): 33.2, (205, 30): 29.1, (210, 30): 24.8, (215, 30): 20.9, (220, 30): 17.1, (225, 30): 13.8, (230, 30): 11.2, (235, 30): 8.9, (240, 30): 6.8, (245, 30): 5.2, (250, 30): 3.9, (255, 30): 2.9, (260, 30): 2.1,
        (200, 45): 35.6, (205, 45): 31.8, (210, 45): 27.9, (215, 45): 24.2, (220, 45): 20.8, (225, 45): 17.8, (230, 45): 15.1, (235, 45): 12.7, (240, 45): 10.7, (245, 45): 8.9, (250, 45): 7.4, (255, 45): 6.1, (260, 45): 5.0,
        (200, 60): 37.8, (205, 60): 34.2, (210, 60): 30.5, (215, 60): 27.1, (220, 60): 24.0, (225, 60): 21.1, (230, 60): 18.6, (235, 60): 16.2, (240, 60): 14.2, (245, 60): 12.3, (250, 60): 10.8, (255, 60): 9.4, (260, 60): 8.1,
        (200, 90): 40.9, (205, 90): 37.5, (210, 90): 34.2, (215, 90): 31.1, (220, 90): 28.1, (225, 90): 25.4, (230, 90): 22.8, (235, 90): 20.5, (240, 90): 18.4, (245, 90): 16.4, (250, 90): 14.9, (255, 90): 13.4, (260, 90): 12.0,
        (200, 180): 46.5, (205, 180): 43.6, (210, 180): 40.8, (215, 180): 38.2, (220, 180): 35.7, (225, 180): 33.4, (230, 180): 31.2, (235, 180): 29.1, (240, 180): 27.1, (245, 180): 25.3, (250, 180): 23.6, (255, 180): 22.0, (260, 180): 20.5
    }

    today = datetime.today().date()
    for i, date in enumerate(option_dates):
        exp = datetime.strptime(date, "%Y-%m-%d").date()
        T = (exp - today).days / 252
        
        days_key = [30, 45, 60, 90, 180][i]
        for strike in [200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260]:
            if (strike, days_key) in option_prices:
                market_price = option_prices[(strike, days_key)]
                iv = implied_volatility(market_price, stock_price, strike, T, 0.0422, "call")
                if iv:
                    strikes.append(strike)
                    times.append(T)
                    ivs.append(iv)

    df = pd.DataFrame({
        'time': times,
        'strike': strikes,
        'iv': ivs
    })

    grid = df.pivot(index='time', columns='strike', values='iv')

    plt.figure(figsize=(9, 6))
    heatmap = plt.imshow(grid.values, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(heatmap, label='Implied Volatility')
    plt.xlabel('Strike Price ($)')
    plt.ylabel('Time to Expiration (Years)')
    plt.title('AMZN Implied Volatility Surface Heatmap')
    plt.tight_layout()
    plt.show()

def model_comp_visualization():
    scenarios = ["At the Money", "Out of the Money", "In the Money"]
    bs_prices = []
    mc_prices = []
    option_scenarios = [
        {"S": 100, "K": 100, "T": 0.25, "r": 0.05, "sigma": 0.2},
        {"S": 100, "K": 110, "T": 0.5, "r": 0.05, "sigma": 0.3},
        {"S": 100, "K": 90, "T": 1.0, "r": 0.05, "sigma": 0.15},
    ]

    for scenario in option_scenarios:
        bs_price = bs_pricing(**scenario, option_type="call")
        mc_price = mc_pricing(**scenario, option_type="call", n=10000)

        bs_prices.append(bs_price)
        mc_prices.append(mc_price)

    x = range(len(scenarios))
    width = 0.3
    colors = plt.cm.viridis([0.25, 0.75])

    plt.figure(figsize=(9, 6))
    plt.bar(x, bs_prices, width, label="Black-Scholes", color=colors[0])
    plt.bar([i + width for i in x], mc_prices, width, label="Monte Carlo", color=colors[1])
    plt.xticks([i + width / 2 for i in x], scenarios)
    plt.xlabel("Scenarios")
    plt.ylabel("Option Price ($)")
    plt.legend()
    plt.show()

def call_visualizations():
    option_delta_visualization()
    iv_visualization()
    model_comp_visualization()

if __name__ == "__main__":
    call_visualizations()