import numpy as np
from scipy.stats import norm
import math

# Helper functions for Black-Scholes

def d_1(S, K, T, r, sigma, q=0):
    return (math.log(S/K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
   
def d_2(S, K, T, r, sigma, q=0):
    return d_1(S, K, T, r, sigma, q) - sigma * math.sqrt(T)

# Analytical Greeks for Black-Scholes

def bs_greeks(S, K, T, r, sigma, option_type, q=0):
    """
    Calculate greeks for Black-Scholes options pricing model
    Uses analytical partial derivations of Black-Scholes model
    Parameters:
        S (float): current stock price 
        K (float): strike price
        T (float): time to expiration in years 
        r (float): risk-free interest rate
        sigma (float): volatility
        option_type (str) : "call" or "put"
        q (float): dividend yield (annualized dividend rate as percentage of stock price)

    Returns:
       greek (float): sensitivity of option price to differentiation variable
    """

    # Calculate Greeks using partial derivatives of Black-Scholes formula

    if option_type.lower() == "call":
        delta = math.exp(-q * T) * norm.cdf(d_1(S, K, T, r, sigma, q))
        theta = -(S * norm.pdf(d_1(S, K, T, r, sigma, q)) * sigma * math.exp(-q * T) / (2 * math.sqrt(T)) + q * S * norm.cdf(d_1(S, K, T, r, sigma, q)) * math.exp(-q * T) - r * K * math.exp(-r * T) * norm.cdf(d_2(S, K, T, r, sigma, q)))
        rho = K * T * math.exp(-r * T) * norm.cdf(d_2(S, K, T, r, sigma, q))
    elif option_type.lower() == "put":
        delta = math.exp(-q * T) * (norm.cdf(d_1(S, K, T, r, sigma, q)) - 1)
        theta = -(S * norm.pdf(d_1(S, K, T, r, sigma, q)) * sigma * math.exp(-q * T) / (2 * math.sqrt(T)) - q * S * norm.cdf(-d_1(S, K, T, r, sigma, q)) * math.exp(-q * T) + r * K * math.exp(-r * T) * norm.cdf(-d_2(S, K, T, r, sigma, q)))
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d_2(S, K, T, r, sigma, q))

    gamma = math.exp(-q * T) * norm.pdf(d_1(S, K, T, r, sigma, q)) / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * norm.pdf(d_1(S, K, T, r, sigma, q)) * math.sqrt(T) / 100

    # Sort Greeks into dictionary

    greeks = {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho
    }
    
    return greeks
    

def mc_greeks(S, K, T, r, sigma, option_type, n, h, dt, q=0):
    """
    Calculate greeks for monte carlo options pricing model
    Uses pathwise / finite difference methods
     
    Parameters:
        S (float): current stock price 
        K (float): strike price
        T (float): time to expiration in years 
        r (float): risk-free interest rate
        sigma (float): volatility
        option_type (str) : "call" or "put"
        n (int): number of simulations
        h: bump size for finite difference (for gamma)
        dt: change in time for finite difference (for theta)
        q (float): dividend yield (annualized dividend rate as percentage of stock price)

    Returns:
       greek (float): sensitivity of option price to differentiation variable
    """
    
    # Randomly generate n numbers (shock factors) using normal distribution
    Z  = np.random.standard_normal(n)

    # Calculate stock prices using Geomatric Brownian Motion
    final_stock_price = S * np.exp((r - q - 0.5 * (sigma ** 2)) * T + sigma * np.sqrt(T) * Z)

    # Calculate payoff for calls and puts (0 if option is OTM)
    if option_type.lower() == "call":
        payoff = np.maximum(final_stock_price - K, 0)
    elif option_type.lower() == "put":
        payoff = np.maximum(K - final_stock_price, 0)
   
    # Calculate fair price by taking mean of payoffs and adjusting
    option_price = np.exp(-r * T) * np.mean(payoff)

    # Calculate Greeks using pathwise derivatves of Monte Carlo simulations
    # Pathwise derivative = We simulate using Monte Carlo and see how our option price moves for each path with respect to changes in our other variables
    if option_type.lower() == "call":
        delta = np.exp(-r * T) * np.mean((final_stock_price > K) * final_stock_price / S)
        vega = np.exp(-r * T) * np.mean((final_stock_price > K) * final_stock_price * np.sqrt(T) * Z) / 100
        rho = np.exp(-r * T) * np.mean((final_stock_price > K) * final_stock_price * T)
    elif option_type.lower() == "put":
        delta = np.exp(-r * T) * np.mean((final_stock_price < K) * (-final_stock_price / S))
        vega = np.exp(-r * T) * np.mean((final_stock_price < K) * (-final_stock_price * np.sqrt(T) * Z)) / 100
        rho = np.exp(-r * T) * np.mean((final_stock_price < K) * (-final_stock_price * T))
    
    # Finite difference methods for Gamma
    stock_price_up = (S + h) * np.exp((r - q - 0.5 * (sigma ** 2)) * T + sigma * np.sqrt(T) * Z)
    stock_price_down = (S - h) * np.exp((r - q - 0.5 * (sigma ** 2)) * T + sigma * np.sqrt(T) * Z)

    if option_type.lower() == "call":
        payoff_up = np.maximum(stock_price_up - K, 0)
        payoff_down = np.maximum(stock_price_down - K, 0)
    elif option_type.lower() == "put":
        payoff_up = np.maximum(K - stock_price_up, 0)
        payoff_down = np.maximum(K - stock_price_down, 0)
    
    option_price_up = np.exp(-r * T) * np.mean(payoff_up)
    option_price_down = np.exp(-r * T) * np.mean(payoff_down)
    
    gamma = (option_price_up - 2 * option_price + option_price_down) / (h ** 2)

    # Finite difference methods for Theta
    time_dt = max(T - dt, 1e-8)
    final_price_dt = S * np.exp((r - q - 0.5 * (sigma ** 2)) * time_dt + sigma * np.sqrt(time_dt) * Z)
    
    if option_type.lower() == "call":
        payoff_dt = np.maximum(final_price_dt - K, 0)
    elif option_type.lower() == "put":
        payoff_dt = np.maximum(K - final_price_dt, 0)
    price_dt = np.exp(-r * time_dt) * np.mean(payoff_dt)

    theta = (price_dt - option_price) / dt

    # Sort Greeks into dictionary
    greeks = {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho
    }

    return greeks

# Test call

if __name__ == "__main__":
    S = 150
    K = 175
    T = 0.5
    r = .0426
    sigma = 0.25
    n = 10000

    mc_dict =  mc_greeks(S, K, T, r, sigma, option_type="call", n=n, h=0.01, dt=1/365, q=0)
    bs_dict = bs_greeks(S, K, T, r, sigma, option_type="call", q=0)

    print("Monte Carlo Greeks:")
    for greek, value in mc_dict.items():
        print(f"{greek}: {value}")
    print("Black-Scholes Greeks:")
    for greek, value in bs_dict.items():
         print(f"{greek}: {value}")

    






