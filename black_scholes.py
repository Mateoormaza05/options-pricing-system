import numpy as np
from scipy.stats import norm
from greeks import bs_greeks

def bs_pricing(S, K, T, r, sigma, option_type, q=0):
    """
    Calculate option price with Black-Scholes model

    Parameters:
        S (float): current stock price 
        K (float): strike price
        T (float): time to expiration in years 
        r (float): risk-free interest rate
        sigma (float): volatility
        option_type (str): type of option ("call" or "put")
        q (float): dividend yield (annualized dividend rate as percentage of stock price)

    Returns:
       option price (float): fair option price determined by the model
    """
   
    # Helpers for Black-Scholes formula

    d1 = (np.log(S/K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Black-Scholes formula for both call and put options

    if  option_type.lower() == "call":
        # call option price formula: S*N(d1) - K*e^(-rT)*N(d2)
        bs_price = S * np.exp(-q * T) * norm.cdf(d1) - K * (np.exp(-r * T)) * norm.cdf(d2)
    elif option_type.lower() == "put":
        # put option price formula: K*e^(-rT)*N(-d2) - S*N(-d1)
        bs_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return bs_price

def implied_volatility(market_price, S, K, T, r, option_type, error_tolerance=1e-4, max_iter=100, q=0):
    """
    Calculate Implied Volatility (forward looking measure of volatility)

    Parameters:
        market_price (float): current market option price
        S (float): current stock price 
        K (float): strike price
        T (float): time to expiration in years 
        r (float): risk-free interest rate
        option_type (str): type of option ("call" or "put")
        error_tolerance (float): acceptable difference between model and market prices
        max_iter (int): maximum number of iterations of Newton-Raphson Algorithm
        q (float): dividend yield (annualized dividend rate as percentage of stock price)

    Returns:
        Implied Volatility (float): market forecast of likely sigma (price volatility)
    """
    
    sigma = 0.3
    
    # Newton-Raphson Algorithm

    for i in range(max_iter):
        model_price = bs_pricing(S, K, T, r, sigma, option_type, q)
        vega = bs_greeks(S, K, T, r, sigma, option_type, q)["vega"]
        difference = model_price - market_price

        if abs(difference) < error_tolerance * max(1.0, market_price):
            return sigma
        
        if vega < 1e-8:
            break

        sigma -= difference / vega
        sigma = max(min(sigma, 5.0), 1e-6)
    
    # Backup option if Newton-Raphson fails to converge

    low = 1e-6
    high = 5.0
    tolerance_price = error_tolerance * max(1.0, market_price)

    for i in range(max_iter):
        mid = (low + high) / 2
        price = bs_pricing(S, K, T, r, mid, option_type, q)

        if abs(price  - market_price) < tolerance_price:
            return mid
        
        if price > market_price:
            high = mid
        else:
            low = mid
    
    return None
    
# Test Call

if __name__ == "__main__":
    S = 100       
    K = 100   
    T = 1       
    r = 0.05  
    sigma = 0.2

    
    call = bs_pricing(S, K, T, r, sigma, "call")
    put = bs_pricing(S, K, T, r, sigma, "put")
    iv = implied_volatility(call, S, K, T, r, "call", .001, 1000)

    print(f"Call: {call}")
    print(f"Put: {put}")
    print(f"Implied volatility: {iv}")
