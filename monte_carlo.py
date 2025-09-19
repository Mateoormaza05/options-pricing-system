import numpy as np

def mc_pricing(S, K, T, r, sigma, option_type, n, q=0):
    """
    Calculate option price with Monte Carlo methods
     
    Parameters:
        S (float): current stock price 
        K (float): strike price
        T (float): time to expiration in years 
        r (float): risk-free interest rate
        sigma (float): volatility
        option_type (str) : "call" or "put"
        n (int): number of simulations
        q (float): dividend yield (annualized dividend rate as percentage of stock price)

    Returns:
        option price (float): fair call/put option price based on n simulations
    """
   
    # Randomly generate n numbers (shock factors) using normal distribution
    Z = np.random.standard_normal(n)

    # Calculate stock prices using Geomatric Brownian Motion
    final_stock_price = S * np.exp((r - q - 0.5 * (sigma ** 2)) * T + sigma * np.sqrt(T) * Z)

    # Calculate payoff for calls and puts (0 if option is OTM)
    if option_type.lower() == 'call':
        option_payoff = np.maximum(final_stock_price - K, 0)
    elif option_type.lower() == 'put':
        option_payoff = np.maximum(K - final_stock_price, 0)
    
    # Calculate fair price by taking mean of payoffs and adjusting
    mc_price = np.exp(-r * T) * np.mean(option_payoff)
    return mc_price


# Calculate Monte Carlo prices for increasing simulation (n) values
def mc_convergence(S, K, T, r, sigma, option_type, q, n_values):
    prices = []
    for sim_num in n_values:
        price = mc_pricing(S, K, T, r, sigma, option_type, sim_num, q=q)
        prices.append(float(price))
    return prices

# Test call
if __name__ == "__main__":
    S = 150
    K = 175
    T = .5
    r = .0426
    sigma = .25
    n = 10000


    call = mc_pricing(S, K, T, r, sigma, option_type="call", n=10000, q=0)
    put = mc_pricing(S, K, T, r, sigma, option_type="put", n=10000, q=0)

    print(f"Call option price: {call}")
    print(f"Put option price: {put}")



    
