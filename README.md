# Options Pricing and Greeks Calculator
Implements Black-Scholes and Monte Carlo option pricing models with risk Greeks and real market data integration using Python.

# What this Project Does
This project uses the Black-Scholes and Monte Carlo models to calculate fair option prices. It also implements option Greeks to convey risk sensitivities for options, as well as an Implied Volatility calculator to implement a more realistic non-constant price volatility value. I built this project to gain perspective on how theoretical pricing works in quantitative finance. 

# Key Features

Black-Scholes Option Pricing - Analytical option price calculation with dividends included.

Monte Carlo Simulation Option Pricing - Numerical option price calculation with repeated simulations.

Greeks Calculation - Analytical greeks for Black-Scholes model and numerical greeks for Monte Carlo model to identify option risks.

Implied Volatility - Newton-Raphson method to calculate forward-looking volatility (with bisection method fallback for failing convergence).

Convergence Analysis - Testing Monte Carlo pricing method with increasing simulation values (n).

Mock Data - Visualizing option delta and implied volatility using mock AMZN data.

# Quick Example

```python
from black_scholes import bs_pricing, implied_volatility
from greeks import bs_greeks

# Price a call option
call_price = bs_pricing(S=100, K=105, T=0.25, r=0.05, sigma=0.2, option_type="call")
print(f"Call price: {call_price}")

# Calculate option Greeks
greeks = bs_greeks(S=100, K=105, T=0.25, r=0.05, sigma=0.2, option_type="call")
print(f"Delta: {greeks['delta']}")
print(f"Gamma: {greeks['gamma']}")

# Calculate implied volatility
iv = implied_volatility(market_price=3.0, S=100, K=105, T=0.25, r=0.05, option_type="call")
print(f"Implied Volatility: {iv}")
```

# How to Run

```bash
# Install dependencies
pip install numpy scipy matplotlib pandas yfinance

# Run the main demonstration
python main.py
```

# What I learned

Black-Scholes Method - Exposing myself to the Black-Scholes PDE was intimidating yet fascinating. Building a mathematical understanding of this model helped me understand the applications of my mathematics experience. In particular, I enjoyed finding the partial derivatives of the model in order to find the Greeks. 

Monte Carlo Method - Building this simulation system helped me build a practical understanding of many statistical and mathematical principles, including the Law of Large Numbers. It also helped me understand the programming difficulties involved with quantitative finance and big data in general. A simple for-loop was not practical here, as I needed to create a NumPy array to keep track of each simulation.

Implied Volatility - I found the concept of IV to be interesting, particularly because models like the Black-Scholes models must always make assumptions. Constant price volatility/sigma is an unrealistic yet necessary assumption of the model. Implementing IV into my project helped me understand both the limitations and extensions of theoretical models in finance and beyond.

Option Greeks - Stemming from my enjoyment of calculus, I found the derivation and approximation of my risk sensitivities to extend my foundational understanding of partial derivatives.

Applied Programming - After building a fundamental understanding of Python, this project exposed me to real bugs, syntax issues, and quantitative finance topics. I also learned the basics of matplotlib.pyplot, datetime, scipy.stats, as well as built upon my math, NumPy and pandas skills.

# Technical Approach

Black-Scholes Model: Implements the traditional option pricing formula for European-style options using stock price, strike price, time to expiration, volatility, option type, and dividend yield.

Monte Carlo Model: Generates n stochastic/random underlying stock prices and averages payoffs to estimate a fair value for the option. 

Greeks: Uses analytical and numerical derivatives to find option price sensitivities and aid risk management. 

# Project Files
'black_scholes.py' - Black-Scholes pricing model and Implied Volatility calculator

'monte_carlo.py' - Monte Carlo simulation pricing with convergence testing

'greeks.py' - Option Greeks using analytical (for Black-Scholes) and numerical (for Monte Carlo) methods

'visualizations.py' - Market data visualizations using mock market data

'main.py' - Main demonstration script

---

**Contact**: mormaza05@icloud.com | **LinkedIn**: Mateo Ormaza
