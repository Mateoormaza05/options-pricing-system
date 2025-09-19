import black_scholes
import monte_carlo
import greeks
import visualizations

"""
Options Pricing and Greeks System 
Black-Scholes, Monte Carlo, and Greeks implementations with real market analysis 
"""
S = 100
K = 115
T = 0.5
r = 0.05
sigma = 0.2

print(f"\nOption Parameters:")
print(f"Stock Price: ${S}")
print(f"Strike Price: ${K}")
print(f"Time to Expiry: {T} years")
print(f"Volatility: {sigma:.1%}")

print(f"\nPricing Results:")

bs_call = black_scholes.bs_pricing(S, K, T, r, sigma, "call")
mc_call = monte_carlo.mc_pricing(S, K, T, r, sigma, "call", n=10000)
bs_put = black_scholes.bs_pricing(S, K, T, r, sigma, "put")
mc_put = monte_carlo.mc_pricing(S, K, T, r, sigma, "put", n=10000)

print(f"BS Call Option Price: {bs_call}")
print(f"MC Call Option Price: {mc_call}")
print(f"BS Put Option Price: {bs_put}")
print(f"MC Put Option Price: {mc_put}")

greeks1 = greeks.bs_greeks(S, K, T, r, sigma, "call")
greeks2 = greeks.mc_greeks(S, K, T, r, sigma, "call", n=10000, h=0.01, dt=0.01)
print(f"\nBS Option Greeks:")
for greek, value in greeks1.items():
    print(f"{greek}: {value}")
print(f"MC Option Greeks:")
for greek, value in greeks2.items():
    print(f"{greek}: {value}")

iv = black_scholes.implied_volatility(5, S, K, T, r, "call", error_tolerance=1e-4, max_iter=100, q=0)
print(f"\nImplied Volatility: {iv}")

nlist = [10, 100, 250, 500, 1000, 5000, 7500, 10000]
convergence = monte_carlo.mc_convergence(S, K, T, r, sigma, "call", q=0, n_values=nlist)
print(f"\nSequence of Converging Monte Carlo Option Prices: {convergence}")

visualizations.call_visualizations()


