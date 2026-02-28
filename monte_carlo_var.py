"""
Monte Carlo Simulation for Value at Risk (VaR) Analysis
========================================================
Using:
  - Վdelays (Refinancing Rate) from Central Bank of Armenia
  - USD/AMD Exchange Rate

This script performs:
  1. Data loading & preprocessing
  2. Exploratory analysis & correlation
  3. Monte Carlo simulation of joint dynamics
  4. VaR estimation at 95% and 99% confidence levels
  5. Comprehensive visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.figsize': (14, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  MONTE CARLO SIMULATION – VaR ANALYSIS")
print("  Refinancing Rate & USD/AMD Exchange Rate")
print("=" * 70)

# --- Refinancing Rate ---
df_ref = pd.read_excel('data/refinance rate.xlsx', header=1)
df_ref.columns = ['Date', 'Refinance_Rate', 'Lombard_Repo', 'Cash_Attraction']
df_ref['Date'] = pd.to_datetime(df_ref['Date'])
df_ref = df_ref.sort_values('Date').reset_index(drop=True)
print(f"\n[1] Refinancing Rate: {len(df_ref)} observations "
      f"({df_ref['Date'].min().date()} → {df_ref['Date'].max().date()})")
print(f"    Current rate: {df_ref['Refinance_Rate'].iloc[-1]}%")

# --- USD/AMD Exchange Rate ---
df_fx = pd.read_csv('data/exchange rate.csv')
df_fx.columns = ['Date', 'EUR_AMD', 'RUB_AMD', 'USD_AMD']
df_fx['Date'] = pd.to_datetime(df_fx['Date'], format='%d.%m.%Y')
df_fx['USD_AMD'] = pd.to_numeric(df_fx['USD_AMD'], errors='coerce')
df_fx = df_fx.dropna(subset=['USD_AMD']).sort_values('Date').reset_index(drop=True)
print(f"[2] USD/AMD Rate:     {len(df_fx)} observations "
      f"({df_fx['Date'].min().date()} → {df_fx['Date'].max().date()})")
print(f"    Current rate: {df_fx['USD_AMD'].iloc[-1]} AMD per 1 USD")

# ══════════════════════════════════════════════════════════════════════════════
# 2. CREATE DAILY REFINANCING RATE (forward-fill policy rate)
# ══════════════════════════════════════════════════════════════════════════════

# Refinancing rate changes at policy meetings → forward-fill to daily
date_range = pd.date_range(df_ref['Date'].min(), df_fx['Date'].max(), freq='D')
df_ref_daily = pd.DataFrame({'Date': date_range})
df_ref_daily = df_ref_daily.merge(df_ref[['Date', 'Refinance_Rate']], on='Date', how='left')
df_ref_daily['Refinance_Rate'] = df_ref_daily['Refinance_Rate'].ffill()

# Merge with USD/AMD
df = df_fx[['Date', 'USD_AMD']].merge(df_ref_daily, on='Date', how='inner')
df = df.dropna().sort_values('Date').reset_index(drop=True)

print(f"\n[3] Merged dataset:   {len(df)} daily observations "
      f"({df['Date'].min().date()} → {df['Date'].max().date()})")

# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE RETURNS / CHANGES
# ══════════════════════════════════════════════════════════════════════════════

# Log returns for FX rate
df['FX_LogReturn'] = np.log(df['USD_AMD'] / df['USD_AMD'].shift(1))

# Absolute changes in refinancing rate (bps)
df['Rate_Change_bps'] = (df['Refinance_Rate'] - df['Refinance_Rate'].shift(1)) * 100

df = df.dropna().reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("  DESCRIPTIVE STATISTICS")
print("─" * 70)

fx_returns = df['FX_LogReturn'].values
rate_changes = df['Rate_Change_bps'].values

stats_table = pd.DataFrame({
    'USD/AMD Log Return': [
        fx_returns.mean(), fx_returns.std(),
        stats.skew(fx_returns), stats.kurtosis(fx_returns),
        np.percentile(fx_returns, 1), np.percentile(fx_returns, 5)
    ],
    'Refi Rate Δ (bps)': [
        rate_changes.mean(), rate_changes.std(),
        stats.skew(rate_changes), stats.kurtosis(rate_changes),
        np.percentile(rate_changes, 1), np.percentile(rate_changes, 5)
    ]
}, index=['Mean', 'Std Dev', 'Skewness', 'Kurtosis', '1st Percentile', '5th Percentile'])

print(stats_table.to_string(float_format='{:.6f}'.format))

# Correlation
corr = df[['FX_LogReturn', 'Rate_Change_bps']].corr()
print(f"\nCorrelation (FX log-return vs Rate change): {corr.iloc[0, 1]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("  MONTE CARLO SIMULATION PARAMETERS")
print("─" * 70)

N_SIMULATIONS = 10_000       # number of scenarios
HORIZON_DAYS = 252           # 1-year horizon (trading days)
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]

# Current values
S0_fx = df['USD_AMD'].iloc[-1]       # current FX rate
S0_rate = df['Refinance_Rate'].iloc[-1]  # current refinancing rate (%)

print(f"  Simulations:        {N_SIMULATIONS:,}")
print(f"  Horizon:            {HORIZON_DAYS} trading days (~1 year)")
print(f"  Initial USD/AMD:    {S0_fx}")
print(f"  Initial Refi Rate:  {S0_rate}%")

# --- Estimate parameters from historical data ---
# For FX: use GBM (Geometric Brownian Motion) parameters
mu_fx = fx_returns.mean()
sigma_fx = fx_returns.std()

# For interest rate: use Vasicek-like mean-reversion model
# dr = κ(θ - r)dt + σ_r dW
# Estimate via OLS: Δr = α + β*r_{t-1} + ε
df_rate = df[df['Rate_Change_bps'] != 0].copy()  # only days with changes
# Use full series for rate dynamics
rate_series = df['Refinance_Rate'].values
dr = np.diff(rate_series)
r_lag = rate_series[:-1]

# Filter non-zero changes for mean reversion estimation
mask = dr != 0
if mask.sum() > 5:
    from numpy.polynomial.polynomial import polyfit
    # Simple regression Δr = a + b*r
    b, a = np.polyfit(r_lag[mask], dr[mask], 1)
    kappa = -b  # speed of mean reversion
    theta = -a / b if b != 0 else rate_series.mean()  # long-run mean
    sigma_r = np.std(dr[mask])
else:
    kappa = 0.1
    theta = rate_series.mean()
    sigma_r = 0.25

# Ensure sensible parameters
kappa = max(kappa, 0.01)
theta = max(theta, 1.0)
sigma_r = max(sigma_r, 0.01)

print(f"\n  FX Model (GBM):     μ={mu_fx:.6f}/day, σ={sigma_fx:.6f}/day")
print(f"  Rate Model (Vasicek): κ={kappa:.4f}, θ={theta:.2f}%, σ_r={sigma_r:.4f}")

# Correlation between FX and rate innovations
# Use overlapping periods
valid_mask = (df['FX_LogReturn'] != 0) & (df['Rate_Change_bps'] != 0)
if valid_mask.sum() > 2:
    rho = np.corrcoef(df.loc[valid_mask, 'FX_LogReturn'],
                       df.loc[valid_mask, 'Rate_Change_bps'])[0, 1]
    if np.isnan(rho):
        rho = 0.0
else:
    rho = 0.0

print(f"  Correlation (ρ):    {rho:.4f}")

# --- Cholesky decomposition for correlated shocks ---
cov_matrix = np.array([[1, rho], [rho, 1]])
try:
    L = np.linalg.cholesky(cov_matrix)
except np.linalg.LinAlgError:
    L = np.eye(2)

# --- Run Simulation ---
print("\n  Running Monte Carlo simulation...")
np.random.seed(42)

fx_paths = np.zeros((N_SIMULATIONS, HORIZON_DAYS + 1))
rate_paths = np.zeros((N_SIMULATIONS, HORIZON_DAYS + 1))

fx_paths[:, 0] = S0_fx
rate_paths[:, 0] = S0_rate

for t in range(1, HORIZON_DAYS + 1):
    # Generate correlated normal shocks
    Z = np.random.standard_normal((N_SIMULATIONS, 2))
    Z_corr = Z @ L.T

    z_fx = Z_corr[:, 0]
    z_rate = Z_corr[:, 1]

    # GBM for FX
    fx_paths[:, t] = fx_paths[:, t-1] * np.exp(
        (mu_fx - 0.5 * sigma_fx**2) + sigma_fx * z_fx
    )

    # Vasicek for refinancing rate (discretized)
    rate_paths[:, t] = (rate_paths[:, t-1]
                         + kappa * (theta - rate_paths[:, t-1])
                         + sigma_r * z_rate)
    # Floor at 0%
    rate_paths[:, t] = np.maximum(rate_paths[:, t], 0.0)

print("  ✓ Simulation complete!\n")

# ══════════════════════════════════════════════════════════════════════════════
# 6. VaR CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

print("─" * 70)
print("  VALUE AT RISK (VaR) RESULTS")
print("─" * 70)

# --- FX VaR ---
fx_terminal = fx_paths[:, -1]
fx_pnl = fx_terminal - S0_fx            # absolute P&L
fx_pnl_pct = (fx_terminal / S0_fx - 1) * 100  # percentage P&L

print(f"\n  ▶ USD/AMD Exchange Rate VaR ({HORIZON_DAYS}-day horizon)")
print(f"    Initial rate: {S0_fx:.2f} AMD/USD")
print(f"    Mean terminal rate: {fx_terminal.mean():.2f}")
print(f"    Std of terminal rate: {fx_terminal.std():.2f}")

for cl in CONFIDENCE_LEVELS:
    var_abs = np.percentile(fx_pnl, (1 - cl) * 100)
    var_pct = np.percentile(fx_pnl_pct, (1 - cl) * 100)
    # Expected Shortfall (CVaR)
    es_abs = fx_pnl[fx_pnl <= var_abs].mean()
    es_pct = fx_pnl_pct[fx_pnl_pct <= var_pct].mean()
    print(f"\n    {cl*100:.0f}% Confidence Level:")
    print(f"      VaR (absolute):  {var_abs:+.2f} AMD  "
          f"(rate could drop to {S0_fx + var_abs:.2f})")
    print(f"      VaR (% change):  {var_pct:+.2f}%")
    print(f"      CVaR/ES (abs):   {es_abs:+.2f} AMD")
    print(f"      CVaR/ES (%):     {es_pct:+.2f}%")

# --- Refinancing Rate VaR ---
rate_terminal = rate_paths[:, -1]
rate_pnl = rate_terminal - S0_rate       # change in rate (pp)
rate_pnl_bps = rate_pnl * 100           # change in bps

print(f"\n  ▶ Refinancing Rate VaR ({HORIZON_DAYS}-day horizon)")
print(f"    Initial rate: {S0_rate:.2f}%")
print(f"    Mean terminal rate: {rate_terminal.mean():.2f}%")
print(f"    Std of terminal rate: {rate_terminal.std():.2f}%")

for cl in CONFIDENCE_LEVELS:
    var_rate = np.percentile(rate_pnl, (1 - cl) * 100)
    var_bps = np.percentile(rate_pnl_bps, (1 - cl) * 100)
    es_rate = rate_pnl[rate_pnl <= var_rate].mean()
    print(f"\n    {cl*100:.0f}% Confidence Level:")
    print(f"      VaR (pp change): {var_rate:+.4f} pp  "
          f"(rate could move to {S0_rate + var_rate:.2f}%)")
    print(f"      VaR (bps):       {var_bps:+.2f} bps")
    print(f"      CVaR/ES (pp):    {es_rate:+.4f} pp")

# --- Combined Portfolio VaR ---
# Consider a portfolio: 1 USD position valued in AMD + interest rate exposure
# Portfolio value = FX_rate * (1 + refi_rate/100)
portfolio_initial = S0_fx * (1 + S0_rate / 100)
portfolio_terminal = fx_paths[:, -1] * (1 + rate_paths[:, -1] / 100)
portfolio_pnl = portfolio_terminal - portfolio_initial
portfolio_pnl_pct = (portfolio_terminal / portfolio_initial - 1) * 100

print(f"\n  ▶ Combined Portfolio VaR (FX × Rate, {HORIZON_DAYS}-day horizon)")
print(f"    Initial portfolio value: {portfolio_initial:.2f} AMD")
print(f"    Mean terminal value: {portfolio_terminal.mean():.2f} AMD")

for cl in CONFIDENCE_LEVELS:
    var_p = np.percentile(portfolio_pnl, (1 - cl) * 100)
    var_p_pct = np.percentile(portfolio_pnl_pct, (1 - cl) * 100)
    es_p = portfolio_pnl[portfolio_pnl <= var_p].mean()
    print(f"\n    {cl*100:.0f}% Confidence Level:")
    print(f"      VaR (absolute):  {var_p:+.2f} AMD")
    print(f"      VaR (% change):  {var_p_pct:+.2f}%")
    print(f"      CVaR/ES (abs):   {es_p:+.2f} AMD")

# ══════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 24))

# --- Plot 1: Historical Data ---
ax1 = fig.add_subplot(4, 2, 1)
ax1.plot(df['Date'], df['USD_AMD'], color='#2196F3', linewidth=0.8)
ax1.set_title('Historical USD/AMD Exchange Rate')
ax1.set_ylabel('AMD per 1 USD')
ax1.xaxis.set_major_locator(mdates.YearLocator(5))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax2 = fig.add_subplot(4, 2, 2)
ax2.step(df['Date'], df['Refinance_Rate'], where='post', color='#E91E63', linewidth=1.2)
ax2.set_title('Refinancing Rate (Վdelays)')
ax2.set_ylabel('Rate (%)')
ax2.xaxis.set_major_locator(mdates.YearLocator(5))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- Plot 2: Simulated FX Paths ---
ax3 = fig.add_subplot(4, 2, 3)
time_axis = np.arange(HORIZON_DAYS + 1)
# Plot subset of paths
n_show = min(200, N_SIMULATIONS)
for i in range(n_show):
    ax3.plot(time_axis, fx_paths[i], alpha=0.05, color='#2196F3', linewidth=0.5)
ax3.plot(time_axis, np.mean(fx_paths, axis=0), color='black', linewidth=2, label='Mean')
ax3.plot(time_axis, np.percentile(fx_paths, 5, axis=0), 'r--', linewidth=1.5, label='5th percentile')
ax3.plot(time_axis, np.percentile(fx_paths, 95, axis=0), 'g--', linewidth=1.5, label='95th percentile')
ax3.set_title(f'Monte Carlo – USD/AMD ({N_SIMULATIONS:,} paths)')
ax3.set_xlabel('Days')
ax3.set_ylabel('AMD per 1 USD')
ax3.legend(fontsize=9)

# --- Plot 3: Simulated Rate Paths ---
ax4 = fig.add_subplot(4, 2, 4)
for i in range(n_show):
    ax4.plot(time_axis, rate_paths[i], alpha=0.05, color='#E91E63', linewidth=0.5)
ax4.plot(time_axis, np.mean(rate_paths, axis=0), color='black', linewidth=2, label='Mean')
ax4.plot(time_axis, np.percentile(rate_paths, 5, axis=0), 'r--', linewidth=1.5, label='5th percentile')
ax4.plot(time_axis, np.percentile(rate_paths, 95, axis=0), 'g--', linewidth=1.5, label='95th percentile')
ax4.set_title(f'Monte Carlo – Refinancing Rate ({N_SIMULATIONS:,} paths)')
ax4.set_xlabel('Days')
ax4.set_ylabel('Rate (%)')
ax4.legend(fontsize=9)

# --- Plot 4: FX P&L Distribution with VaR ---
ax5 = fig.add_subplot(4, 2, 5)
ax5.hist(fx_pnl, bins=100, density=True, alpha=0.7, color='#2196F3', edgecolor='white')
for cl, color, ls in zip(CONFIDENCE_LEVELS, ['#FF9800', '#F44336', '#9C27B0'],
                          ['--', '-', '-.']):
    var_val = np.percentile(fx_pnl, (1-cl)*100)
    ax5.axvline(var_val, color=color, linestyle=ls, linewidth=2,
                label=f'VaR {cl*100:.0f}%: {var_val:+.2f}')
ax5.set_title('USD/AMD – P&L Distribution & VaR')
ax5.set_xlabel('P&L (AMD)')
ax5.set_ylabel('Density')
ax5.legend(fontsize=9)

# --- Plot 5: Rate Change Distribution with VaR ---
ax6 = fig.add_subplot(4, 2, 6)
ax6.hist(rate_pnl, bins=100, density=True, alpha=0.7, color='#E91E63', edgecolor='white')
for cl, color, ls in zip(CONFIDENCE_LEVELS, ['#FF9800', '#F44336', '#9C27B0'],
                          ['--', '-', '-.']):
    var_val = np.percentile(rate_pnl, (1-cl)*100)
    ax6.axvline(var_val, color=color, linestyle=ls, linewidth=2,
                label=f'VaR {cl*100:.0f}%: {var_val:+.4f} pp')
ax6.set_title('Refinancing Rate – Change Distribution & VaR')
ax6.set_xlabel('Rate Change (pp)')
ax6.set_ylabel('Density')
ax6.legend(fontsize=9)

# --- Plot 6: Combined Portfolio P&L ---
ax7 = fig.add_subplot(4, 2, 7)
ax7.hist(portfolio_pnl_pct, bins=100, density=True, alpha=0.7, color='#4CAF50', edgecolor='white')
for cl, color, ls in zip(CONFIDENCE_LEVELS, ['#FF9800', '#F44336', '#9C27B0'],
                          ['--', '-', '-.']):
    var_val = np.percentile(portfolio_pnl_pct, (1-cl)*100)
    ax7.axvline(var_val, color=color, linestyle=ls, linewidth=2,
                label=f'VaR {cl*100:.0f}%: {var_val:+.2f}%')
ax7.set_title('Combined Portfolio – P&L Distribution & VaR')
ax7.set_xlabel('P&L (%)')
ax7.set_ylabel('Density')
ax7.legend(fontsize=9)

# --- Plot 7: Terminal value scatter ---
ax8 = fig.add_subplot(4, 2, 8)
scatter = ax8.scatter(fx_terminal, rate_terminal, c=portfolio_pnl_pct,
                      cmap='RdYlGn', alpha=0.3, s=5, edgecolors='none')
ax8.set_xlabel('Terminal USD/AMD')
ax8.set_ylabel('Terminal Refi Rate (%)')
ax8.set_title('Joint Terminal Distribution')
plt.colorbar(scatter, ax=ax8, label='Portfolio P&L (%)')
ax8.axvline(S0_fx, color='blue', linestyle=':', alpha=0.5)
ax8.axhline(S0_rate, color='red', linestyle=':', alpha=0.5)

plt.tight_layout(pad=3.0)
plt.savefig('monte_carlo_var_results.png', dpi=150, bbox_inches='tight')
print("\n  ✓ Charts saved to 'monte_carlo_var_results.png'")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  VaR SUMMARY TABLE")
print("═" * 70)

summary_data = []
for cl in CONFIDENCE_LEVELS:
    q = (1 - cl) * 100
    summary_data.append({
        'Confidence': f'{cl*100:.0f}%',
        'FX VaR (AMD)': f'{np.percentile(fx_pnl, q):+.2f}',
        'FX VaR (%)': f'{np.percentile(fx_pnl_pct, q):+.2f}%',
        'Rate VaR (pp)': f'{np.percentile(rate_pnl, q):+.4f}',
        'Portfolio VaR (%)': f'{np.percentile(portfolio_pnl_pct, q):+.2f}%',
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "═" * 70)
print("  INTERPRETATION")
print("═" * 70)

var_95_fx = np.percentile(fx_pnl, 5)
var_99_fx = np.percentile(fx_pnl, 1)
var_95_rate = np.percentile(rate_pnl, 5)

print(f"""
  • Over a {HORIZON_DAYS}-day horizon, with 95% confidence:
    – The USD/AMD rate will not depreciate by more than {abs(var_95_fx):.2f} AMD
      (from {S0_fx:.2f} to a floor of ~{S0_fx + var_95_fx:.2f} AMD/USD)
    – The refinancing rate will not decrease by more than {abs(var_95_rate):.4f} pp
      (from {S0_rate:.2f}% to ~{max(S0_rate + var_95_rate, 0):.2f}%)

  • At 99% confidence, the FX worst-case loss is {abs(var_99_fx):.2f} AMD

  • CVaR (Expected Shortfall) measures the average loss in the tail,
    providing insight into extreme risk beyond the VaR threshold.

  • The combined portfolio VaR captures the joint risk from both
    exchange rate fluctuations and interest rate movements.
""")

plt.show()

print("Done.")
