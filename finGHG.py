import pandas as pd
import numpy as np
import warnings

# --- 1. FINANCIAL AND TECHNICAL ASSUMPTIONS & FACTORS ---
warnings.filterwarnings("ignore", category=FutureWarning)

# System and Financial Parameters
N = 15                 # System Lifetime in Years (for Annualized Capital Cost)
r = 0.03               # Real Discount Rate (3.0%)
INITIAL_COST = 15000   # Estimated initial cost of the HP system (before subsidy)
ANNUAL_DEMAND_GJ = 75.0 # Annual space heating demand (GJ/year)

# Efficiency Assumptions
HP_S_COP = 3.0         # Heat Pump S-COP (300% efficiency)
OIL_FURNACE_EFF = 0.70 # Old Oil Furnace Efficiency
NG_FURNACE_EFF = 0.80  # Old NG Furnace Efficiency
ELEC_BASEBOARD_EFF = 1.0 # Electric Baseboard/Furnace Efficiency (100%)

# Energy Conversion Constants
LITRES_PER_GJ = 26.15  # Litres of oil per Gigajoule
KWH_PER_GJ = 277.8     # kWh per Gigajoule

# CO2 Emission Factors (t CO2e / GJ or kWh)
EF_OIL_PER_GJ = 0.074  # tonnes CO2e per GJ
EF_NG_PER_GJ = 0.050   # tonnes CO2e per GJ

# Provincial Electricity Grid Emission Factors (t CO2e / kWh)
EF_GRID = {
    'Quebec': 0.0000012,        
    'British Columbia': 0.000013,
    'Manitoba': 0.0000026,
    'Ontario': 0.000028,
    'Nova Scotia': 0.000496,
    'Saskatchewan': 0.000674,
    'Alberta': 0.000500,
    'New Brunswick': 0.000160,
    'Newfoundland and Labrador': 0.000030,
    'Prince Edward Island': 0.000000,
}

# Provincial Natural Gas Prices ($/GJ) - Based on NRCan data and utility rates
NG_PRICES = {
    'Alberta': 7.00,
    'Saskatchewan': 8.00,
    'Manitoba': 11.00,
    'Ontario': 13.00,
    'British Columbia': 10.00,
    'Quebec': 14.00,
    'New Brunswick': 14.00,
    'Nova Scotia': 14.00,
    'Newfoundland and Labrador': 14.00,
    'Prince Edward Island': 14.00
}

# Updated Fuel Switch Weights from 2023 HES Data (Current Stock Composition)
df_weights_data = {
    'Province': ['Alberta', 'British Columbia', 'Manitoba', 'New Brunswick', 
                 'Newfoundland and Labrador', 'Nova Scotia', 'Ontario', 
                 'Prince Edward Island', 'Quebec', 'Saskatchewan'],
    'W_Oil': [0.00, 0.01, 0.00, 0.07, 0.17, 0.49, 0.01, 0.61, 0.03, 0.00],
    'W_NG': [0.84, 0.53, 0.61, 0.00, 0.00, 0.00, 0.75, 0.00, 0.06, 0.82],
    'W_Elec': [0.16, 0.46, 0.39, 0.93, 0.83, 0.51, 0.23, 0.39, 0.91, 0.18]
}
df_weights = pd.DataFrame(df_weights_data)

# --- 2. DATA PREPARATION ---

# Load data
df_panel = pd.read_csv("heat_pump_panel_with_fuel_prices.csv")

# Standardize province names
province_mapping = {
    'British Columbia and the Territories': 'British Columbia'
}
df_panel['province'] = df_panel['province'].replace(province_mapping)

# Calculate average fuel prices (2020-2024 average)
df_avg_prices = df_panel.groupby('province')[[
    'electricity_cents_per_kwh',
    'heating_oil_cents_per_litre'
]].mean().reset_index().rename(columns={'province': 'Province'})

df_avg_prices['elec_price_per_kwh_dollar'] = df_avg_prices['electricity_cents_per_kwh'] / 100
df_avg_prices['oil_price_per_litre_dollar'] = df_avg_prices['heating_oil_cents_per_litre'] / 100

# Add provincial NG prices
df_avg_prices['ng_price_per_gj_dollar'] = df_avg_prices['Province'].map(NG_PRICES)

# Get Max Subsidy from the panel data
df_max_subsidy = df_panel.groupby('province')['total_baseline_subsidy'].max().reset_index().rename(
    columns={'province': 'Province', 'total_baseline_subsidy': 'Max_Subsidy'}
)
df_final = pd.merge(df_max_subsidy, df_avg_prices, on='Province', how='left')
df_final = pd.merge(df_final, df_weights, on='Province', how='left')
df_final['EF_Grid'] = df_final['Province'].map(EF_GRID)

# Drop provinces outside the scope (keep only those with weight data)
df_final = df_final.dropna(subset=['W_Oil']) 


# --- 3. WEIGHTED FINANCIAL CALCULATION ---

# Function to calculate cost per GJ of USEFUL HEAT
def cost_per_gj_heat(price_per_unit, unit_per_gj, efficiency):
    return (price_per_unit * unit_per_gj) / efficiency

# Cost of HP Heat (same for all switches)
df_final['Cost_HP_Heat'] = cost_per_gj_heat(df_final['elec_price_per_kwh_dollar'], KWH_PER_GJ, HP_S_COP)

# Cost of Old Fuel Heat (per GJ)
df_final['Cost_Oil_Heat'] = cost_per_gj_heat(df_final['oil_price_per_litre_dollar'], LITRES_PER_GJ, OIL_FURNACE_EFF)
df_final['Cost_NG_Heat'] = cost_per_gj_heat(df_final['ng_price_per_gj_dollar'], 1.0, NG_FURNACE_EFF)
df_final['Cost_Elec_Heat'] = cost_per_gj_heat(df_final['elec_price_per_kwh_dollar'], KWH_PER_GJ, ELEC_BASEBOARD_EFF)

# WEIGHTED Annual Operational Savings (Savings = (Old Cost - HP Cost) * Demand * Weight)
df_final['Savings_Oil_HP'] = (df_final['Cost_Oil_Heat'] - df_final['Cost_HP_Heat']) * ANNUAL_DEMAND_GJ * df_final['W_Oil']
df_final['Savings_NG_HP'] = (df_final['Cost_NG_Heat'] - df_final['Cost_HP_Heat']) * ANNUAL_DEMAND_GJ * df_final['W_NG']
df_final['Savings_Elec_HP'] = (df_final['Cost_Elec_Heat'] - df_final['Cost_HP_Heat']) * ANNUAL_DEMAND_GJ * df_final['W_Elec']

df_final['Weighted_Annual_Savings'] = df_final[['Savings_Oil_HP', 'Savings_NG_HP', 'Savings_Elec_HP']].sum(axis=1)


# --- 4. WEIGHTED EMISSIONS CALCULATION ---

# Calculate HP Emissions (Tonnes CO2e / year)
HP_ELEC_CONSUMPTION_KWH = ANNUAL_DEMAND_GJ * KWH_PER_GJ / HP_S_COP
df_final['Emiss_HP'] = HP_ELEC_CONSUMPTION_KWH * df_final['EF_Grid']

# Baseline Emissions (Tonnes CO2e / year)
df_final['Emiss_Oil'] = (ANNUAL_DEMAND_GJ / OIL_FURNACE_EFF) * EF_OIL_PER_GJ
df_final['Emiss_NG'] = (ANNUAL_DEMAND_GJ / NG_FURNACE_EFF) * EF_NG_PER_GJ
df_final['Emiss_Elec'] = (ANNUAL_DEMAND_GJ / ELEC_BASEBOARD_EFF) * KWH_PER_GJ * df_final['EF_Grid']

# Weighted Annual Emission Reduction (Reduction = (Baseline - HP) * Weight)
df_final['Reduction_Oil_HP'] = (df_final['Emiss_Oil'] - df_final['Emiss_HP']) * df_final['W_Oil']
df_final['Reduction_NG_HP'] = (df_final['Emiss_NG'] - df_final['Emiss_HP']) * df_final['W_NG']
df_final['Reduction_Elec_HP'] = (df_final['Emiss_Elec'] - df_final['Emiss_HP']) * df_final['W_Elec']

df_final['Weighted_CO2_Reduction_Tonnes'] = df_final[['Reduction_Oil_HP', 'Reduction_NG_HP', 'Reduction_Elec_HP']].sum(axis=1)


# --- 5. LIFETIME TOTALS (15-year system life) ---
df_final['Lifetime_Savings'] = df_final['Weighted_Annual_Savings'] * N
df_final['Lifetime_GHG_Reduction_Tonnes'] = df_final['Weighted_CO2_Reduction_Tonnes'] * N


# --- 6. FINAL OUTPUT TABLE ---
df_final['Net_Cap_Cost'] = np.maximum(0, INITIAL_COST - df_final['Max_Subsidy'])
df_final['Payback_Years'] = df_final['Net_Cap_Cost'] / df_final['Weighted_Annual_Savings']
df_final['Annual_CO2_Reduction_kg'] = df_final['Weighted_CO2_Reduction_Tonnes'] * 1000
df_final['Lifetime_GHG_Reduction_kg'] = df_final['Lifetime_GHG_Reduction_Tonnes'] * 1000

df_output = df_final[[
    'Province', 
    'Max_Subsidy', 
    'Weighted_Annual_Savings',
    'Lifetime_Savings',
    'Net_Cap_Cost',
    'Payback_Years',
    'Annual_CO2_Reduction_kg',
    'Lifetime_GHG_Reduction_kg'
]].rename(columns={
    'Weighted_Annual_Savings': 'Annual_Savings',
    'Net_Cap_Cost': 'Net_Capital_Cost',
    'Annual_CO2_Reduction_kg': 'Annual_GHG_kg',
    'Lifetime_GHG_Reduction_kg': 'Lifetime_GHG_kg'
}).sort_values(by='Annual_Savings', ascending=False)

# Format output
df_output['Annual_Savings'] = df_output['Annual_Savings'].round(0).astype(int)
df_output['Lifetime_Savings'] = df_output['Lifetime_Savings'].round(0).astype(int)
df_output['Net_Capital_Cost'] = df_output['Net_Capital_Cost'].round(0).astype(int)
df_output['Payback_Years'] = df_output['Payback_Years'].replace([np.inf, -np.inf], np.nan).round(1)
df_output['Annual_GHG_kg'] = df_output['Annual_GHG_kg'].round(0).astype(int)
df_output['Lifetime_GHG_kg'] = df_output['Lifetime_GHG_kg'].round(0).astype(int)

# Save to CSV
df_output.to_csv("heat_pump_household_benefits_2023.csv", index=False)

print("="*100)
print("HOUSEHOLD BENEFITS FROM HEAT PUMP SUBSIDIES (2023 Fuel Mix)")
print("="*100)
print(df_output.to_markdown(index=False))

print("\n\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)
print(f"\nNational Average (weighted by provinces):")
print(f"  Annual Household Savings:        ${df_output['Annual_Savings'].mean():,.0f}")
print(f"  Lifetime Household Savings:      ${df_output['Lifetime_Savings'].mean():,.0f}")
print(f"  Annual GHG Reduction per HH:     {df_output['Annual_GHG_kg'].mean():,.0f} kg CO2e")
print(f"  Lifetime GHG Reduction per HH:   {df_output['Lifetime_GHG_kg'].mean():,.0f} kg CO2e ({df_output['Lifetime_GHG_kg'].mean()/1000:.1f} tonnes)")
print(f"  Average Payback Period:          {df_output['Payback_Years'].mean():.1f} years")

print(f"\nRange Across Provinces:")
print(f"  Annual Savings:    ${df_output['Annual_Savings'].min():,.0f} to ${df_output['Annual_Savings'].max():,.0f}")
print(f"  Lifetime Savings:  ${df_output['Lifetime_Savings'].min():,.0f} to ${df_output['Lifetime_Savings'].max():,.0f}")
print(f"  Annual GHG:        {df_output['Annual_GHG_kg'].min():,.0f} to {df_output['Annual_GHG_kg'].max():,.0f} kg")
print(f"  Lifetime GHG:      {df_output['Lifetime_GHG_kg'].min():,.0f} to {df_output['Lifetime_GHG_kg'].max():,.0f} kg")

print("\n" + "="*100)
print("Saved to: heat_pump_household_benefits_2023.csv")
print("="*100)