# ============================================================================
# ADVANCED GEX + DEX ANALYSIS - MODIFIED VERSION
# WITH UPDATED GEX FLOW LOGIC & ATM STRADDLE CHART
# ============================================================================

# STEP 1: INSTALL REQUIRED LIBRARIES
# ----------------------------------------------------------------------------
# !pip install requests pandas numpy plotly scipy tabulate -q

import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from tabulate import tabulate
import warnings
import time
from IPython.display import clear_output, HTML
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully!")
print("="*80)

# ============================================================================
# BLACK-SCHOLES CALCULATOR (GAMMA + DELTA)
# ============================================================================

class BlackScholesCalculator:
    """Calculate accurate gamma and delta using Black-Scholes formula"""

    @staticmethod
    def calculate_d1(S, K, T, r, sigma):
        """Calculate d1 for Black-Scholes"""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))

    @staticmethod
    def calculate_gamma(S, K, T, r, sigma):
        """Calculate Black-Scholes Gamma"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0

        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            n_prime_d1 = norm.pdf(d1)
            gamma = n_prime_d1 / (S * sigma * np.sqrt(T))
            return gamma
        except Exception as e:
            return 0

    @staticmethod
    def calculate_call_delta(S, K, T, r, sigma):
        """Calculate Call Delta = N(d1)"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0

        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            return norm.cdf(d1)
        except:
            return 0

    @staticmethod
    def calculate_put_delta(S, K, T, r, sigma):
        """Calculate Put Delta = N(d1) - 1"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0

        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            return norm.cdf(d1) - 1
        except:
            return 0

# ============================================================================
# ENHANCED NSE DATA FETCHER WITH GEX + DEX CALCULATIONS
# ============================================================================

class EnhancedGEXDEXCalculator:
    """Advanced GEX + DEX calculations with improved futures fetching"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.nseindia.com/',
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.base_url = "https://www.nseindia.com"
        self.option_chain_url = "https://www.nseindia.com/api/option-chain-indices"
        self.risk_free_rate = 0.07
        self.bs_calc = BlackScholesCalculator()

        # Initialize session
        try:
            self.session.get(self.base_url, timeout=10)
            print("✅ Connected to NSE")
        except:
            print("⚠️ NSE connection initialized")

    def fetch_futures_ltp_method1(self, symbol, expiry_date=None):
        """Method 1: Fetch from Groww.in (Most reliable!)"""
        try:
            symbol_map = {
                'NIFTY': 'nifty',
                'BANKNIFTY': 'bank-nifty',
                'FINNIFTY': 'finnifty',
                'MIDCPNIFTY': 'midcpnifty'
            }

            groww_symbol = symbol_map.get(symbol, 'nifty')
            url = f"https://groww.in/futures/{groww_symbol}"

            print(f"   Fetching from Groww: {url}")

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }

            response = self.session.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                html_content = response.text
                import re

                script_patterns = [
                    r'"ltp":\s*([0-9.]+)',
                    r'"lastPrice":\s*([0-9.]+)',
                    r'"close":\s*([0-9.]+)',
                    r'"currentPrice":\s*([0-9.]+)',
                    r'ltp.*?([0-9]{5,6}\.[0-9]{1,2})',
                ]

                for pattern in script_patterns:
                    matches = re.findall(pattern, html_content)
                    if matches:
                        for match in matches:
                            price = float(match)
                            if symbol == 'NIFTY' and 15000 < price < 35000:
                                print(f"   ✅ Found futures LTP: {price}")
                                return price, expiry_date
                            elif symbol == 'BANKNIFTY' and 35000 < price < 60000:
                                print(f"   ✅ Found futures LTP: {price}")
                                return price, expiry_date
                            elif symbol == 'FINNIFTY' and 15000 < price < 30000:
                                print(f"   ✅ Found futures LTP: {price}")
                                return price, expiry_date

                api_url = f"https://groww.in/v1/api/stocks_fo_data/v1/tr_live_prices/exchange/NSE/segment/FO/latest/{symbol}FUT"
                try:
                    api_response = self.session.get(api_url, headers=headers, timeout=10)
                    if api_response.status_code == 200:
                        api_data = api_response.json()
                        if 'ltp' in api_data:
                            price = float(api_data['ltp'])
                            print(f"   ✅ Found futures LTP via API: {price}")
                            return price, expiry_date
                except:
                    pass

            return None, None
        except Exception as e:
            print(f"   Groww method failed: {e}")
            return None, None

    def fetch_futures_ltp_method2(self, symbol, spot_price, expiry_date):
        """Method 2: Calculate from ATM options using Put-Call Parity"""
        try:
            print(f"   Method 2: Calculating from ATM options...")
            url = f"{self.option_chain_url}?symbol={symbol}"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                records = data['records']

                atm_strike = None
                min_diff = float('inf')

                for item in records.get('data', []):
                    if expiry_date and item.get('expiryDate') != expiry_date:
                        continue

                    strike = item.get('strikePrice', 0)
                    diff = abs(strike - spot_price)

                    if diff < min_diff:
                        min_diff = diff
                        atm_strike = strike

                if atm_strike:
                    for item in records.get('data', []):
                        if item.get('strikePrice') == atm_strike:
                            if expiry_date and item.get('expiryDate') != expiry_date:
                                continue

                            ce = item.get('CE', {})
                            pe = item.get('PE', {})

                            call_ltp = ce.get('lastPrice', 0)
                            put_ltp = pe.get('lastPrice', 0)

                            if call_ltp > 0 and put_ltp > 0:
                                futures_price = atm_strike + call_ltp - put_ltp
                                print(f"   ✅ ATM Strike: {atm_strike}, Call: {call_ltp}, Put: {put_ltp}")
                                print(f"   ✅ Calculated Futures: {futures_price:.2f}")
                                return futures_price, expiry_date

            return None, None
        except Exception as e:
            print(f"   Method 2 failed: {e}")
            return None, None

    def fetch_futures_ltp_method3(self, symbol, spot_price, days_to_expiry):
        """Method 3: Theoretical futures price using cost of carry"""
        try:
            print(f"   Method 3: Cost of carry calculation...")
            T = days_to_expiry / 365.0
            futures_price = spot_price * np.exp(self.risk_free_rate * T)
            print(f"   ✅ Theoretical Futures (Cost of Carry): {futures_price:.2f}")
            return futures_price, None
        except Exception as e:
            print(f"   Method 3 failed: {e}")
            return None, None

    def fetch_futures_ltp_comprehensive(self, symbol, spot_price, expiry_date, days_to_expiry):
        """Comprehensive futures fetching with multiple fallback methods"""
        print("\n" + "="*80)
        print("🔍 FETCHING INDEX FUTURES LTP - TRYING MULTIPLE METHODS...")
        print("="*80)

        print("\n🔄 Method 1: Fetching from Groww.in...")
        futures_ltp, futures_expiry = self.fetch_futures_ltp_method1(symbol, expiry_date)

        if futures_ltp and futures_ltp > 0:
            print(f"✅ SUCCESS - Method 1 (Groww.in) worked!")
            return futures_ltp, futures_expiry, "Groww.in"

        print("\n🔄 Method 2: Calculating from ATM options (Put-Call Parity)...")
        futures_ltp, futures_expiry = self.fetch_futures_ltp_method2(symbol, spot_price, expiry_date)

        if futures_ltp and futures_ltp > 0:
            print(f"✅ SUCCESS - Method 2 (Put-Call Parity) worked!")
            return futures_ltp, futures_expiry, "Put-Call Parity"

        print("\n🔄 Method 3: Theoretical calculation (Cost of Carry)...")
        futures_ltp, _ = self.fetch_futures_ltp_method3(symbol, spot_price, days_to_expiry)

        if futures_ltp and futures_ltp > 0:
            print(f"✅ SUCCESS - Method 3 (Cost of Carry) worked!")
            return futures_ltp, expiry_date, "Cost of Carry Model"

        print("\n⚠️ All methods failed - Using SPOT price as fallback")
        return spot_price, expiry_date, "Spot (Fallback)"

    def calculate_time_to_expiry(self, expiry_date_str):
        """Calculate time to expiry in years"""
        try:
            expiry_date = datetime.strptime(expiry_date_str, "%d-%b-%Y")
            today = datetime.now()
            days_to_expiry = (expiry_date - today).days
            time_to_expiry = max(days_to_expiry / 365, 0.001)
            return time_to_expiry, days_to_expiry
        except:
            return 7/365, 7

    def fetch_and_calculate_gex_dex(self, symbol="NIFTY", strikes_range=10, expiry_index=0):
        """
        Fetch option chain and calculate both GEX and DEX
        """
        try:
            print(f"\n🔄 Fetching live {symbol} data...")

            url = f"{self.option_chain_url}?symbol={symbol}"
            response = self.session.get(url, timeout=10)

            if response.status_code != 200:
                print(f"❌ Failed to fetch data. Status: {response.status_code}")
                return None, None, None, None

            data = response.json()
            records = data['records']

            spot_price = records.get('underlyingValue', 0)
            timestamp = records.get('timestamp', '')
            expiry_dates = records.get('expiryDates', [])

            print(f"📍 Underlying Spot Price: {spot_price:,.2f}")
            print(f"🕒 Last Updated: {timestamp}")

            if expiry_dates:
                print(f"\n📅 Available Expiries ({len(expiry_dates)} total):")
                for idx, exp_date in enumerate(expiry_dates[:5]):
                    marker = "👉" if idx == expiry_index else "  "
                    print(f"   {marker} [{idx}] {exp_date}")

            if not expiry_dates:
                selected_expiry = None
                time_to_expiry = 7/365
                days_to_expiry = 7
            elif expiry_index >= len(expiry_dates):
                selected_expiry = expiry_dates[0]
                time_to_expiry, days_to_expiry = self.calculate_time_to_expiry(selected_expiry)
            else:
                selected_expiry = expiry_dates[expiry_index]
                time_to_expiry, days_to_expiry = self.calculate_time_to_expiry(selected_expiry)

            print(f"\n✅ Selected Expiry: {selected_expiry}")
            print(f"⏰ Days to Expiry: {days_to_expiry}")

            futures_ltp, futures_expiry, fetch_method = self.fetch_futures_ltp_comprehensive(
                symbol, spot_price, selected_expiry, days_to_expiry
            )

            basis = futures_ltp - spot_price
            basis_pct = (basis / spot_price * 100) if spot_price > 0 else 0

            print("\n" + "="*80)
            print("💰 PRICE COMPARISON & FUTURES DETAILS:")
            print("="*80)
            print(f"📊 Underlying Spot:        {spot_price:>15,.2f}")
            print(f"🔥 Index Futures LTP:      {futures_ltp:>15,.2f}  ⬅️ USING THIS!")
            print(f"📈 Basis (F - S):          {basis:>15,.2f}  ({basis_pct:+.3f}%)")
            print(f"🔧 Fetch Method:           {fetch_method:>15}")
            print(f"📅 Futures Expiry:         {futures_expiry if futures_expiry else 'N/A':>15}")
            print("="*80)

            reference_price = futures_ltp

            # Contract specifications
            if 'BANKNIFTY' in symbol:
                contract_size = 15
                strike_interval = 100
            elif 'FINNIFTY' in symbol:
                contract_size = 40
                strike_interval = 50
            elif 'MIDCPNIFTY' in symbol:
                contract_size = 75
                strike_interval = 25
            else:
                contract_size = 25
                strike_interval = 50

            # Process strikes data
            all_strikes = []
            processed_strikes = set()
            atm_strike = None
            min_atm_diff = float('inf')
            atm_call_premium = 0
            atm_put_premium = 0

            for item in records.get('data', []):
                if selected_expiry and item.get('expiryDate') != selected_expiry:
                    continue

                strike = item.get('strikePrice', 0)
                if strike == 0 or strike in processed_strikes:
                    continue

                processed_strikes.add(strike)

                strike_distance = abs(strike - reference_price) / strike_interval
                if strike_distance > strikes_range:
                    continue

                ce = item.get('CE', {})
                pe = item.get('PE', {})

                call_oi = ce.get('openInterest', 0)
                put_oi = pe.get('openInterest', 0)
                call_oi_change = ce.get('changeinOpenInterest', 0)
                put_oi_change = pe.get('changeinOpenInterest', 0)
                call_volume = ce.get('totalTradedVolume', 0)
                put_volume = pe.get('totalTradedVolume', 0)
                call_iv = ce.get('impliedVolatility', 0)
                put_iv = pe.get('impliedVolatility', 0)
                call_ltp = ce.get('lastPrice', 0)
                put_ltp = pe.get('lastPrice', 0)

                # Find ATM strike
                strike_diff = abs(strike - reference_price)
                if strike_diff < min_atm_diff:
                    min_atm_diff = strike_diff
                    atm_strike = strike
                    atm_call_premium = call_ltp
                    atm_put_premium = put_ltp

                call_iv_decimal = call_iv / 100 if call_iv > 0 else 0.15
                put_iv_decimal = put_iv / 100 if put_iv > 0 else 0.15

                # Calculate Gammas
                call_gamma = self.bs_calc.calculate_gamma(
                    S=reference_price, K=strike, T=time_to_expiry,
                    r=self.risk_free_rate, sigma=call_iv_decimal
                )

                put_gamma = self.bs_calc.calculate_gamma(
                    S=reference_price, K=strike, T=time_to_expiry,
                    r=self.risk_free_rate, sigma=put_iv_decimal
                )

                # Calculate Deltas
                call_delta = self.bs_calc.calculate_call_delta(
                    S=reference_price, K=strike, T=time_to_expiry,
                    r=self.risk_free_rate, sigma=call_iv_decimal
                )

                put_delta = self.bs_calc.calculate_put_delta(
                    S=reference_price, K=strike, T=time_to_expiry,
                    r=self.risk_free_rate, sigma=put_iv_decimal
                )

                # Calculate GEX (in Billions)
                call_gex = (call_oi * call_gamma * reference_price * reference_price * contract_size) / 1_000_000_000
                put_gex = -(put_oi * put_gamma * reference_price * reference_price * contract_size) / 1_000_000_000

                # Calculate DEX (in Billions)
                call_dex = (call_oi * call_delta * reference_price * contract_size) / 1_000_000_000
                put_dex = (put_oi * put_delta * reference_price * contract_size) / 1_000_000_000

                # Flow GEX
                call_flow_gex = (call_oi_change * call_gamma * reference_price * reference_price * contract_size) / 1_000_000_000
                put_flow_gex = -(put_oi_change * put_gamma * reference_price * reference_price * contract_size) / 1_000_000_000

                # Flow DEX
                call_flow_dex = (call_oi_change * call_delta * reference_price * contract_size) / 1_000_000_000
                put_flow_dex = (put_oi_change * put_delta * reference_price * contract_size) / 1_000_000_000

                all_strikes.append({
                    'Strike': strike,
                    'Call_OI': call_oi,
                    'Put_OI': put_oi,
                    'Call_OI_Change': call_oi_change,
                    'Put_OI_Change': put_oi_change,
                    'Call_Volume': call_volume,
                    'Put_Volume': put_volume,
                    'Call_IV': call_iv,
                    'Put_IV': put_iv,
                    'Call_LTP': call_ltp,
                    'Put_LTP': put_ltp,
                    'Call_Gamma': call_gamma,
                    'Put_Gamma': put_gamma,
                    'Call_Delta': call_delta,
                    'Put_Delta': put_delta,
                    'Call_GEX': call_gex,
                    'Put_GEX': put_gex,
                    'Net_GEX': call_gex + put_gex,
                    'Call_DEX': call_dex,
                    'Put_DEX': put_dex,
                    'Net_DEX': call_dex + put_dex,
                    'Call_Flow_GEX': call_flow_gex,
                    'Put_Flow_GEX': put_flow_gex,
                    'Net_Flow_GEX': call_flow_gex + put_flow_gex,
                    'Call_Flow_DEX': call_flow_dex,
                    'Put_Flow_DEX': put_flow_dex,
                    'Net_Flow_DEX': call_flow_dex + put_flow_dex
                })

            if not all_strikes:
                print("❌ No strikes data found")
                return None, None, None, None

            df = pd.DataFrame(all_strikes)
            df = df.sort_values('Strike').reset_index(drop=True)

            df['Call_GEX_B'] = df['Call_GEX']
            df['Put_GEX_B'] = df['Put_GEX']
            df['Net_GEX_B'] = df['Net_GEX']
            df['Call_DEX_B'] = df['Call_DEX']
            df['Put_DEX_B'] = df['Put_DEX']
            df['Net_DEX_B'] = df['Net_DEX']
            df['Call_Flow_GEX_B'] = df['Call_Flow_GEX']
            df['Put_Flow_GEX_B'] = df['Put_Flow_GEX']
            df['Net_Flow_GEX_B'] = df['Net_Flow_GEX']
            df['Call_Flow_DEX_B'] = df['Call_Flow_DEX']
            df['Put_Flow_DEX_B'] = df['Put_Flow_DEX']
            df['Net_Flow_DEX_B'] = df['Net_Flow_DEX']
            df['Total_Volume'] = df['Call_Volume'] + df['Put_Volume']

            # Fixed: Ensure scalar division
            max_net_gex = df['Net_GEX_B'].abs().max()
            if max_net_gex > 0:
                df['Hedging_Pressure'] = (df['Net_GEX_B'] / max_net_gex) * 100
            else:
                df['Hedging_Pressure'] = 0

            # Calculate ATM Straddle
            atm_straddle_premium = atm_call_premium + atm_put_premium

            print(f"✅ Processed {len(df)} strikes")
            print(f"📊 Total Net GEX: {df['Net_GEX_B'].sum():.4f} B")
            print(f"📊 Total Net DEX: {df['Net_DEX_B'].sum():.4f} B")
            print(f"🎯 ATM Strike: {atm_strike}")
            print(f"💰 ATM Straddle Premium: ₹{atm_straddle_premium:.2f}")

            # Return ATM info as dictionary
            atm_info = {
                'atm_strike': atm_strike,
                'atm_call_premium': atm_call_premium,
                'atm_put_premium': atm_put_premium,
                'atm_straddle_premium': atm_straddle_premium
            }

            return df, reference_price, fetch_method, atm_info

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None

# ============================================================================
# MODIFIED GEX + DEX FLOW CALCULATION
# ============================================================================

def calculate_dual_gex_dex_flow(df, futures_ltp):
    """
    MODIFIED: Calculate GEX flow based on 5 positive + 5 negative strikes closest to spot
    """
    df_unique = df.drop_duplicates(subset=['Strike']).sort_values('Strike').reset_index(drop=True)

    # ===== NEW GEX FLOW LOGIC =====
    # Get strikes with positive Net GEX, sorted by distance from spot
    positive_gex_df = df_unique[df_unique['Net_GEX_B'] > 0].copy()
    positive_gex_df['Distance'] = abs(positive_gex_df['Strike'] - futures_ltp)
    positive_gex_df = positive_gex_df.sort_values('Distance').head(5)

    # Get strikes with negative Net GEX, sorted by distance from spot
    negative_gex_df = df_unique[df_unique['Net_GEX_B'] < 0].copy()
    negative_gex_df['Distance'] = abs(negative_gex_df['Strike'] - futures_ltp)
    negative_gex_df = negative_gex_df.sort_values('Distance').head(5)

    # Calculate flows
    gex_near_positive = float(positive_gex_df['Net_GEX_B'].sum()) if len(positive_gex_df) > 0 else 0.0
    gex_near_negative = float(negative_gex_df['Net_GEX_B'].sum()) if len(negative_gex_df) > 0 else 0.0
    gex_near_total = gex_near_positive + gex_near_negative

    # Total GEX (all strikes)
    positive_gex_mask = df_unique['Net_GEX_B'] > 0
    negative_gex_mask = df_unique['Net_GEX_B'] < 0

    gex_total_positive = float(df_unique.loc[positive_gex_mask, 'Net_GEX_B'].sum()) if positive_gex_mask.any() else 0.0
    gex_total_negative = float(df_unique.loc[negative_gex_mask, 'Net_GEX_B'].sum()) if negative_gex_mask.any() else 0.0
    gex_total_all = gex_total_positive + gex_total_negative

    # ===== DEX FLOW (keep same as before) =====
    above_futures = df_unique[df_unique['Strike'] > futures_ltp].head(5)
    below_futures = df_unique[df_unique['Strike'] < futures_ltp].tail(5)

    dex_near_positive = float(above_futures['Net_DEX_B'].sum()) if len(above_futures) > 0 else 0.0
    dex_near_negative = float(below_futures['Net_DEX_B'].sum()) if len(below_futures) > 0 else 0.0
    dex_near_total = dex_near_positive + dex_near_negative

    positive_dex_mask = df_unique['Net_DEX_B'] > 0
    negative_dex_mask = df_unique['Net_DEX_B'] < 0

    dex_total_positive = float(df_unique.loc[positive_dex_mask, 'Net_DEX_B'].sum()) if positive_dex_mask.any() else 0.0
    dex_total_negative = float(df_unique.loc[negative_dex_mask, 'Net_DEX_B'].sum()) if negative_dex_mask.any() else 0.0
    dex_total_all = dex_total_positive + dex_total_negative

    # ===== MODIFIED BIAS LOGIC =====
    def get_gex_bias(flow_value):
        """MODIFIED: Positive GEX = Sideways/Bullish, Negative GEX = Bearish/Volatile"""
        if flow_value > 50:
            return "🟢 STRONG BULLISH (Sideways to Bullish)", "green"
        elif flow_value > 0:
            return "🟢 BULLISH (Sideways to Bullish)", "lightgreen"
        elif flow_value < -50:
            return "🔴 STRONG BEARISH (High Volatility)", "red"
        elif flow_value < 0:
            return "🔴 BEARISH (High Volatility)", "lightcoral"
        else:
            return "⚖️ NEUTRAL", "orange"

    def get_dex_bias(flow_value):
        """DEX bias (unchanged)"""
        if flow_value > 50:
            return "🟢 BULLISH", "green"
        elif flow_value < -50:
            return "🔴 BEARISH", "red"
        elif flow_value > 0:
            return "🟢 Mild Bullish", "lightgreen"
        elif flow_value < 0:
            return "🔴 Mild Bearish", "lightcoral"
        else:
            return "⚖️ NEUTRAL", "orange"

    gex_near_bias, gex_near_color = get_gex_bias(gex_near_total)
    gex_total_bias, gex_total_color = get_gex_bias(gex_total_all)
    dex_near_bias, dex_near_color = get_dex_bias(dex_near_total)
    dex_total_bias, dex_total_color = get_dex_bias(dex_total_all)

    # Combined directional signal
    combined_signal = (gex_near_total + dex_near_total) / 2
    combined_bias, combined_color = get_gex_bias(combined_signal)

    return {
        # GEX metrics
        'gex_near_positive': gex_near_positive,
        'gex_near_negative': gex_near_negative,
        'gex_near_total': gex_near_total,
        'gex_near_bias': gex_near_bias,
        'gex_near_color': gex_near_color,
        'gex_total_positive': gex_total_positive,
        'gex_total_negative': gex_total_negative,
        'gex_total_all': gex_total_all,
        'gex_total_bias': gex_total_bias,
        'gex_total_color': gex_total_color,

        # DEX metrics
        'dex_near_positive': dex_near_positive,
        'dex_near_negative': dex_near_negative,
        'dex_near_total': dex_near_total,
        'dex_near_bias': dex_near_bias,
        'dex_near_color': dex_near_color,
        'dex_total_positive': dex_total_positive,
        'dex_total_negative': dex_total_negative,
        'dex_total_all': dex_total_all,
        'dex_total_bias': dex_total_bias,
        'dex_total_color': dex_total_color,

        # Combined
        'combined_signal': combined_signal,
        'combined_bias': combined_bias,
        'combined_color': combined_color,

        # Store strike lists for reference
        'positive_gex_strikes': positive_gex_df['Strike'].tolist() if len(positive_gex_df) > 0 else [],
        'negative_gex_strikes': negative_gex_df['Strike'].tolist() if len(negative_gex_df) > 0 else [],
        'above_strikes': above_futures['Strike'].tolist(),
        'below_strikes': below_futures['Strike'].tolist(),
    }

# ============================================================================
# ENHANCED VISUALIZATION WITH 7 CHARTS (ADDED ATM STRADDLE)
# ============================================================================

def create_enhanced_dashboard(df, futures_ltp, symbol, flow_metrics, fetch_method, atm_info):
    """Enhanced dashboard with GEX + DEX + ATM Straddle analysis"""

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            '📊 Net GEX Profile with Volume',
            '📈 Delta Exposure (DEX) Profile with Volume',
            '🔄 GEX Flow (OI Changes) with Volume',
            '📉 DEX Flow with Volume',
            '🎯 Hedging Pressure Index with Volume',
            '⚡ Combined GEX+DEX Directional Bias',
            '💰 ATM Straddle Analysis',
            ''  # Empty subplot
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        vertical_spacing=0.10, horizontal_spacing=0.12,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )

    # Prepare volume scaling for all charts
    max_gex = df['Net_GEX_B'].abs().max()
    max_dex = df['Net_DEX_B'].abs().max()
    max_vol = df['Total_Volume'].max()

    if max_vol > 0:
        vol_scale_gex = (max_gex * 0.3) / max_vol
        vol_scale_dex = (max_dex * 0.3) / max_vol
        scaled_volume_gex = df['Total_Volume'] * vol_scale_gex
        scaled_volume_dex = df['Total_Volume'] * vol_scale_dex
    else:
        scaled_volume_gex = df['Total_Volume']
        scaled_volume_dex = df['Total_Volume']

    # Get S&R levels
    positive_gex_mask = df['Net_GEX_B'] > 0
    positive_gex = df[positive_gex_mask].nlargest(3, 'Net_GEX_B')

    negative_gex_mask = df['Net_GEX_B'] < 0
    negative_gex = df[negative_gex_mask].nsmallest(3, 'Net_GEX_B')

    # ========================================================================
    # CHART 1: Net GEX Profile with Volume
    # ========================================================================
    colors = ['green' if x > 0 else 'red' for x in df['Net_GEX_B']]
    fig.add_trace(go.Bar(y=df['Strike'], x=df['Net_GEX_B'], name='Net GEX',
                         orientation='h', marker_color=colors,
                         hovertemplate='<b>Strike:</b> %{y}<br><b>Net GEX:</b> %{x:.4f} B<extra></extra>'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(y=df['Strike'], x=scaled_volume_gex, name='Volume',
                             mode='lines+markers', line=dict(color='blue', width=2),
                             marker=dict(size=4),
                             hovertemplate='<b>Strike:</b> %{y}<br><b>Volume:</b> %{customdata:,.0f}<extra></extra>',
                             customdata=df['Total_Volume']), row=1, col=1)

    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="blue",
                  line_width=3, annotation_text="FUTURES", row=1, col=1)

    # Add S&R lines
    for idx, (_, row) in enumerate(positive_gex.iterrows()):
        if row['Strike'] < futures_ltp:
            fig.add_hline(y=row['Strike'], line_dash="dot", line_color="green",
                         line_width=1, opacity=0.5, annotation_text=f"S{idx+1}",
                         annotation_position="left", row=1, col=1)
        elif row['Strike'] > futures_ltp:
            fig.add_hline(y=row['Strike'], line_dash="dot", line_color="red",
                         line_width=1, opacity=0.5, annotation_text=f"R{idx+1}",
                         annotation_position="right", row=1, col=1)

    # ========================================================================
    # CHART 2: DEX Profile with Volume
    # ========================================================================
    dex_colors = ['green' if x > 0 else 'red' for x in df['Net_DEX_B']]
    fig.add_trace(go.Bar(y=df['Strike'], x=df['Net_DEX_B'], name='Net DEX',
                         orientation='h', marker_color=dex_colors,
                         hovertemplate='<b>Strike:</b> %{y}<br><b>Net DEX:</b> %{x:.4f} B<extra></extra>'),
                  row=1, col=2)

    fig.add_trace(go.Scatter(y=df['Strike'], x=scaled_volume_dex, name='Volume',
                             mode='lines+markers', line=dict(color='purple', width=2),
                             marker=dict(size=4),
                             hovertemplate='<b>Strike:</b> %{y}<br><b>Volume:</b> %{customdata:,.0f}<extra></extra>',
                             customdata=df['Total_Volume']), row=1, col=2)

    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="blue",
                  line_width=3, annotation_text="FUTURES", row=1, col=2)

    # ========================================================================
    # CHART 3: GEX Flow with Volume
    # ========================================================================
    flow_colors = ['green' if x > 0 else 'red' for x in df['Net_Flow_GEX_B']]
    fig.add_trace(go.Bar(y=df['Strike'], x=df['Net_Flow_GEX_B'], name='GEX Flow',
                         orientation='h', marker_color=flow_colors,
                         hovertemplate='<b>Strike:</b> %{y}<br><b>Flow GEX:</b> %{x:.4f} B<extra></extra>'),
                  row=2, col=1)

    max_flow = df['Net_Flow_GEX_B'].abs().max()
    vol_scale_flow = (max_flow * 0.3) / max_vol if max_vol > 0 else 1
    scaled_volume_flow = df['Total_Volume'] * vol_scale_flow

    fig.add_trace(go.Scatter(y=df['Strike'], x=scaled_volume_flow, name='Volume',
                             mode='lines+markers', line=dict(color='orange', width=2),
                             marker=dict(size=4),
                             hovertemplate='<b>Strike:</b> %{y}<br><b>Volume:</b> %{customdata:,.0f}<extra></extra>',
                             customdata=df['Total_Volume']), row=2, col=1)

    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="blue",
                  line_width=3, annotation_text="FUTURES", row=2, col=1)

    # ========================================================================
    # CHART 4: DEX Flow with Volume
    # ========================================================================
    dex_flow_colors = ['green' if x > 0 else 'red' for x in df['Net_Flow_DEX_B']]
    fig.add_trace(go.Bar(y=df['Strike'], x=df['Net_Flow_DEX_B'], name='DEX Flow',
                         orientation='h', marker_color=dex_flow_colors,
                         hovertemplate='<b>Strike:</b> %{y}<br><b>Flow DEX:</b> %{x:.4f} B<extra></extra>'),
                  row=2, col=2)

    max_dex_flow = df['Net_Flow_DEX_B'].abs().max()
    vol_scale_dex_flow = (max_dex_flow * 0.3) / max_vol if max_vol > 0 else 1
    scaled_volume_dex_flow = df['Total_Volume'] * vol_scale_dex_flow

    fig.add_trace(go.Scatter(y=df['Strike'], x=scaled_volume_dex_flow, name='Volume',
                             mode='lines+markers', line=dict(color='cyan', width=2),
                             marker=dict(size=4),
                             hovertemplate='<b>Strike:</b> %{y}<br><b>Volume:</b> %{customdata:,.0f}<extra></extra>',
                             customdata=df['Total_Volume']), row=2, col=2)

    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="blue",
                  line_width=3, annotation_text="FUTURES", row=2, col=2)

    # ========================================================================
    # CHART 5: Hedging Pressure with Volume
    # ========================================================================
    fig.add_trace(go.Bar(y=df['Strike'], x=df['Hedging_Pressure'], orientation='h',
                         marker=dict(color=df['Hedging_Pressure'], colorscale='RdYlGn',
                                   showscale=True, colorbar=dict(title="Pressure", x=1.15)),
                         name='Hedge Pressure',
                         hovertemplate='<b>Strike:</b> %{y}<br><b>Pressure:</b> %{x:.2f}<extra></extra>'),
                  row=3, col=1)

    max_pressure = df['Hedging_Pressure'].abs().max()
    vol_scale_pressure = (max_pressure * 0.3) / max_vol if max_vol > 0 else 1
    scaled_volume_pressure = df['Total_Volume'] * vol_scale_pressure

    fig.add_trace(go.Scatter(y=df['Strike'], x=scaled_volume_pressure, name='Volume',
                             mode='lines+markers', line=dict(color='magenta', width=2),
                             marker=dict(size=4),
                             hovertemplate='<b>Strike:</b> %{y}<br><b>Volume:</b> %{customdata:,.0f}<extra></extra>',
                             customdata=df['Total_Volume']), row=3, col=1)

    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="blue",
                  line_width=3, annotation_text="FUTURES", row=3, col=1)

    # ========================================================================
    # CHART 6: Combined GEX+DEX Bias
    # ========================================================================
    df['Combined_Signal'] = (df['Net_GEX_B'] + df['Net_DEX_B']) / 2
    combined_colors = ['green' if x > 0 else 'red' for x in df['Combined_Signal']]

    fig.add_trace(go.Bar(y=df['Strike'], x=df['Combined_Signal'], orientation='h',
                         marker_color=combined_colors, name='Combined Signal',
                         hovertemplate='<b>Strike:</b> %{y}<br><b>Combined:</b> %{x:.4f} B<extra></extra>'),
                  row=3, col=2)

    fig.add_trace(go.Scatter(y=df['Strike'], x=df['Net_Flow_DEX_B'],
                             name='DEX Flow Curve',
                             mode='lines', line=dict(color='yellow', width=3, dash='dash'),
                             hovertemplate='<b>Strike:</b> %{y}<br><b>DEX Flow:</b> %{x:.4f} B<extra></extra>'),
                  row=3, col=2)

    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="blue",
                  line_width=3, annotation_text="FUTURES", row=3, col=2)

    # ========================================================================
    # CHART 7: ATM STRADDLE ANALYSIS (NEW!)
    # ========================================================================
    atm_strike = atm_info['atm_strike']
    atm_call_premium = atm_info['atm_call_premium']
    atm_put_premium = atm_info['atm_put_premium']
    atm_straddle_premium = atm_info['atm_straddle_premium']

    # Create straddle payoff diagram
    strike_range = np.linspace(atm_strike * 0.90, atm_strike * 1.10, 100)

    # Long straddle payoff at expiry
    call_payoff = np.maximum(strike_range - atm_strike, 0) - atm_call_premium
    put_payoff = np.maximum(atm_strike - strike_range, 0) - atm_put_premium
    straddle_payoff = call_payoff + put_payoff

    # Breakeven points
    upper_breakeven = atm_strike + atm_straddle_premium
    lower_breakeven = atm_strike - atm_straddle_premium

    fig.add_trace(go.Scatter(x=strike_range, y=straddle_payoff,
                             name='Straddle P&L',
                             mode='lines', line=dict(color='purple', width=3),
                             hovertemplate='<b>Price:</b> %{x:.0f}<br><b>P&L:</b> ₹%{y:.2f}<extra></extra>'),
                  row=4, col=1)

    fig.add_trace(go.Scatter(x=strike_range, y=call_payoff,
                             name='Call P&L',
                             mode='lines', line=dict(color='green', width=2, dash='dot'),
                             hovertemplate='<b>Price:</b> %{x:.0f}<br><b>Call P&L:</b> ₹%{y:.2f}<extra></extra>'),
                  row=4, col=1)

    fig.add_trace(go.Scatter(x=strike_range, y=put_payoff,
                             name='Put P&L',
                             mode='lines', line=dict(color='red', width=2, dash='dot'),
                             hovertemplate='<b>Price:</b> %{x:.0f}<br><b>Put P&L:</b> ₹%{y:.2f}<extra></extra>'),
                  row=4, col=1)

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=4, col=1)

    # Add ATM strike line
    fig.add_vline(x=atm_strike, line_dash="solid", line_color="blue",
                  line_width=2, annotation_text=f"ATM: {atm_strike}",
                  row=4, col=1)

    # Add breakeven lines
    fig.add_vline(x=upper_breakeven, line_dash="dash", line_color="orange",
                  line_width=2, annotation_text=f"Upper BE: {upper_breakeven:.0f}",
                  row=4, col=1)
    fig.add_vline(x=lower_breakeven, line_dash="dash", line_color="orange",
                  line_width=2, annotation_text=f"Lower BE: {lower_breakeven:.0f}",
                  row=4, col=1)

    # Add current futures price
    fig.add_vline(x=futures_ltp, line_dash="solid", line_color="red",
                  line_width=2, annotation_text=f"Current: {futures_ltp:.0f}",
                  row=4, col=1)

    # Update axes
    fig.update_xaxes(title_text="Net GEX (B)", row=1, col=1)
    fig.update_xaxes(title_text="Net DEX (B)", row=1, col=2)
    fig.update_xaxes(title_text="GEX Flow (B)", row=2, col=1)
    fig.update_xaxes(title_text="DEX Flow (B)", row=2, col=2)
    fig.update_xaxes(title_text="Pressure", row=3, col=1)
    fig.update_xaxes(title_text="Combined (B)", row=3, col=2)
    fig.update_xaxes(title_text="Underlying Price", row=4, col=1)

    fig.update_yaxes(title_text="Strike", row=1, col=1)
    fig.update_yaxes(title_text="Strike", row=1, col=2)
    fig.update_yaxes(title_text="Strike", row=2, col=1)
    fig.update_yaxes(title_text="Strike", row=2, col=2)
    fig.update_yaxes(title_text="Strike", row=3, col=1)
    fig.update_yaxes(title_text="Strike", row=3, col=2)
    fig.update_yaxes(title_text="Profit/Loss (₹)", row=4, col=1)

    timestamp = datetime.now().strftime('%H:%M:%S')
    fig.update_layout(
        title=dict(
            text=f'<b>{symbol} - GEX + DEX Analysis (Futures: {futures_ltp:,.2f} via {fetch_method})</b><br>' +
                 f'<sup>{timestamp} | GEX: {flow_metrics["gex_near_bias"]} | DEX: {flow_metrics["dex_near_bias"]} | ' +
                 f'Combined: {flow_metrics["combined_bias"]} | ATM Straddle: ₹{atm_straddle_premium:.2f}</sup>',
            font=dict(size=14)
        ),
        height=1600, showlegend=True, template='plotly_white', hovermode='closest'
    )

    fig.show()

# ============================================================================
# SECTION 3: TRADING STRATEGIES BASED ON LIVE SETUP
# ============================================================================

def generate_trading_strategies(df, futures_ltp, flow_metrics, atm_info):
    """
    Generate option trading strategies based on GEX+DEX analysis
    MODIFIED: Updated interpretation based on new GEX flow logic
    """
    print("\n" + "="*80)
    print("💼 SECTION 3: OPTION TRADING STRATEGIES")
    print("="*80)

    # Get key levels
    positive_gex_mask = df['Net_GEX_B'] > 0
    positive_gex = df[positive_gex_mask].nlargest(5, 'Net_GEX_B')

    supports_below = positive_gex[positive_gex['Strike'] < futures_ltp]
    resistances_above = positive_gex[positive_gex['Strike'] > futures_ltp]

    nearest_support = supports_below.iloc[0] if not supports_below.empty else None
    nearest_resistance = resistances_above.iloc[0] if not resistances_above.empty else None

    # Extract metrics
    gex_bias = flow_metrics['gex_near_total']
    dex_bias = flow_metrics['dex_near_total']
    combined_signal = flow_metrics['combined_signal']

    # ATM Straddle info
    atm_strike = atm_info['atm_strike']
    atm_straddle_premium = atm_info['atm_straddle_premium']

    print(f"\n📊 MARKET SETUP ANALYSIS:")
    print(f"{'Metric':<30} {'Value':>20} {'Bias':>35}")
    print("-"*85)
    print(f"{'GEX Flow (Near-term):':<30} {gex_bias:>20.2f} {flow_metrics['gex_near_bias']:>35}")
    print(f"{'DEX Flow (Near-term):':<30} {dex_bias:>20.2f} {flow_metrics['dex_near_bias']:>35}")
    print(f"{'Combined Signal:':<30} {combined_signal:>20.2f} {flow_metrics['combined_bias']:>35}")
    print(f"{'ATM Strike:':<30} {atm_strike:>20,.0f}")
    print(f"{'ATM Straddle Premium:':<30} {atm_straddle_premium:>20.2f}")

    if nearest_support is not None:
        support_strike = float(nearest_support['Strike'])
        support_pct = ((futures_ltp - support_strike)/futures_ltp*100)
        print(f"{'Nearest Support:':<30} {support_strike:>20,.0f} {support_pct:>34.2f}%")
    if nearest_resistance is not None:
        resistance_strike = float(nearest_resistance['Strike'])
        resistance_pct = ((resistance_strike - futures_ltp)/futures_ltp*100)
        print(f"{'Nearest Resistance:':<30} {resistance_strike:>20,.0f} {resistance_pct:>34.2f}%")

    # ========================================================================
    # STRATEGY SELECTION LOGIC (MODIFIED)
    # ========================================================================
    print("\n" + "="*80)
    print("🎯 RECOMMENDED STRATEGIES:")
    print("="*80)

    strategies = []

    # SCENARIO 1: Strong Positive GEX (>50) - Sideways to Bullish
    if gex_bias > 50:
        print("\n📌 PRIMARY STRATEGY: SIDEWAYS TO BULLISH SETUP (Strong Positive GEX)")
        print("-"*75)

        if nearest_support is not None and nearest_resistance is not None:
            # Iron Condor
            strategies.append({
                'name': '🦅 Iron Condor',
                'rationale': 'Strong positive GEX → Sideways movement expected, sell premium',
                'setup': f"Sell {int(futures_ltp)} CE + Buy {int(resistance_strike)} CE | " +
                         f"Sell {int(futures_ltp)} PE + Buy {int(support_strike)} PE",
                'max_profit': 'Net Premium Received',
                'max_loss': 'Limited to strike width minus premium',
                'risk_level': '⚠️ MODERATE',
                'conditions': 'Hold if price stays between support and resistance'
            })

            # Bull Call Spread (if DEX also positive)
            if dex_bias > 0:
                strategies.append({
                    'name': '📈 Bull Call Spread',
                    'rationale': 'Positive GEX + Bullish DEX → Mild upside with limited risk',
                    'setup': f"Buy {int(futures_ltp)} CE + Sell {int(resistance_strike)} CE",
                    'max_profit': 'Strike width - Premium',
                    'max_loss': 'Premium Paid',
                    'risk_level': '✅ LOW-MODERATE',
                    'conditions': 'Bullish bias within resistance zone'
                })

        # Straddle selling (for high positive GEX)
        strategies.append({
            'name': '🔒 Short ATM Straddle',
            'rationale': 'Strong positive GEX → Low volatility, collect premium',
            'setup': f"Sell {int(atm_strike)} CE + Sell {int(atm_strike)} PE (ATM Straddle: ₹{atm_straddle_premium:.2f})",
            'max_profit': f'₹{atm_straddle_premium:.2f} per lot',
            'max_loss': 'UNLIMITED (use stops or hedges)',
            'risk_level': '⚠️⚠️ HIGH (Requires experience)',
            'conditions': 'Price stays near ATM, low volatility persists'
        })

    # SCENARIO 2: Negative GEX (<-50) - Bearish & High Volatility
    elif gex_bias < -50:
        print("\n📌 PRIMARY STRATEGY: HIGH VOLATILITY EXPECTED (Negative GEX - Bearish)")
        print("-"*75)

        # Long Straddle
        strategies.append({
            'name': '🎭 Long ATM Straddle',
            'rationale': 'Negative GEX → High volatility expected, buy options',
            'setup': f"Buy {int(atm_strike)} CE + Buy {int(atm_strike)} PE (Cost: ₹{atm_straddle_premium:.2f})",
            'max_profit': 'Unlimited (both directions)',
            'max_loss': f'Premium Paid (₹{atm_straddle_premium:.2f})',
            'risk_level': '⚠️ HIGH (Needs big move)',
            'conditions': f'Price must move beyond ₹{atm_straddle_premium:.2f} to profit'
        })

        if dex_bias < -20:
            # Bearish directional play
            strategies.append({
                'name': '📉 Long Put',
                'rationale': 'Negative GEX + Bearish DEX → Downside breakout likely',
                'setup': f"Buy {int(futures_ltp)} PE (ATM) or {int(futures_ltp - 100)} PE (OTM)",
                'max_profit': 'Substantial (down to zero)',
                'max_loss': 'Premium Paid',
                'risk_level': '⚠️ HIGH (Limited to premium)',
                'conditions': 'Breakdown below support expected'
            })

            strategies.append({
                'name': '🐻 Bear Put Spread',
                'rationale': 'Reduce cost while maintaining downside exposure',
                'setup': f"Buy {int(futures_ltp)} PE + Sell {int(futures_ltp - 200)} PE",
                'max_profit': 'Strike width - Premium',
                'max_loss': 'Premium Paid',
                'risk_level': '✅ MODERATE',
                'conditions': 'Defined risk with bearish bias'
            })
        elif dex_bias > 20:
            # Counter-trend opportunity
            strategies.append({
                'name': '🚀 Long Call (Counter-trend)',
                'rationale': 'Negative GEX (volatility) + Bullish DEX → Upside volatility',
                'setup': f"Buy {int(futures_ltp)} CE (ATM) or {int(futures_ltp + 100)} CE (OTM)",
                'max_profit': 'Unlimited',
                'max_loss': 'Premium Paid',
                'risk_level': '⚠️ HIGH (Limited to premium)',
                'conditions': 'Volatile upside breakout expected'
            })

    # SCENARIO 3: Mild GEX (-50 to +50) - Mixed signals
    else:
        print("\n📌 PRIMARY STRATEGY: CAUTIOUS APPROACH (Mixed Signals)")
        print("-"*75)

        if abs(dex_bias) > 20:
            # Follow DEX bias
            if dex_bias > 0:
                strategies.append({
                    'name': '📈 Bull Call Spread',
                    'rationale': 'Neutral GEX but bullish DEX → Defined risk bullish play',
                    'setup': f"Buy {int(futures_ltp)} CE + Sell {int(futures_ltp + 100)} CE",
                    'max_profit': 'Strike width - Premium',
                    'max_loss': 'Premium Paid',
                    'risk_level': '✅ MODERATE',
                    'conditions': 'Mild upside move expected'
                })
            else:
                strategies.append({
                    'name': '📉 Bear Put Spread',
                    'rationale': 'Neutral GEX but bearish DEX → Defined risk bearish play',
                    'setup': f"Buy {int(futures_ltp)} PE + Sell {int(futures_ltp - 100)} PE",
                    'max_profit': 'Strike width - Premium',
                    'max_loss': 'Premium Paid',
                    'risk_level': '✅ MODERATE',
                    'conditions': 'Mild downside move expected'
                })
        else:
            # Very uncertain
            strategies.append({
                'name': '⏸️ WAIT FOR CLARITY',
                'rationale': 'Mixed signals from both GEX and DEX → No clear edge',
                'setup': 'Stay in cash or small positions only',
                'max_profit': 'N/A',
                'max_loss': 'Opportunity cost',
                'risk_level': '✅ ZERO RISK',
                'conditions': 'Wait for stronger directional signals'
            })

    # ========================================================================
    # PRINT STRATEGIES
    # ========================================================================
    for idx, strategy in enumerate(strategies, 1):
        print(f"\n{'='*75}")
        print(f"STRATEGY #{idx}: {strategy['name']}")
        print(f"{'='*75}")
        print(f"{'Rationale:':<20} {strategy['rationale']}")
        print(f"{'Setup:':<20} {strategy['setup']}")
        print(f"{'Max Profit:':<20} {strategy['max_profit']}")
        print(f"{'Max Loss:':<20} {strategy['max_loss']}")
        print(f"{'Risk Level:':<20} {strategy['risk_level']}")
        print(f"{'Conditions:':<20} {strategy['conditions']}")

    # ========================================================================
    # MODIFIED GEX INTERPRETATION
    # ========================================================================
    print("\n" + "="*80)
    print("📖 MODIFIED GEX FLOW INTERPRETATION:")
    print("="*80)
    print("""
✅ POSITIVE GEX FLOW (Sideways to Bullish):
   • 5 closest strikes with positive Net GEX near spot
   • Market makers delta hedge by BUYING underlying on dips
   • Acts as price SUPPORT → sideways to bullish movement
   • STRATEGY: Sell premium (Iron Condor, Credit Spreads, Short Straddle)

❌ NEGATIVE GEX FLOW (Bearish & High Volatility):
   • 5 closest strikes with negative Net GEX near spot
   • Market makers delta hedge by SELLING underlying on rallies
   • Acts as price RESISTANCE → bearish and volatile movement
   • STRATEGY: Buy volatility (Long Straddle, Long Options)

⚖️ NEUTRAL GEX:
   • Balanced positive and negative GEX
   • No strong hedging bias from market makers
   • Follow DEX (Delta) bias for directional plays
    """)

    # ========================================================================
    # RISK MANAGEMENT
    # ========================================================================
    print("\n" + "="*80)
    print("⚠️ RISK MANAGEMENT RULES:")
    print("="*80)
    print(f"""
1. 🛡️ POSITION SIZING:
   • Never risk more than 2% of capital per trade
   • For spreads: Risk defined by strike width minus premium
   • For long options: Max loss = Premium paid
   • For short straddles: USE STOP LOSSES or protective wings

2. 🎯 ENTRY TIMING:
   • Wait for price to approach key GEX support/resistance levels
   • Enter when combined GEX+DEX bias aligns with your strategy
   • Avoid trading during first 15 mins and last 30 mins

3. 🚪 EXIT RULES:
   • Take profit at 50-70% of max profit for spreads
   • Use trailing stops for long options (20-30% of unrealized profit)
   • Exit immediately if GEX/DEX bias changes significantly
   • For short straddle: Exit if price moves > ₹{atm_straddle_premium*0.5:.2f} from ATM

4. ⏰ TIME DECAY:
   • Selling strategies: Theta works in your favor (premium decay)
   • Buying strategies: Monitor theta - don't hold too close to expiry
   • Weekly options: Higher gamma risk, faster decay

5. 📊 MONITORING:
   • Check GEX+DEX every 1-3 hours during market
   • Watch for changes in flow metrics (OI changes)
   • If combined bias flips, reassess positions immediately
   • Monitor ATM straddle premium for volatility clues
    """)

    print("\n" + "="*80)
    print("💡 IMPORTANT NOTES:")
    print("="*80)
    print("""
• These strategies are based on CURRENT market structure
• GEX/DEX levels change throughout the day
• Always use stop losses and defined risk strategies
• Past performance does not guarantee future results
• Consult with a financial advisor before trading
• Short straddles require experience and strict risk management
    """)
    print("="*80)

    return strategies

# ============================================================================
# ENHANCED TABLE WITH DUAL FLOW
# ============================================================================

def create_enhanced_flow_table(df, futures_ltp, flow_metrics, atm_info):
    """Enhanced table showing dual flow metrics with ATM straddle info"""

    df_unique = df.drop_duplicates(subset=['Strike']).copy()

    # Get 5 positive GEX strikes closest to spot
    positive_gex_df = df_unique[df_unique['Net_GEX_B'] > 0].copy()
    positive_gex_df['Distance'] = abs(positive_gex_df['Strike'] - futures_ltp)
    positive_gex_strikes = positive_gex_df.nsmallest(5, 'Distance')

    # Get 5 negative GEX strikes closest to spot
    negative_gex_df = df_unique[df_unique['Net_GEX_B'] < 0].copy()
    negative_gex_df['Distance'] = abs(negative_gex_df['Strike'] - futures_ltp)
    negative_gex_strikes = negative_gex_df.nsmallest(5, 'Distance')

    # Combine and sort
    relevant_strikes = pd.concat([positive_gex_strikes, negative_gex_strikes]).sort_values('Strike')

    table_data = []
    for idx, row in relevant_strikes.iterrows():
        strike = row['Strike']
        position = "🔼 ABOVE" if strike > futures_ltp else "🔽 BELOW"
        if abs(strike - futures_ltp) < 10:
            position = "⚡ ATM"

        net_gex = row['Net_GEX_B']
        gex_indicator = f"🟢 +{net_gex:.4f}B" if net_gex > 0.001 else f"🔴 {net_gex:.4f}B" if net_gex < -0.001 else f"⚪ {net_gex:.4f}B"

        net_dex = row['Net_DEX_B']
        dex_indicator = f"🟢 +{net_dex:.4f}B" if net_dex > 0.001 else f"🔴 {net_dex:.4f}B" if net_dex < -0.001 else f"⚪ {net_dex:.4f}B"

        table_data.append([
            position, f"{strike:,.0f}", gex_indicator, dex_indicator,
            f"{row['Total_Volume']:,.0f}",
            f"{row['Call_OI']:,.0f}", f"{row['Put_OI']:,.0f}"
        ])

    headers = ['Position', 'Strike', 'Net GEX', 'Net DEX', 'Volume', 'Call OI', 'Put OI']

    print("\n" + "="*120)
    print("📋 GEX + DEX ANALYSIS - STRIKES NEAR FUTURES LTP")
    print("="*120)
    print(tabulate(table_data, headers=headers, tablefmt='fancy_grid'))

    # Add dual flow summary
    print("\n" + "="*120)
    print("📊 DUAL FLOW ANALYSIS SUMMARY (GEX + DEX)")
    print("="*120)

    print(f"\n{'GEX METRICS:':<40} {'Near-Term':<25} {'Total (All)':<20}")
    print("-"*85)
    print(f"{'Positive GEX Flow:':<40} {flow_metrics['gex_near_positive']:>20.4f} B {flow_metrics['gex_total_positive']:>15.4f} B")
    print(f"{'Negative GEX Flow:':<40} {flow_metrics['gex_near_negative']:>20.4f} B {flow_metrics['gex_total_negative']:>15.4f} B")
    print(f"{'Net GEX Flow:':<40} {flow_metrics['gex_near_total']:>20.4f} B {flow_metrics['gex_total_all']:>15.4f} B")
    print(f"{'GEX Bias:':<40} {flow_metrics['gex_near_bias']:>25} {flow_metrics['gex_total_bias']:>20}")

    print(f"\n{'DEX METRICS:':<40} {'Near-Term':<25} {'Total (All)':<20}")
    print("-"*85)
    print(f"{'Positive DEX Flow:':<40} {flow_metrics['dex_near_positive']:>20.4f} B {flow_metrics['dex_total_positive']:>15.4f} B")
    print(f"{'Negative DEX Flow:':<40} {flow_metrics['dex_near_negative']:>20.4f} B {flow_metrics['dex_total_negative']:>15.4f} B")
    print(f"{'Net DEX Flow:':<40} {flow_metrics['dex_near_total']:>20.4f} B {flow_metrics['dex_total_all']:>15.4f} B")
    print(f"{'DEX Bias:':<40} {flow_metrics['dex_near_bias']:>25} {flow_metrics['dex_total_bias']:>20}")

    print(f"\n{'COMBINED SIGNAL:':<40} {flow_metrics['combined_signal']:>20.4f} B")
    print(f"{'COMBINED BIAS:':<40} {flow_metrics['combined_bias']:>25}")

    print(f"\n{'ATM STRADDLE INFO:':<40}")
    print("-"*85)
    print(f"{'ATM Strike:':<40} {atm_info['atm_strike']:>20,.0f}")
    print(f"{'ATM Call Premium:':<40} ₹{atm_info['atm_call_premium']:>19.2f}")
    print(f"{'ATM Put Premium:':<40} ₹{atm_info['atm_put_premium']:>19.2f}")
    print(f"{'ATM Straddle Premium:':<40} ₹{atm_info['atm_straddle_premium']:>19.2f}")
    print(f"{'Upper Breakeven:':<40} {atm_info['atm_strike'] + atm_info['atm_straddle_premium']:>20,.2f}")
    print(f"{'Lower Breakeven:':<40} {atm_info['atm_strike'] - atm_info['atm_straddle_premium']:>20,.2f}")
    print("="*120)

    print("\n💡 MODIFIED INTERPRETATION:")
    print(f"   • GEX (Gamma): {flow_metrics['gex_near_bias']}")
    print(f"      └─ Positive GEX = Sideways/Bullish (Market makers support dips)")
    print(f"      └─ Negative GEX = Bearish/Volatile (Market makers sell rallies)")
    print(f"   • DEX (Delta): {flow_metrics['dex_near_bias']} → Directional bias")
    print(f"   • Combined: {flow_metrics['combined_bias']} → Overall prediction")
    print(f"   • ATM Straddle: ₹{atm_info['atm_straddle_premium']:.2f} → Volatility expectation")

    if flow_metrics['gex_near_bias'] != flow_metrics['dex_near_bias']:
        print(f"   ⚠️  DIVERGENCE: GEX and DEX showing different biases!")
        print(f"       → GEX controls volatility/direction, DEX shows hedging flow")
    print("="*120)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_enhanced_analysis(symbol="NIFTY", strikes_range=12, expiry_index=0):
    """Run enhanced GEX + DEX analysis with trading strategies"""

    print(f"🚀 ENHANCED GEX + DEX ANALYSIS - {symbol}")
    print(f"📊 Modified GEX Flow Logic + Delta Exposure + ATM Straddle Chart")
    print("="*80)

    calculator = EnhancedGEXDEXCalculator()
    df, futures_ltp, fetch_method, atm_info = calculator.fetch_and_calculate_gex_dex(
        symbol, strikes_range, expiry_index
    )

    if df is not None and atm_info is not None:
        flow_metrics = calculate_dual_gex_dex_flow(df, futures_ltp)
        create_enhanced_flow_table(df, futures_ltp, flow_metrics, atm_info)
        create_enhanced_dashboard(df, futures_ltp, symbol, flow_metrics, fetch_method, atm_info)
        generate_trading_strategies(df, futures_ltp, flow_metrics, atm_info)

        filename = f"{symbol}_GEX_DEX_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Data saved: {filename}")

        return df, flow_metrics, atm_info
    else:
        print("❌ Analysis failed")
        return None, None, None

# ============================================================================
# REAL-TIME MODE
# ============================================================================

def run_realtime_analysis(symbol="NIFTY", strikes_range=12, expiry_index=0,
                          update_interval_seconds=180, max_iterations=None):
    """Real-time mode with enhanced features"""

    print("="*80)
    print("🔴 REAL-TIME GEX + DEX ANALYSIS (MODIFIED)")
    print("="*80)
    print(f"📊 Symbol: {symbol}")
    print(f"⏱️  Interval: {update_interval_seconds}s")
    print(f"🛑 Stop: Ctrl+C")
    print("="*80)

    iteration = 0
    try:
        while True:
            iteration += 1
            try:
                clear_output(wait=True)
            except:
                pass

            print(f"\n{'='*80}")
            print(f"🔄 UPDATE #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)

            df, flow_metrics, atm_info = run_enhanced_analysis(symbol, strikes_range, expiry_index)

            if max_iterations and iteration >= max_iterations:
                print(f"\n✅ Completed {iteration} updates")
                break

            print(f"\n⏳ Next update in {update_interval_seconds}s...")
            time.sleep(update_interval_seconds)

    except KeyboardInterrupt:
        print(f"\n🛑 Stopped after {iteration} updates")

# ============================================================================
# EXECUTE
# ============================================================================

if __name__ == "__main__":
    SYMBOL = "NIFTY"
    STRIKES_RANGE = 12
    EXPIRY_INDEX = 0

    # Single analysis
    df, flow_metrics, atm_info = run_enhanced_analysis(SYMBOL, STRIKES_RANGE, EXPIRY_INDEX)

    # Real-time (uncomment to use)
    run_realtime_analysis(SYMBOL, STRIKES_RANGE, EXPIRY_INDEX, update_interval_seconds=60)

    print("\n✅ MODIFIED ANALYSIS COMPLETE!")
    print("\n" + "="*80)
    print("📝 KEY MODIFICATIONS APPLIED:")
    print("="*80)
    print("""
1. ✅ MODIFIED GEX FLOW CALCULATION:
   - Positive GEX Flow = Sum of 5 strikes with POSITIVE Net GEX closest to spot
   - Negative GEX Flow = Sum of 5 strikes with NEGATIVE Net GEX closest to spot
   - Continuous strikes near the spot price

2. ✅ UPDATED GEX INTERPRETATION:
   - Positive GEX = Sideways to Bullish (MM support)
   - Negative GEX = Bearish & High Volatility (MM resistance)

3. ✅ NEW ATM STRADDLE CHART (7th Chart):
   - Displays ATM straddle payoff diagram
   - Shows breakeven points
   - Includes Call, Put, and combined P&L
   - Useful for volatility trading strategies

4. ✅ ENHANCED TRADING STRATEGIES:
   - Updated strategy recommendations based on new GEX logic
   - Added short straddle strategies for high positive GEX
   - Added long straddle strategies for negative GEX
   - Included ATM straddle premium in all recommendations

5. ✅ ALL ORIGINAL FEATURES RETAINED:
   - Delta Exposure (DEX) Analysis
   - Volume overlay on all 6 main charts
   - Combined GEX + DEX bias
   - Trading strategies generation
   - Risk management rules
   - Real-time mode
""")
    print("="*80)
