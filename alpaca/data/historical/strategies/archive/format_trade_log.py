#!/usr/bin/env python3
"""
Format the corrected trade log in a readable format
"""

def format_trade_log():
    print('📊 DETAILED TRADE LOG - CORRECTED POSITION SIZING BACKTEST')
    print('=' * 80)
    print()
    
    big_winners = []
    big_losses = []
    
    with open('corrected_position_sizing_trades.csv', 'r') as f:
        lines = f.readlines()
        headers = lines[0].strip().split(',')
        
        for i, line in enumerate(lines[1:], 1):
            values = line.strip().split(',')
            
            # Extract key values
            date = values[0]
            vix = float(values[1])
            conviction = int(values[3])
            contracts = int(values[5])
            net_pnl = float(values[14])
            return_pct = float(values[15])
            account_balance = float(values[16])
            
            status = '🟢 WIN' if net_pnl > 0 else '🔴 LOSS'
            
            print(f'Trade #{i:2d} | {date} | VIX: {vix:5.1f} | Conviction: {conviction}')
            print(f'   Position: {contracts} contracts | Risk: $1,350')
            print(f'   Result: {status} | P&L: ${net_pnl:+8,.0f} | Return: {return_pct:+6.1f}%')
            print(f'   Account: ${account_balance:,.0f}')
            print()
            
            # Track big winners and losses
            if net_pnl > 5000:
                big_winners.append((date, net_pnl, return_pct))
            if net_pnl < -1000:
                big_losses.append((date, net_pnl, return_pct))
    
    print('📈 BIG WINNERS (>$5,000):')
    for date, pnl, ret in big_winners:
        print(f'   {date}: ${pnl:,.0f} ({ret:+.0f}%)')
    
    print()
    print('📉 NOTABLE LOSSES (<-$1,000):')
    for date, pnl, ret in big_losses:
        print(f'   {date}: ${pnl:,.0f} ({ret:+.0f}%)')

if __name__ == "__main__":
    format_trade_log() 