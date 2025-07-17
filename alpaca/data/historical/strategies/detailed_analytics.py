"""
Detailed Analytics for High Frequency 0DTE Strategy

Comprehensive analysis of trading results including:
- Profit & Loss breakdown
- Win/Loss rates  
- Risk metrics
- Performance by signal type
- Exit reason analysis
- Statistical summaries
- Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingAnalytics:
    """Comprehensive trading analytics for 0DTE strategy"""
    
    def __init__(self):
        self.trades_df = None
        self.daily_df = None
        
    def analyze_demo_results(self, demo_results: list) -> dict:
        """
        Analyze demo trading results and generate comprehensive analytics
        """
        # Flatten all trades from demo results
        all_trades = []
        daily_summary = []
        
        for day_result in demo_results:
            daily_summary.append({
                'date': day_result['date'],
                'signals': day_result['signals'], 
                'trades': day_result['trades'],
                'pnl': day_result['pnl']
            })
            
            for trade in day_result['trade_details']:
                all_trades.append(trade)
        
        if not all_trades:
            return {'error': 'No trades to analyze'}
        
        # Create DataFrames
        self.trades_df = pd.DataFrame(all_trades)
        self.daily_df = pd.DataFrame(daily_summary)
        
        # Generate comprehensive analytics
        analytics = {
            'overview': self._calculate_overview_metrics(),
            'pnl_analysis': self._analyze_pnl(),
            'win_loss_analysis': self._analyze_win_loss(),
            'signal_analysis': self._analyze_by_signal_type(),
            'exit_analysis': self._analyze_exit_reasons(),
            'risk_metrics': self._calculate_risk_metrics(),
            'daily_performance': self._analyze_daily_performance(),
            'position_sizing': self._analyze_position_sizing(),
            'statistical_summary': self._calculate_statistics()
        }
        
        return analytics
    
    def _calculate_overview_metrics(self) -> dict:
        """Calculate high-level overview metrics"""
        total_trades = len(self.trades_df)
        total_pnl = self.trades_df['pnl'].sum()
        trading_days = len(self.daily_df)
        
        return {
            'total_trades': total_trades,
            'trading_days': trading_days,
            'avg_trades_per_day': total_trades / trading_days,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / total_trades,
            'avg_daily_pnl': total_pnl / trading_days,
            'capital_efficiency': f"{total_pnl / 25000 * 100:.2f}%" if total_pnl != 0 else "0%"
        }
    
    def _analyze_pnl(self) -> dict:
        """Detailed P&L analysis"""
        pnl_series = self.trades_df['pnl']
        
        return {
            'total_pnl': pnl_series.sum(),
            'gross_profit': pnl_series[pnl_series > 0].sum(),
            'gross_loss': pnl_series[pnl_series < 0].sum(),
            'max_win': pnl_series.max(),
            'max_loss': pnl_series.min(),
            'avg_win': pnl_series[pnl_series > 0].mean() if len(pnl_series[pnl_series > 0]) > 0 else 0,
            'avg_loss': pnl_series[pnl_series < 0].mean() if len(pnl_series[pnl_series < 0]) > 0 else 0,
            'profit_factor': abs(pnl_series[pnl_series > 0].sum() / pnl_series[pnl_series < 0].sum()) if pnl_series[pnl_series < 0].sum() != 0 else float('inf'),
            'pnl_std': pnl_series.std(),
            'pnl_variance': pnl_series.var()
        }
    
    def _analyze_win_loss(self) -> dict:
        """Win/Loss rate analysis"""
        winning_trades = self.trades_df[self.trades_df['pnl'] > 0]
        losing_trades = self.trades_df[self.trades_df['pnl'] < 0]
        breakeven_trades = self.trades_df[self.trades_df['pnl'] == 0]
        
        total_trades = len(self.trades_df)
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'breakeven_trades': len(breakeven_trades),
            'win_rate': (len(winning_trades) / total_trades) * 100,
            'loss_rate': (len(losing_trades) / total_trades) * 100,
            'breakeven_rate': (len(breakeven_trades) / total_trades) * 100,
            'win_loss_ratio': len(winning_trades) / len(losing_trades) if len(losing_trades) > 0 else float('inf'),
            'expectancy': self.trades_df['pnl'].mean(),
            'largest_win_streak': self._calculate_win_streak(winning=True),
            'largest_loss_streak': self._calculate_win_streak(winning=False)
        }
    
    def _calculate_win_streak(self, winning=True) -> int:
        """Calculate longest winning or losing streak"""
        if winning:
            wins = (self.trades_df['pnl'] > 0).astype(int)
        else:
            wins = (self.trades_df['pnl'] < 0).astype(int)
        
        max_streak = 0
        current_streak = 0
        
        for win in wins:
            if win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
                
        return max_streak
    
    def _analyze_by_signal_type(self) -> dict:
        """Analyze performance by signal type (CALL vs PUT)"""
        call_trades = self.trades_df[self.trades_df['signal'] == 'BUY_CALL']
        put_trades = self.trades_df[self.trades_df['signal'] == 'BUY_PUT']
        
        call_analysis = self._analyze_subset(call_trades, "CALL")
        put_analysis = self._analyze_subset(put_trades, "PUT")
        
        return {
            'call_performance': call_analysis,
            'put_performance': put_analysis,
            'better_performer': 'CALL' if call_analysis['total_pnl'] > put_analysis['total_pnl'] else 'PUT'
        }
    
    def _analyze_subset(self, subset_df: pd.DataFrame, label: str) -> dict:
        """Analyze a subset of trades"""
        if len(subset_df) == 0:
            return {'trades': 0, 'total_pnl': 0, 'win_rate': 0, 'avg_pnl': 0}
        
        winning = subset_df[subset_df['pnl'] > 0]
        
        return {
            'trades': len(subset_df),
            'total_pnl': subset_df['pnl'].sum(),
            'win_rate': (len(winning) / len(subset_df)) * 100,
            'avg_pnl': subset_df['pnl'].mean(),
            'max_win': subset_df['pnl'].max(),
            'max_loss': subset_df['pnl'].min(),
            'avg_confidence': subset_df['confidence'].mean()
        }
    
    def _analyze_exit_reasons(self) -> dict:
        """Analyze performance by exit reason"""
        exit_counts = self.trades_df['exit_reason'].value_counts()
        exit_analysis = {}
        
        for exit_reason in exit_counts.index:
            subset = self.trades_df[self.trades_df['exit_reason'] == exit_reason]
            exit_analysis[exit_reason] = {
                'count': len(subset),
                'percentage': (len(subset) / len(self.trades_df)) * 100,
                'total_pnl': subset['pnl'].sum(),
                'avg_pnl': subset['pnl'].mean(),
                'win_rate': (len(subset[subset['pnl'] > 0]) / len(subset)) * 100
            }
        
        return exit_analysis
    
    def _calculate_risk_metrics(self) -> dict:
        """Calculate risk-adjusted performance metrics"""
        pnl_series = self.trades_df['pnl']
        daily_pnl = self.daily_df['pnl']
        
        # Sharpe-like ratio (using daily returns)
        if daily_pnl.std() != 0:
            sharpe_ratio = daily_pnl.mean() / daily_pnl.std()
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative_pnl = pnl_series.cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        # Risk metrics
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': pnl_series.sum() / abs(max_drawdown) if max_drawdown != 0 else float('inf'),
            'volatility': pnl_series.std(),
            'downside_deviation': pnl_series[pnl_series < 0].std(),
            'value_at_risk_95': np.percentile(pnl_series, 5),
            'expected_shortfall_95': pnl_series[pnl_series <= np.percentile(pnl_series, 5)].mean()
        }
    
    def _analyze_daily_performance(self) -> dict:
        """Analyze daily performance patterns"""
        daily_stats = {
            'best_day': self.daily_df['pnl'].max(),
            'worst_day': self.daily_df['pnl'].min(),
            'avg_daily_pnl': self.daily_df['pnl'].mean(),
            'profitable_days': len(self.daily_df[self.daily_df['pnl'] > 0]),
            'losing_days': len(self.daily_df[self.daily_df['pnl'] < 0]),
            'breakeven_days': len(self.daily_df[self.daily_df['pnl'] == 0]),
            'daily_win_rate': (len(self.daily_df[self.daily_df['pnl'] > 0]) / len(self.daily_df)) * 100,
            'avg_trades_per_day': self.daily_df['trades'].mean(),
            'most_active_day': self.daily_df['trades'].max(),
            'least_active_day': self.daily_df['trades'].min()
        }
        
        return daily_stats
    
    def _analyze_position_sizing(self) -> dict:
        """Analyze position sizing patterns"""
        contract_analysis = {}
        
        for contracts in sorted(self.trades_df['contracts'].unique()):
            subset = self.trades_df[self.trades_df['contracts'] == contracts]
            contract_analysis[f'{contracts}_contracts'] = {
                'count': len(subset),
                'total_pnl': subset['pnl'].sum(),
                'avg_pnl': subset['pnl'].mean(),
                'win_rate': (len(subset[subset['pnl'] > 0]) / len(subset)) * 100
            }
        
        return {
            'position_size_analysis': contract_analysis,
            'avg_position_size': self.trades_df['contracts'].mean(),
            'max_position_size': self.trades_df['contracts'].max(),
            'min_position_size': self.trades_df['contracts'].min(),
            'position_size_std': self.trades_df['contracts'].std()
        }
    
    def _calculate_statistics(self) -> dict:
        """Calculate comprehensive statistical summary"""
        return {
            'pnl_statistics': {
                'mean': self.trades_df['pnl'].mean(),
                'median': self.trades_df['pnl'].median(),
                'std': self.trades_df['pnl'].std(),
                'skewness': self.trades_df['pnl'].skew(),
                'kurtosis': self.trades_df['pnl'].kurtosis(),
                'min': self.trades_df['pnl'].min(),
                'max': self.trades_df['pnl'].max(),
                'q25': self.trades_df['pnl'].quantile(0.25),
                'q75': self.trades_df['pnl'].quantile(0.75)
            },
            'return_statistics': {
                'mean': self.trades_df['return_pct'].mean(),
                'median': self.trades_df['return_pct'].median(),
                'std': self.trades_df['return_pct'].std(),
                'min': self.trades_df['return_pct'].min(),
                'max': self.trades_df['return_pct'].max()
            },
            'confidence_statistics': {
                'mean': self.trades_df['confidence'].mean(),
                'median': self.trades_df['confidence'].median(),
                'std': self.trades_df['confidence'].std(),
                'min': self.trades_df['confidence'].min(),
                'max': self.trades_df['confidence'].max()
            }
        }
    
    def generate_report(self, analytics: dict):
        """Generate a comprehensive formatted report"""
        print("\n" + "="*80)
        print("🏆 HIGH FREQUENCY 0DTE STRATEGY - COMPREHENSIVE ANALYTICS")
        print("="*80)
        
        # Overview
        overview = analytics['overview']
        print(f"\n📊 STRATEGY OVERVIEW")
        print(f"Total Trades: {overview['total_trades']}")
        print(f"Trading Days: {overview['trading_days']}")
        print(f"Avg Trades/Day: {overview['avg_trades_per_day']:.1f}")
        print(f"Total P&L: ${overview['total_pnl']:+,.2f}")
        print(f"Avg P&L/Trade: ${overview['avg_pnl_per_trade']:+.2f}")
        print(f"Avg Daily P&L: ${overview['avg_daily_pnl']:+.2f}")
        print(f"Capital Efficiency: {overview['capital_efficiency']}")
        
        # P&L Analysis
        pnl = analytics['pnl_analysis']
        print(f"\n💰 PROFIT & LOSS ANALYSIS")
        print(f"Total P&L: ${pnl['total_pnl']:+,.2f}")
        print(f"Gross Profit: ${pnl['gross_profit']:+,.2f}")
        print(f"Gross Loss: ${pnl['gross_loss']:+,.2f}")
        print(f"Profit Factor: {pnl['profit_factor']:.2f}")
        print(f"Max Win: ${pnl['max_win']:+,.2f}")
        print(f"Max Loss: ${pnl['max_loss']:+,.2f}")
        print(f"Avg Win: ${pnl['avg_win']:+.2f}")
        print(f"Avg Loss: ${pnl['avg_loss']:+.2f}")
        
        # Win/Loss Analysis
        wl = analytics['win_loss_analysis']
        print(f"\n🎯 WIN/LOSS ANALYSIS")
        print(f"Total Trades: {wl['total_trades']}")
        print(f"Winning Trades: {wl['winning_trades']} ({wl['win_rate']:.1f}%)")
        print(f"Losing Trades: {wl['losing_trades']} ({wl['loss_rate']:.1f}%)")
        print(f"Breakeven Trades: {wl['breakeven_trades']} ({wl['breakeven_rate']:.1f}%)")
        print(f"Win/Loss Ratio: {wl['win_loss_ratio']:.2f}")
        print(f"Expectancy: ${wl['expectancy']:+.2f}")
        print(f"Max Win Streak: {wl['largest_win_streak']}")
        print(f"Max Loss Streak: {wl['largest_loss_streak']}")
        
        # Signal Analysis
        signals = analytics['signal_analysis']
        print(f"\n📈 SIGNAL TYPE ANALYSIS")
        call_perf = signals['call_performance']
        put_perf = signals['put_performance']
        print(f"CALL Performance:")
        print(f"  Trades: {call_perf['trades']}, P&L: ${call_perf['total_pnl']:+.2f}, Win Rate: {call_perf['win_rate']:.1f}%")
        print(f"PUT Performance:")
        print(f"  Trades: {put_perf['trades']}, P&L: ${put_perf['total_pnl']:+.2f}, Win Rate: {put_perf['win_rate']:.1f}%")
        print(f"Better Performer: {signals['better_performer']}")
        
        # Exit Reason Analysis
        exits = analytics['exit_analysis']
        print(f"\n🚪 EXIT REASON ANALYSIS")
        for reason, data in exits.items():
            print(f"{reason}: {data['count']} trades ({data['percentage']:.1f}%), "
                  f"P&L: ${data['total_pnl']:+.2f}, Win Rate: {data['win_rate']:.1f}%")
        
        # Risk Metrics
        risk = analytics['risk_metrics']
        print(f"\n⚠️ RISK METRICS")
        print(f"Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: ${risk['max_drawdown']:+,.2f}")
        print(f"Calmar Ratio: {risk['calmar_ratio']:.2f}")
        print(f"Volatility: ${risk['volatility']:.2f}")
        print(f"Value at Risk (95%): ${risk['value_at_risk_95']:+.2f}")
        print(f"Expected Shortfall (95%): ${risk['expected_shortfall_95']:+.2f}")
        
        # Daily Performance
        daily = analytics['daily_performance']
        print(f"\n📅 DAILY PERFORMANCE")
        print(f"Best Day: ${daily['best_day']:+.2f}")
        print(f"Worst Day: ${daily['worst_day']:+.2f}")
        print(f"Avg Daily P&L: ${daily['avg_daily_pnl']:+.2f}")
        print(f"Profitable Days: {daily['profitable_days']}/{len(self.daily_df)} ({daily['daily_win_rate']:.1f}%)")
        print(f"Avg Trades/Day: {daily['avg_trades_per_day']:.1f}")
        print(f"Most Active Day: {daily['most_active_day']} trades")
        
        # Position Sizing
        position = analytics['position_sizing']
        print(f"\n📏 POSITION SIZING")
        print(f"Avg Position Size: {position['avg_position_size']:.1f} contracts")
        print(f"Position Range: {position['min_position_size']}-{position['max_position_size']} contracts")
        print(f"Position Std Dev: {position['position_size_std']:.2f}")
        
        # Key Takeaways
        print(f"\n🔑 KEY TAKEAWAYS")
        if wl['win_rate'] > 50:
            print(f"✅ Strong win rate of {wl['win_rate']:.1f}%")
        else:
            print(f"⚠️ Win rate of {wl['win_rate']:.1f}% needs improvement")
        
        if pnl['profit_factor'] > 1.0:
            print(f"✅ Positive profit factor of {pnl['profit_factor']:.2f}")
        else:
            print(f"❌ Negative profit factor of {pnl['profit_factor']:.2f}")
        
        if overview['avg_trades_per_day'] >= 1.0:
            print(f"✅ High frequency achieved: {overview['avg_trades_per_day']:.1f} trades/day")
        else:
            print(f"⚠️ Low frequency: {overview['avg_trades_per_day']:.1f} trades/day")
        
        print("="*80)
    
    def save_detailed_csv(self, filename: str = None):
        """Save detailed trade data to CSV"""
        if filename is None:
            filename = f"detailed_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Add calculated fields
        self.trades_df['cumulative_pnl'] = self.trades_df['pnl'].cumsum()
        self.trades_df['win'] = self.trades_df['pnl'] > 0
        self.trades_df['trade_number'] = range(1, len(self.trades_df) + 1)
        
        self.trades_df.to_csv(filename, index=False)
        print(f"💾 Detailed trade data saved to {filename}")
        
        # Save daily summary
        daily_filename = filename.replace('detailed_trades', 'daily_summary')
        self.daily_df.to_csv(daily_filename, index=False)
        print(f"💾 Daily summary saved to {daily_filename}")

def run_analytics_on_demo():
    """Run analytics on the demo results"""
    # Import and run the demo to get fresh results
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    from simple_hf_demo import HighFrequencyDemo
    
    print("🚀 Running High Frequency Demo for Analytics...")
    
    demo = HighFrequencyDemo()
    
    # Run 5 trading days
    dates = ['2024-07-15', '2024-07-16', '2024-07-17', '2024-07-18', '2024-07-19']
    daily_results = []
    
    for date in dates:
        import random
        spy_price = random.uniform(580, 600)
        day_result = demo.run_demo_day(date, spy_price)
        daily_results.append(day_result)
    
    # Analyze results
    analytics = TradingAnalytics()
    results = analytics.analyze_demo_results(daily_results)
    
    # Generate comprehensive report
    analytics.generate_report(results)
    
    # Save detailed data
    analytics.save_detailed_csv()
    
    return results

if __name__ == "__main__":
    # Run the analytics
    results = run_analytics_on_demo() 