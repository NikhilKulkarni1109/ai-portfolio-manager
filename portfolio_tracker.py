"""
Portfolio performance tracking and S&P 500 comparison
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os

from config import config
from alpaca_client import AlpacaClient
from data_sources import DataSources

logger = logging.getLogger(__name__)

class PortfolioTracker:
    """Portfolio performance tracking and analysis"""
    
    def __init__(self, data_file: str = 'data/portfolio_performance.json'):
        """Initialize portfolio tracker"""
        try:
            self.alpaca = AlpacaClient()
            self.data_sources = DataSources()
            self.data_file = data_file
            
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            
            # Load historical performance data
            self.performance_data = self._load_performance_data()
            
            logger.info("Portfolio tracker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize portfolio tracker: {e}")
            raise
    
    def update_performance(self) -> Dict:
        """Update and record current portfolio performance"""
        try:
            timestamp = datetime.now()
            
            # Get current portfolio state
            account_info = self.alpaca.get_account_info()
            positions = self.alpaca.get_positions()
            portfolio_history = self.alpaca.get_portfolio_history(period='1M')
            
            # Get SPY benchmark data
            spy_data = self.data_sources.get_spy_data()
            
            # Calculate current metrics
            current_metrics = self._calculate_current_metrics(
                account_info, positions, portfolio_history, spy_data
            )
            
            # Calculate historical performance
            historical_metrics = self._calculate_historical_metrics(portfolio_history, spy_data)
            
            # Combine all metrics
            performance_record = {
                'timestamp': timestamp.isoformat(),
                'current': current_metrics,
                'historical': historical_metrics,
                'positions': positions,
                'spy_benchmark': {
                    'price': spy_data['current_price'],
                    'day_change_pct': spy_data['day_change_pct'],
                    'volatility': spy_data['volatility']
                }
            }
            
            # Add to performance history
            self.performance_data.append(performance_record)
            
            # Save to file
            self._save_performance_data()
            
            logger.info(f"Performance updated: Portfolio=${current_metrics['portfolio_value']:,.2f}, "
                       f"SPY Outperformance={historical_metrics.get('spy_outperformance', 0):.2f}%")
            
            return performance_record
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
            return {}
    
    def generate_performance_report(self, days: int = 30) -> Dict:
        """Generate comprehensive performance report"""
        try:
            if not self.performance_data:
                return self._get_default_report()
            
            # Filter data for the specified period
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_data = [
                record for record in self.performance_data
                if datetime.fromisoformat(record['timestamp']) >= cutoff_date
            ]
            
            if not recent_data:
                return self._get_default_report()
            
            # Extract time series data
            timestamps = [datetime.fromisoformat(r['timestamp']) for r in recent_data]
            portfolio_values = [r['current']['portfolio_value'] for r in recent_data]
            spy_prices = [r['spy_benchmark']['price'] for r in recent_data]
            
            # Calculate returns
            portfolio_returns = self._calculate_returns(portfolio_values)
            spy_returns = self._calculate_returns(spy_prices)
            
            # Generate comprehensive analysis
            report = {
                'period_days': days,
                'report_date': datetime.now().isoformat(),
                'portfolio_summary': self._analyze_portfolio_performance(
                    portfolio_values, portfolio_returns, recent_data
                ),
                'benchmark_comparison': self._analyze_benchmark_comparison(
                    portfolio_returns, spy_returns, portfolio_values, spy_prices
                ),
                'risk_metrics': self._calculate_risk_metrics(
                    portfolio_returns, spy_returns, recent_data
                ),
                'trading_analysis': self._analyze_trading_activity(recent_data),
                'position_analysis': self._analyze_positions(recent_data[-1] if recent_data else {}),
                'recommendations': self._generate_recommendations(recent_data),
                'charts_data': self._prepare_chart_data(timestamps, portfolio_values, spy_prices)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return self._get_default_report()
    
    def create_performance_charts(self, output_dir: str = 'charts') -> List[str]:
        """Create performance visualization charts"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            chart_files = []
            
            if not self.performance_data:
                logger.warning("No performance data available for charts")
                return []
            
            # Extract data for charts
            timestamps = [datetime.fromisoformat(r['timestamp']) for r in self.performance_data]
            portfolio_values = [r['current']['portfolio_value'] for r in self.performance_data]
            spy_prices = [r['spy_benchmark']['price'] for r in self.performance_data]
            
            # Normalize values to show relative performance
            portfolio_normalized = (np.array(portfolio_values) / portfolio_values[0] - 1) * 100
            spy_normalized = (np.array(spy_prices) / spy_prices[0] - 1) * 100
            
            # Chart 1: Portfolio vs SPY Performance
            plt.figure(figsize=(12, 8))
            plt.plot(timestamps, portfolio_normalized, label='AI Portfolio', linewidth=2, color='blue')
            plt.plot(timestamps, spy_normalized, label='S&P 500 (SPY)', linewidth=2, color='red')
            plt.title('AI Portfolio vs S&P 500 Performance', fontsize=16, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Return (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            chart1_path = os.path.join(output_dir, 'portfolio_vs_spy.png')
            plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_files.append(chart1_path)
            
            # Chart 2: Portfolio Value Over Time
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, portfolio_values, linewidth=2, color='green')
            plt.title('Portfolio Value Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            chart2_path = os.path.join(output_dir, 'portfolio_value.png')
            plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_files.append(chart2_path)
            
            # Chart 3: Daily Returns Distribution
            if len(portfolio_values) > 1:
                daily_returns = self._calculate_returns(portfolio_values)
                
                plt.figure(figsize=(10, 6))
                plt.hist(daily_returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
                plt.axvline(np.mean(daily_returns), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(daily_returns):.3f}%')
                plt.title('Daily Returns Distribution', fontsize=16, fontweight='bold')
                plt.xlabel('Daily Return (%)')
                plt.ylabel('Frequency')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                chart3_path = os.path.join(output_dir, 'returns_distribution.png')
                plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files.append(chart3_path)
            
            # Chart 4: Position Allocation (if we have recent data)
            if self.performance_data:
                recent_positions = self.performance_data[-1].get('positions', [])
                if recent_positions:
                    symbols = [pos['symbol'] for pos in recent_positions]
                    values = [abs(pos['market_value']) for pos in recent_positions]
                    
                    plt.figure(figsize=(10, 8))
                    plt.pie(values, labels=symbols, autopct='%1.1f%%', startangle=90)
                    plt.title('Current Position Allocation', fontsize=16, fontweight='bold')
                    plt.axis('equal')
                    
                    chart4_path = os.path.join(output_dir, 'position_allocation.png')
                    plt.savefig(chart4_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_files.append(chart4_path)
            
            logger.info(f"Created {len(chart_files)} performance charts in {output_dir}")
            return chart_files
            
        except Exception as e:
            logger.error(f"Error creating performance charts: {e}")
            return []
    
    def get_performance_summary(self) -> Dict:
        """Get quick performance summary"""
        try:
            if not self.performance_data:
                return {'status': 'no_data'}
            
            latest = self.performance_data[-1]
            
            # Calculate period performance (30 days)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            baseline_record = None
            
            for record in self.performance_data:
                if datetime.fromisoformat(record['timestamp']) >= thirty_days_ago:
                    baseline_record = record
                    break
            
            if baseline_record:
                baseline_value = baseline_record['current']['portfolio_value']
                current_value = latest['current']['portfolio_value']
                period_return = ((current_value - baseline_value) / baseline_value) * 100
            else:
                period_return = 0
            
            return {
                'current_value': latest['current']['portfolio_value'],
                'initial_value': config.INITIAL_CAPITAL,
                'total_return': latest['current'].get('total_return_pct', 0),
                'period_return_30d': period_return,
                'spy_outperformance': latest['historical'].get('spy_outperformance', 0),
                'position_count': len(latest.get('positions', [])),
                'cash_percentage': latest['current'].get('cash_percentage', 100),
                'last_updated': latest['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'status': 'error'}
    
    def _calculate_current_metrics(self, account_info: Dict, positions: List[Dict], 
                                 portfolio_history: pd.DataFrame, spy_data: Dict) -> Dict:
        """Calculate current portfolio metrics"""
        try:
            portfolio_value = account_info.get('portfolio_value', 0)
            cash = account_info.get('cash', 0)
            equity = account_info.get('equity', 0)
            
            # Calculate allocation
            cash_pct = (cash / portfolio_value * 100) if portfolio_value > 0 else 100
            equity_pct = (equity / portfolio_value * 100) if portfolio_value > 0 else 0
            
            # Calculate total return
            initial_value = config.INITIAL_CAPITAL
            total_return = ((portfolio_value - initial_value) / initial_value * 100) if initial_value > 0 else 0
            
            # Calculate unrealized P&L
            total_unrealized_pl = sum(pos.get('unrealized_pl', 0) for pos in positions)
            
            return {
                'portfolio_value': portfolio_value,
                'cash': cash,
                'equity': equity,
                'cash_percentage': cash_pct,
                'equity_percentage': equity_pct,
                'total_return_pct': total_return,
                'unrealized_pl': total_unrealized_pl,
                'position_count': len(positions),
                'buying_power': account_info.get('buying_power', 0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating current metrics: {e}")
            return {}
    
    def _calculate_historical_metrics(self, portfolio_history: pd.DataFrame, spy_data: Dict) -> Dict:
        """Calculate historical performance metrics"""
        try:
            if portfolio_history.empty:
                return {'status': 'no_history'}
            
            # Get portfolio returns
            if 'profit_loss_pct' in portfolio_history.columns:
                portfolio_returns = portfolio_history['profit_loss_pct'].dropna()
            else:
                portfolio_returns = pd.Series([])
            
            if portfolio_returns.empty:
                return {'status': 'no_returns_data'}
            
            # Calculate basic statistics
            total_return = portfolio_returns.iloc[-1] if not portfolio_returns.empty else 0
            
            # Calculate volatility (annualized)
            daily_returns = portfolio_returns.pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) if not daily_returns.empty else 0
            
            # Calculate Sharpe ratio (assuming 0% risk-free rate)
            avg_return = daily_returns.mean() * 252 if not daily_returns.empty else 0
            sharpe_ratio = (avg_return / volatility) if volatility > 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + daily_returns).cumprod() if not daily_returns.empty else pd.Series([1])
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0
            
            # Compare to SPY
            spy_return = spy_data.get('day_change_pct', 0)  # This is just daily, would need historical for proper comparison
            spy_outperformance = total_return - spy_return  # Simplified
            
            return {
                'total_return_pct': total_return,
                'volatility': volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'spy_outperformance': spy_outperformance,
                'win_rate': self._calculate_win_rate(daily_returns),
                'avg_daily_return': avg_return,
                'best_day': daily_returns.max() * 100 if not daily_returns.empty else 0,
                'worst_day': daily_returns.min() * 100 if not daily_returns.empty else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating historical metrics: {e}")
            return {}
    
    def _calculate_returns(self, values: List[float]) -> List[float]:
        """Calculate percentage returns from values"""
        if len(values) < 2:
            return []
        
        returns = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                ret = ((values[i] - values[i-1]) / values[i-1]) * 100
                returns.append(ret)
        
        return returns
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate (percentage of positive days)"""
        try:
            if returns.empty:
                return 0.0
            
            positive_days = (returns > 0).sum()
            total_days = len(returns)
            
            return (positive_days / total_days * 100) if total_days > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def _analyze_portfolio_performance(self, values: List[float], 
                                     returns: List[float], recent_data: List[Dict]) -> Dict:
        """Analyze overall portfolio performance"""
        try:
            if not values or len(values) < 2:
                return {'status': 'insufficient_data'}
            
            current_value = values[-1]
            initial_value = values[0]
            total_return = ((current_value - initial_value) / initial_value * 100)
            
            avg_return = np.mean(returns) if returns else 0
            volatility = np.std(returns) if returns else 0
            
            return {
                'current_value': current_value,
                'initial_value': initial_value,
                'total_return_pct': total_return,
                'avg_daily_return': avg_return,
                'volatility': volatility,
                'best_day': max(returns) if returns else 0,
                'worst_day': min(returns) if returns else 0,
                'positive_days': len([r for r in returns if r > 0]),
                'negative_days': len([r for r in returns if r < 0]),
                'win_rate': len([r for r in returns if r > 0]) / len(returns) * 100 if returns else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio performance: {e}")
            return {}
    
    def _analyze_benchmark_comparison(self, portfolio_returns: List[float], 
                                    spy_returns: List[float],
                                    portfolio_values: List[float], 
                                    spy_prices: List[float]) -> Dict:
        """Analyze performance vs benchmark"""
        try:
            if not portfolio_returns or not spy_returns:
                return {'status': 'insufficient_data'}
            
            # Calculate total returns
            portfolio_total = ((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100)
            spy_total = ((spy_prices[-1] - spy_prices[0]) / spy_prices[0] * 100)
            
            outperformance = portfolio_total - spy_total
            
            # Calculate correlation
            min_length = min(len(portfolio_returns), len(spy_returns))
            if min_length > 1:
                correlation = np.corrcoef(
                    portfolio_returns[:min_length], 
                    spy_returns[:min_length]
                )[0, 1]
            else:
                correlation = 0
            
            # Calculate beta
            if len(portfolio_returns) > 1 and len(spy_returns) > 1:
                covariance = np.cov(portfolio_returns[:min_length], spy_returns[:min_length])[0, 1]
                spy_variance = np.var(spy_returns[:min_length])
                beta = covariance / spy_variance if spy_variance > 0 else 1
            else:
                beta = 1
            
            return {
                'portfolio_return': portfolio_total,
                'spy_return': spy_total,
                'outperformance': outperformance,
                'correlation': correlation,
                'beta': beta,
                'is_outperforming': outperformance > 0,
                'tracking_error': np.std([p - s for p, s in zip(portfolio_returns[:min_length], spy_returns[:min_length])]) if min_length > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing benchmark comparison: {e}")
            return {}
    
    def _calculate_risk_metrics(self, portfolio_returns: List[float], 
                              spy_returns: List[float], recent_data: List[Dict]) -> Dict:
        """Calculate risk metrics"""
        try:
            if not portfolio_returns:
                return {'status': 'insufficient_data'}
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5) if portfolio_returns else 0
            
            # Expected Shortfall (Conditional VaR)
            returns_below_var = [r for r in portfolio_returns if r <= var_95]
            expected_shortfall = np.mean(returns_below_var) if returns_below_var else 0
            
            # Maximum consecutive losses
            consecutive_losses = 0
            max_consecutive_losses = 0
            for ret in portfolio_returns:
                if ret < 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            return {
                'value_at_risk_95': var_95,
                'expected_shortfall': expected_shortfall,
                'max_consecutive_losses': max_consecutive_losses,
                'volatility': np.std(portfolio_returns) if portfolio_returns else 0,
                'downside_volatility': np.std([r for r in portfolio_returns if r < 0]) if portfolio_returns else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _analyze_trading_activity(self, recent_data: List[Dict]) -> Dict:
        """Analyze trading activity"""
        try:
            # This would analyze actual trades from the trade journal
            # For now, we'll return basic position change analysis
            
            position_changes = 0
            symbols_traded = set()
            
            if len(recent_data) > 1:
                for i in range(1, len(recent_data)):
                    prev_positions = {pos['symbol']: pos['qty'] for pos in recent_data[i-1].get('positions', [])}
                    curr_positions = {pos['symbol']: pos['qty'] for pos in recent_data[i].get('positions', [])}
                    
                    for symbol, qty in curr_positions.items():
                        if symbol not in prev_positions or prev_positions[symbol] != qty:
                            position_changes += 1
                            symbols_traded.add(symbol)
            
            return {
                'position_changes': position_changes,
                'unique_symbols_traded': len(symbols_traded),
                'symbols_traded': list(symbols_traded),
                'avg_positions_held': np.mean([len(record.get('positions', [])) for record in recent_data]) if recent_data else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trading activity: {e}")
            return {}
    
    def _analyze_positions(self, latest_data: Dict) -> Dict:
        """Analyze current positions"""
        try:
            positions = latest_data.get('positions', [])
            
            if not positions:
                return {'status': 'no_positions'}
            
            total_value = sum(abs(pos['market_value']) for pos in positions)
            
            # Position analysis
            winning_positions = [pos for pos in positions if pos.get('unrealized_pl', 0) > 0]
            losing_positions = [pos for pos in positions if pos.get('unrealized_pl', 0) < 0]
            
            # Sector/concentration analysis
            sectors = {}
            for pos in positions:
                # This would require additional data to map symbols to sectors
                # For now, we'll just use the symbol
                sector = pos.get('symbol', 'Unknown')[:2]  # Simplified
                sectors[sector] = sectors.get(sector, 0) + abs(pos['market_value'])
            
            return {
                'total_positions': len(positions),
                'total_market_value': total_value,
                'winning_positions': len(winning_positions),
                'losing_positions': len(losing_positions),
                'largest_position': max(positions, key=lambda x: abs(x['market_value']))['symbol'] if positions else None,
                'largest_position_pct': max(abs(pos['market_value']) for pos in positions) / total_value * 100 if total_value > 0 else 0,
                'total_unrealized_pl': sum(pos.get('unrealized_pl', 0) for pos in positions),
                'sector_allocation': sectors
            }
            
        except Exception as e:
            logger.error(f"Error analyzing positions: {e}")
            return {}
    
    def _generate_recommendations(self, recent_data: List[Dict]) -> List[str]:
        """Generate performance-based recommendations"""
        recommendations = []
        
        try:
            if not recent_data:
                return ["Insufficient data for recommendations"]
            
            latest = recent_data[-1]
            portfolio_value = latest['current']['portfolio_value']
            positions = latest.get('positions', [])
            
            # Cash allocation recommendation
            cash_pct = latest['current'].get('cash_percentage', 0)
            if cash_pct > 80:
                recommendations.append("Consider increasing equity allocation - currently holding too much cash")
            elif cash_pct < 10:
                recommendations.append("Consider maintaining higher cash reserves for opportunities")
            
            # Position concentration
            if positions:
                total_value = sum(abs(pos['market_value']) for pos in positions)
                max_position_pct = max(abs(pos['market_value']) for pos in positions) / total_value * 100 if total_value > 0 else 0
                
                if max_position_pct > 20:
                    recommendations.append("Consider reducing concentration in largest position (risk management)")
            
            # Performance-based recommendations
            if len(recent_data) > 7:  # Need at least a week of data
                recent_returns = []
                for i in range(1, min(8, len(recent_data))):
                    prev_val = recent_data[i-1]['current']['portfolio_value']
                    curr_val = recent_data[i]['current']['portfolio_value']
                    if prev_val > 0:
                        ret = (curr_val - prev_val) / prev_val * 100
                        recent_returns.append(ret)
                
                if recent_returns:
                    avg_return = np.mean(recent_returns)
                    volatility = np.std(recent_returns)
                    
                    if volatility > 3:  # High daily volatility
                        recommendations.append("Recent performance shows high volatility - consider risk management review")
                    
                    if avg_return < -1:  # Negative average return
                        recommendations.append("Recent performance underperforming - review strategy and market conditions")
            
            # SPY comparison
            spy_outperformance = latest['historical'].get('spy_outperformance', 0)
            if spy_outperformance < -5:
                recommendations.append("Underperforming S&P 500 significantly - consider strategy adjustment")
            elif spy_outperformance > 10:
                recommendations.append("Strong outperformance vs S&P 500 - maintain current approach")
            
            if not recommendations:
                recommendations.append("Portfolio performance is within normal parameters")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to data error"]
    
    def _prepare_chart_data(self, timestamps: List[datetime], 
                          portfolio_values: List[float], spy_prices: List[float]) -> Dict:
        """Prepare data for chart generation"""
        try:
            # Convert to relative performance
            portfolio_returns = []
            spy_returns = []
            
            if portfolio_values and spy_prices:
                portfolio_base = portfolio_values[0]
                spy_base = spy_prices[0]
                
                portfolio_returns = [(val / portfolio_base - 1) * 100 for val in portfolio_values]
                spy_returns = [(price / spy_base - 1) * 100 for price in spy_prices]
            
            return {
                'timestamps': [ts.isoformat() for ts in timestamps],
                'portfolio_values': portfolio_values,
                'portfolio_returns': portfolio_returns,
                'spy_prices': spy_prices,
                'spy_returns': spy_returns
            }
            
        except Exception as e:
            logger.error(f"Error preparing chart data: {e}")
            return {}
    
    def _load_performance_data(self) -> List[Dict]:
        """Load performance data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data)} performance records")
                return data
            else:
                logger.info("No existing performance data file found")
                return []
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            return []
    
    def _save_performance_data(self):
        """Save performance data to file"""
        try:
            # Keep only last 1000 records to prevent file from growing too large
            if len(self.performance_data) > 1000:
                self.performance_data = self.performance_data[-1000:]
            
            with open(self.data_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
            
            logger.debug(f"Saved {len(self.performance_data)} performance records")
            
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def _get_default_report(self) -> Dict:
        """Get default report when no data is available"""
        return {
            'period_days': 0,
            'report_date': datetime.now().isoformat(),
            'portfolio_summary': {'status': 'no_data'},
            'benchmark_comparison': {'status': 'no_data'},
            'risk_metrics': {'status': 'no_data'},
            'trading_analysis': {'status': 'no_data'},
            'position_analysis': {'status': 'no_data'},
            'recommendations': ['Insufficient data for analysis'],
            'charts_data': {}
        } 