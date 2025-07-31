"""
Trading strategy and decision logic implementation
"""
import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from config import config
from alpaca_client import AlpacaClient
from llm_analyzer import GeminiAnalyzer
from data_sources import DataSources

logger = logging.getLogger(__name__)

class TradingStrategy:
    """AI-powered trading strategy implementation"""
    
    def __init__(self):
        """Initialize trading strategy components"""
        try:
            self.alpaca = AlpacaClient()
            self.llm_analyzer = GeminiAnalyzer()
            self.data_sources = DataSources()
            
            # Strategy parameters
            self.max_positions = config.MAX_POSITIONS
            self.max_position_size = config.MAX_POSITION_SIZE
            self.stop_loss_pct = config.STOP_LOSS_PCT
            self.take_profit_pct = config.TAKE_PROFIT_PCT
            
            # Trade journal
            self.trade_journal = []
            
            logger.info("Trading strategy initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading strategy: {e}")
            raise
    
    def execute_trading_session(self) -> Dict:
        """Execute a complete trading session"""
        try:
            logger.info("=== Starting Trading Session ===")
            session_start = datetime.now()
            
            # Check if market is open
            if not self.alpaca.is_market_open():
                logger.info("Market is closed, skipping trading session")
                return {
                    'status': 'skipped',
                    'reason': 'market_closed',
                    'timestamp': session_start.isoformat()
                }
            
            # Step 1: Gather market data and news
            logger.info("Step 1: Gathering market data and news")
            market_data = self._gather_market_intelligence()
            
            # Step 2: Analyze market sentiment
            logger.info("Step 2: Analyzing market sentiment with LLM")
            sentiment_analysis = self.llm_analyzer.analyze_market_sentiment(
                market_data['news'], 
                market_data['market_overview']
            )
            
            # Step 3: Get current portfolio state
            logger.info("Step 3: Getting current portfolio state")
            portfolio_data = self._get_portfolio_state()
            
            # Step 4: Generate trading decisions
            logger.info("Step 4: Generating trading decisions")
            trading_decisions = self.llm_analyzer.generate_trading_decisions(
                sentiment_analysis,
                portfolio_data,
                config.STOCK_UNIVERSE
            )
            
            # Step 5: Risk assessment and position sizing
            logger.info("Step 5: Performing risk assessment")
            validated_trades = self._validate_and_size_trades(
                trading_decisions, 
                portfolio_data, 
                market_data
            )
            
            # Step 6: Execute trades
            logger.info("Step 6: Executing validated trades")
            execution_results = self._execute_trades(validated_trades)
            
            # Step 7: Update stop losses and take profits
            logger.info("Step 7: Updating risk management orders")
            self._update_risk_management()
            
            # Step 8: Log session results
            session_results = {
                'status': 'completed',
                'timestamp': session_start.isoformat(),
                'duration_seconds': (datetime.now() - session_start).total_seconds(),
                'sentiment_analysis': sentiment_analysis,
                'trading_decisions': trading_decisions,
                'executed_trades': execution_results,
                'portfolio_state': self._get_portfolio_state(),
                'market_data_summary': self._summarize_market_data(market_data)
            }
            
            self._log_trading_session(session_results)
            
            logger.info(f"=== Trading Session Completed in {session_results['duration_seconds']:.2f}s ===")
            
            return session_results
            
        except Exception as e:
            logger.error(f"Error during trading session: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _gather_market_intelligence(self) -> Dict:
        """Gather comprehensive market data and news"""
        try:
            # Get general market news
            market_news = self.data_sources.get_market_news(
                query="stock market OR economy OR earnings OR Federal Reserve",
                days_back=1
            )
            
            # Get stock-specific news
            stock_news = self.data_sources.get_stock_news(
                config.STOCK_UNIVERSE[:10],  # Limit to top 10 stocks
                days_back=2
            )
            
            # Get market data for stock universe
            market_data = self.data_sources.get_market_data(
                config.STOCK_UNIVERSE,
                period='5d'
            )
            
            # Get benchmark data
            spy_data = self.data_sources.get_spy_data()
            vix_data = self.data_sources.get_vix_data()
            
            # Get economic calendar
            economic_events = self.data_sources.get_economic_calendar()
            
            return {
                'news': market_news,
                'stock_news': stock_news,
                'market_data': market_data,
                'market_overview': {
                    'spy_close': spy_data['current_price'],
                    'spy_change_pct': spy_data['day_change_pct'],
                    'vix_level': vix_data['level'],
                    'vix_change_pct': vix_data['change_pct'],
                    'volume_trend': self._assess_volume_trend(market_data)
                },
                'spy_data': spy_data,
                'vix_data': vix_data,
                'economic_events': economic_events
            }
            
        except Exception as e:
            logger.error(f"Error gathering market intelligence: {e}")
            return self._get_default_market_intelligence()
    
    def _get_portfolio_state(self) -> Dict:
        """Get current portfolio state"""
        try:
            account_info = self.alpaca.get_account_info()
            positions = self.alpaca.get_positions()
            portfolio_history = self.alpaca.get_portfolio_history()
            
            # Calculate portfolio metrics
            total_value = account_info.get('portfolio_value', 0)
            cash = account_info.get('cash', 0)
            equity = account_info.get('equity', 0)
            
            # Calculate current allocation
            cash_percentage = (cash / total_value * 100) if total_value > 0 else 100
            equity_percentage = (equity / total_value * 100) if total_value > 0 else 0
            
            # Calculate performance vs SPY
            spy_performance = self._calculate_spy_comparison(portfolio_history)
            
            return {
                'account_info': account_info,
                'positions': positions,
                'portfolio_value': total_value,
                'cash': cash,
                'equity': equity,
                'cash_percentage': cash_percentage,
                'equity_percentage': equity_percentage,
                'position_count': len(positions),
                'available_buying_power': account_info.get('buying_power', 0),
                'spy_comparison': spy_performance,
                'portfolio_history': portfolio_history
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio state: {e}")
            return self._get_default_portfolio_state()
    
    def _validate_and_size_trades(self, trading_decisions: Dict, 
                                 portfolio_data: Dict, 
                                 market_data: Dict) -> List[Dict]:
        """Validate trading decisions and calculate position sizes"""
        validated_trades = []
        
        try:
            proposed_trades = trading_decisions.get('trades', [])
            portfolio_value = portfolio_data.get('portfolio_value', 100000)
            current_positions = portfolio_data.get('positions', [])
            available_cash = portfolio_data.get('cash', 0)
            
            # Create position lookup
            current_symbols = {pos['symbol']: pos for pos in current_positions}
            
            for trade in proposed_trades:
                symbol = trade.get('symbol')
                action = trade.get('action')
                confidence = trade.get('confidence', 0.5)
                
                if not symbol or not action:
                    logger.warning(f"Invalid trade data: {trade}")
                    continue
                
                # Validate symbol is in our universe
                if symbol not in config.STOCK_UNIVERSE:
                    logger.warning(f"Symbol {symbol} not in trading universe")
                    continue
                
                # Get market data for the symbol
                symbol_data = market_data['market_data'].get(symbol)
                if not symbol_data:
                    logger.warning(f"No market data available for {symbol}")
                    continue
                
                current_price = symbol_data['price_data']['current_price']
                if current_price <= 0:
                    logger.warning(f"Invalid price for {symbol}: {current_price}")
                    continue
                
                # Calculate position size based on confidence and risk
                position_size_usd = self._calculate_position_size(
                    confidence, 
                    portfolio_value, 
                    current_price,
                    trade.get('risk_level', 'medium')
                )
                
                # Calculate number of shares
                shares = int(position_size_usd / current_price)
                
                if action == 'buy':
                    # Check if we can afford the position
                    required_cash = shares * current_price
                    if required_cash > available_cash:
                        # Scale down position size
                        shares = int(available_cash * 0.95 / current_price)  # 95% to leave some buffer
                        if shares < 1:
                            logger.warning(f"Not enough cash to buy {symbol}")
                            continue
                    
                    # Check position limits
                    if len(current_positions) >= self.max_positions and symbol not in current_symbols:
                        logger.warning(f"Already at max positions ({self.max_positions})")
                        continue
                
                elif action == 'sell':
                    # Check if we have the position
                    if symbol not in current_symbols:
                        logger.warning(f"Cannot sell {symbol} - no current position")
                        continue
                    
                    current_qty = abs(current_symbols[symbol]['qty'])
                    shares = min(shares, current_qty)  # Don't sell more than we have
                
                if shares > 0:
                    validated_trade = {
                        'symbol': symbol,
                        'action': action,
                        'shares': shares,
                        'estimated_price': current_price,
                        'estimated_value': shares * current_price,
                        'confidence': confidence,
                        'reasoning': trade.get('reasoning', ''),
                        'risk_level': trade.get('risk_level', 'medium'),
                        'stop_loss': self._calculate_stop_loss(current_price, action),
                        'take_profit': self._calculate_take_profit(current_price, action),
                        'validation_timestamp': datetime.now().isoformat()
                    }
                    
                    validated_trades.append(validated_trade)
                    
                    # Update available cash for next calculation
                    if action == 'buy':
                        available_cash -= shares * current_price
                    else:
                        available_cash += shares * current_price
            
            logger.info(f"Validated {len(validated_trades)} out of {len(proposed_trades)} proposed trades")
            
            return validated_trades
            
        except Exception as e:
            logger.error(f"Error validating trades: {e}")
            return []
    
    def _calculate_position_size(self, confidence: float, portfolio_value: float, 
                               price: float, risk_level: str) -> float:
        """Calculate position size based on confidence and risk"""
        try:
            # Base position size as percentage of portfolio
            base_size_pct = self.max_position_size
            
            # Adjust based on confidence (0.5 to 1.5 multiplier)
            confidence_multiplier = 0.5 + confidence
            
            # Adjust based on risk level
            risk_multipliers = {
                'low': 1.2,
                'medium': 1.0,
                'high': 0.6
            }
            risk_multiplier = risk_multipliers.get(risk_level, 1.0)
            
            # Calculate final position size
            final_size_pct = base_size_pct * confidence_multiplier * risk_multiplier
            
            # Cap at max position size
            final_size_pct = min(final_size_pct, self.max_position_size)
            
            position_size_usd = portfolio_value * final_size_pct
            
            logger.debug(f"Position size calculation: {final_size_pct:.2%} of portfolio = ${position_size_usd:.2f}")
            
            return position_size_usd
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return portfolio_value * 0.05  # Default to 5%
    
    def _calculate_stop_loss(self, price: float, action: str) -> float:
        """Calculate stop loss price"""
        if action == 'buy':
            return price * (1 - self.stop_loss_pct)
        else:  # sell
            return price * (1 + self.stop_loss_pct)
    
    def _calculate_take_profit(self, price: float, action: str) -> float:
        """Calculate take profit price"""
        if action == 'buy':
            return price * (1 + self.take_profit_pct)
        else:  # sell
            return price * (1 - self.take_profit_pct)
    
    def _execute_trades(self, validated_trades: List[Dict]) -> List[Dict]:
        """Execute validated trades"""
        execution_results = []
        
        try:
            for trade in validated_trades:
                try:
                    logger.info(f"Executing: {trade['action']} {trade['shares']} {trade['symbol']}")
                    
                    # Place the order
                    order_result = self.alpaca.place_order(
                        symbol=trade['symbol'],
                        qty=trade['shares'],
                        side=trade['action'],
                        order_type='market'
                    )
                    
                    if order_result:
                        execution_result = {
                            'trade': trade,
                            'order': order_result,
                            'status': 'submitted',
                            'execution_timestamp': datetime.now().isoformat()
                        }
                        
                        # Log the trade in journal
                        journal_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'symbol': trade['symbol'],
                            'action': trade['action'],
                            'shares': trade['shares'],
                            'estimated_price': trade['estimated_price'],
                            'order_id': order_result.get('id'),
                            'reasoning': trade['reasoning'],
                            'confidence': trade['confidence'],
                            'risk_level': trade['risk_level']
                        }
                        self.trade_journal.append(journal_entry)
                        
                    else:
                        execution_result = {
                            'trade': trade,
                            'status': 'failed',
                            'error': 'Order placement failed',
                            'execution_timestamp': datetime.now().isoformat()
                        }
                    
                    execution_results.append(execution_result)
                    
                    # Small delay between orders
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error executing trade {trade}: {e}")
                    execution_results.append({
                        'trade': trade,
                        'status': 'error',
                        'error': str(e),
                        'execution_timestamp': datetime.now().isoformat()
                    })
            
            successful_trades = [r for r in execution_results if r['status'] == 'submitted']
            logger.info(f"Successfully executed {len(successful_trades)} out of {len(validated_trades)} trades")
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
            return []
    
    def _update_risk_management(self):
        """Update stop loss and take profit orders for current positions"""
        try:
            positions = self.alpaca.get_positions()
            
            for position in positions:
                symbol = position['symbol']
                qty = position['qty']
                current_price = position['current_price']
                side = position['side']
                
                # For now, we'll implement a simple trailing stop
                # In a full implementation, you'd want more sophisticated risk management
                logger.debug(f"Risk management for {symbol}: {qty} shares at ${current_price}")
                
        except Exception as e:
            logger.error(f"Error updating risk management: {e}")
    
    def _assess_volume_trend(self, market_data: Dict) -> str:
        """Assess overall volume trend across stocks"""
        try:
            volume_ratios = []
            for symbol, data in market_data.items():
                tech_indicators = data.get('technical_indicators', {})
                current_volume = data.get('price_data', {}).get('volume', 0)
                avg_volume = tech_indicators.get('volume_sma', 0)
                
                if avg_volume > 0:
                    ratio = current_volume / avg_volume
                    volume_ratios.append(ratio)
            
            if volume_ratios:
                avg_ratio = np.mean(volume_ratios)
                if avg_ratio > 1.2:
                    return "high"
                elif avg_ratio < 0.8:
                    return "low"
                else:
                    return "normal"
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error assessing volume trend: {e}")
            return "unknown"
    
    def _calculate_spy_comparison(self, portfolio_history: pd.DataFrame) -> Dict:
        """Calculate portfolio performance vs SPY"""
        try:
            if portfolio_history.empty:
                return {'comparison': 'insufficient_data'}
            
            # Get SPY data for the same period
            spy_data = self.data_sources.get_spy_data()
            spy_hist = pd.DataFrame(spy_data['historical_data'])
            
            if spy_hist.empty:
                return {'comparison': 'no_spy_data'}
            
            # Calculate returns
            portfolio_returns = portfolio_history['profit_loss_pct'].iloc[-1] if not portfolio_history.empty else 0
            spy_returns = spy_data['day_change_pct']
            
            outperformance = portfolio_returns - spy_returns
            
            return {
                'portfolio_return': portfolio_returns,
                'spy_return': spy_returns,
                'outperformance': outperformance,
                'is_outperforming': outperformance > 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating SPY comparison: {e}")
            return {'comparison': 'error'}
    
    def _log_trading_session(self, session_results: Dict):
        """Log trading session results"""
        try:
            logger.info("=== TRADING SESSION SUMMARY ===")
            logger.info(f"Status: {session_results['status']}")
            logger.info(f"Duration: {session_results.get('duration_seconds', 0):.2f} seconds")
            
            sentiment = session_results.get('sentiment_analysis', {})
            logger.info(f"Market Sentiment: {sentiment.get('overall_sentiment', 'Unknown')}")
            logger.info(f"Sentiment Score: {sentiment.get('sentiment_score', 0):.2f}")
            
            executed_trades = session_results.get('executed_trades', [])
            successful_trades = [t for t in executed_trades if t['status'] == 'submitted']
            logger.info(f"Trades Executed: {len(successful_trades)}")
            
            for trade in successful_trades:
                trade_info = trade['trade']
                logger.info(f"  - {trade_info['action'].upper()} {trade_info['shares']} {trade_info['symbol']} @ ~${trade_info['estimated_price']:.2f}")
            
            portfolio = session_results.get('portfolio_state', {})
            logger.info(f"Portfolio Value: ${portfolio.get('portfolio_value', 0):,.2f}")
            logger.info(f"Cash: ${portfolio.get('cash', 0):,.2f} ({portfolio.get('cash_percentage', 0):.1f}%)")
            
            spy_comparison = portfolio.get('spy_comparison', {})
            if 'outperformance' in spy_comparison:
                logger.info(f"SPY Outperformance: {spy_comparison['outperformance']:.2f}%")
            
            logger.info("=== END SESSION SUMMARY ===")
            
        except Exception as e:
            logger.error(f"Error logging trading session: {e}")
    
    def _summarize_market_data(self, market_data: Dict) -> Dict:
        """Create summary of market data for logging"""
        return {
            'news_count': len(market_data.get('news', [])),
            'stocks_analyzed': len(market_data.get('market_data', {})),
            'spy_price': market_data.get('market_overview', {}).get('spy_close', 0),
            'spy_change': market_data.get('market_overview', {}).get('spy_change_pct', 0),
            'vix_level': market_data.get('market_overview', {}).get('vix_level', 0)
        }
    
    def _get_default_market_intelligence(self) -> Dict:
        """Default market intelligence when data gathering fails"""
        return {
            'news': [],
            'stock_news': {},
            'market_data': {},
            'market_overview': {
                'spy_close': 400.0,
                'spy_change_pct': 0.0,
                'vix_level': 20.0,
                'vix_change_pct': 0.0,
                'volume_trend': 'unknown'
            },
            'spy_data': self.data_sources._get_default_spy_data(),
            'vix_data': {'level': 20.0, 'change_pct': 0.0},
            'economic_events': []
        }
    
    def _get_default_portfolio_state(self) -> Dict:
        """Default portfolio state when data gathering fails"""
        return {
            'account_info': {},
            'positions': [],
            'portfolio_value': 100000,
            'cash': 100000,
            'equity': 0,
            'cash_percentage': 100,
            'equity_percentage': 0,
            'position_count': 0,
            'available_buying_power': 100000,
            'spy_comparison': {'comparison': 'error'},
            'portfolio_history': pd.DataFrame()
        }
    
    def get_trade_journal(self) -> List[Dict]:
        """Get the current trade journal"""
        return self.trade_journal.copy()
    
    def clear_trade_journal(self):
        """Clear the trade journal"""
        self.trade_journal = []
        logger.info("Trade journal cleared") 