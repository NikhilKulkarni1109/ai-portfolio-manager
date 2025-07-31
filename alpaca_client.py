"""
Alpaca API client for paper trading operations
"""
import logging
import time
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError, TimeFrame
import pandas as pd
from datetime import datetime, timedelta

from config import config

logger = logging.getLogger(__name__)

class AlpacaClient:
    """Alpaca API client for paper trading"""
    
    def __init__(self):
        """Initialize Alpaca client"""
        try:
            self.api = tradeapi.REST(
                config.ALPACA_API_KEY,
                config.ALPACA_SECRET_KEY,
                config.ALPACA_BASE_URL,
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca paper trading. Account ID: {account.id}")
            logger.info(f"Buying power: ${float(account.buying_power):,.2f}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            raise
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'id': account.id,
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'day_trade_count': int(account.day_trade_count),
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked
            }
        except APIError as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'current_price': float(pos.current_price),
                    'side': pos.side
                }
                for pos in positions
            ]
        except APIError as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_portfolio_history(self, period: str = '1M') -> pd.DataFrame:
        """Get portfolio history"""
        try:
            portfolio = self.api.get_portfolio_history(period=period, timeframe='1Day')
            
            if not portfolio.timestamp:
                logger.warning("No portfolio history available")
                return pd.DataFrame()
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(portfolio.timestamp, unit='s'),
                'equity': portfolio.equity,
                'profit_loss': portfolio.profit_loss,
                'profit_loss_pct': portfolio.profit_loss_pct
            })
            
            return df.set_index('timestamp')
            
        except APIError as e:
            logger.error(f"Error getting portfolio history: {e}")
            return pd.DataFrame()
    
    def place_order(self, symbol: str, qty: float, side: str, 
                   order_type: str = 'market', time_in_force: str = 'day',
                   limit_price: Optional[float] = None) -> Optional[Dict]:
        """Place a trading order"""
        try:
            # Validate inputs
            if side not in ['buy', 'sell']:
                raise ValueError("Side must be 'buy' or 'sell'")
            
            if order_type not in ['market', 'limit', 'stop', 'stop_limit']:
                raise ValueError("Invalid order type")
            
            order_data = {
                'symbol': symbol,
                'qty': abs(qty),
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force
            }
            
            if order_type in ['limit', 'stop_limit'] and limit_price:
                order_data['limit_price'] = limit_price
            
            order = self.api.submit_order(**order_data)
            
            logger.info(f"Order placed: {side} {qty} {symbol} at {order_type}")
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'order_type': order.order_type,
                'status': order.status,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0
            }
            
        except APIError as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except APIError as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_orders(self, status: str = 'all', limit: int = 100) -> List[Dict]:
        """Get order history"""
        try:
            orders = self.api.list_orders(status=status, limit=limit)
            return [
                {
                    'id': order.id,
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'side': order.side,
                    'order_type': order.order_type,
                    'status': order.status,
                    'submitted_at': order.submitted_at,
                    'filled_at': order.filled_at,
                    'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0
                }
                for order in orders
            ]
        except APIError as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def get_market_data(self, symbols: List[str], timeframe: str = '1Day', 
                       limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Get market data for symbols"""
        try:
            # Convert timeframe string to Alpaca TimeFrame
            tf_map = {
                '1Min': TimeFrame.Minute,
                '5Min': TimeFrame(5, TimeFrame.Minute),
                '15Min': TimeFrame(15, TimeFrame.Minute),
                '1Hour': TimeFrame.Hour,
                '1Day': TimeFrame.Day
            }
            
            tf = tf_map.get(timeframe, TimeFrame.Day)
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=limit)
            
            bars = self.api.get_bars(
                symbols,
                tf,
                start=start_time.strftime('%Y-%m-%d'),
                end=end_time.strftime('%Y-%m-%d'),
                adjustment='raw'
            )
            
            # Convert to DataFrames
            data = {}
            for symbol in symbols:
                symbol_bars = [bar for bar in bars if bar.symbol == symbol]
                if symbol_bars:
                    df = pd.DataFrame([
                        {
                            'timestamp': bar.timestamp,
                            'open': float(bar.open),
                            'high': float(bar.high),
                            'low': float(bar.low),
                            'close': float(bar.close),
                            'volume': int(bar.volume)
                        }
                        for bar in symbol_bars
                    ])
                    df.set_index('timestamp', inplace=True)
                    data[symbol] = df
                else:
                    logger.warning(f"No data available for {symbol}")
                    data[symbol] = pd.DataFrame()
            
            return data
            
        except APIError as e:
            logger.error(f"Error getting market data: {e}")
            return {}
    
    def close_position(self, symbol: str, qty: Optional[float] = None) -> bool:
        """Close a position (full or partial)"""
        try:
            if qty:
                # Close partial position
                self.api.close_position(symbol, qty=qty)
            else:
                # Close full position
                self.api.close_position(symbol)
            
            logger.info(f"Closed position for {symbol}")
            return True
            
        except APIError as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """Close all positions"""
        try:
            self.api.close_all_positions()
            logger.info("All positions closed")
            return True
        except APIError as e:
            logger.error(f"Error closing all positions: {e}")
            return False
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except APIError as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    def get_market_calendar(self, start: str, end: str) -> List[Dict]:
        """Get market calendar"""
        try:
            calendar = self.api.get_calendar(start=start, end=end)
            return [
                {
                    'date': str(day.date),
                    'open': str(day.open),
                    'close': str(day.close)
                }
                for day in calendar
            ]
        except APIError as e:
            logger.error(f"Error getting market calendar: {e}")
            return [] 