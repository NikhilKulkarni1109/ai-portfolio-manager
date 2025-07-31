"""
Configuration management for AI Trading App
"""
import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TradingConfig:
    """Configuration class for the AI trading application"""
    
    # API Keys and Credentials
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    
    # Trading Parameters
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '10'))
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '100000'))
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))  # 10% of portfolio
    STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.05'))  # 5%
    TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.15'))  # 15%
    
    # Scheduling
    TRADING_SCHEDULE = os.getenv('TRADING_SCHEDULE', '09:30,15:30')  # Pre-market and post-market
    TIMEZONE = os.getenv('TIMEZONE', 'America/New_York')
    
    # Stock Universe
    STOCK_UNIVERSE = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
        'NVDA', 'META', 'NFLX', 'AMD', 'CRM',
        'UBER', 'ABNB', 'COIN', 'PLTR', 'SNOW'
    ]
    
    # S&P 500 Benchmark
    SPY_SYMBOL = 'SPY'
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/trading.log')
    
    # LLM Configuration
    LLM_MODEL = os.getenv('LLM_MODEL', 'gemini-1.5-flash-latest')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.3'))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '4096'))
    
    # News and Data Sources
    NEWS_SOURCES = [
        'reuters', 'bloomberg', 'cnbc', 'marketwatch', 
        'yahoo-finance', 'business-insider'
    ]
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        required_vars = [
            'ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'GEMINI_API_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        return True
    
    @classmethod
    def get_trading_hours(cls) -> List[str]:
        """Get trading schedule as list of hours"""
        return cls.TRADING_SCHEDULE.split(',')
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {
            'max_positions': cls.MAX_POSITIONS,
            'initial_capital': cls.INITIAL_CAPITAL,
            'max_position_size': cls.MAX_POSITION_SIZE,
            'stock_universe_size': len(cls.STOCK_UNIVERSE),
            'trading_schedule': cls.TRADING_SCHEDULE,
            'timezone': cls.TIMEZONE
        }

# Global config instance
config = TradingConfig() 