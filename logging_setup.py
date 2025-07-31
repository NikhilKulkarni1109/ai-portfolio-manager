"""
Logging setup and configuration
"""
import os
import logging
import logging.handlers
from datetime import datetime, timedelta
import colorlog

from config import config

def setup_logging():
    """Set up comprehensive logging for the AI trading application"""
    try:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(config.LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Colored formatter for console
        colored_formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(colored_formatter)
        root_logger.addHandler(console_handler)
        
        # Main log file handler (rotating)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=config.LOG_FILE,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Trading-specific log file
        trading_log_file = config.LOG_FILE.replace('.log', '_trading.log')
        trading_handler = logging.handlers.RotatingFileHandler(
            filename=trading_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10,
            encoding='utf-8'
        )
        trading_handler.setLevel(logging.INFO)
        trading_handler.setFormatter(detailed_formatter)
        
        # Add trading handler only to trading-related loggers
        trading_loggers = [
            'trading_strategy',
            'alpaca_client',
            'llm_analyzer',
            'portfolio_tracker',
            'scheduler'
        ]
        
        for logger_name in trading_loggers:
            logger = logging.getLogger(logger_name)
            logger.addHandler(trading_handler)
        
        # Error log file (errors and critical only)
        error_log_file = config.LOG_FILE.replace('.log', '_errors.log')
        error_handler = logging.handlers.RotatingFileHandler(
            filename=error_log_file,
            maxBytes=2*1024*1024,  # 2MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # Performance log file (for trade journal and performance metrics)
        performance_log_file = config.LOG_FILE.replace('.log', '_performance.log')
        performance_handler = logging.handlers.TimedRotatingFileHandler(
            filename=performance_log_file,
            when='midnight',
            interval=1,
            backupCount=30,  # Keep 30 days
            encoding='utf-8'
        )
        performance_handler.setLevel(logging.INFO)
        performance_formatter = logging.Formatter(
            fmt='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        performance_handler.setFormatter(performance_formatter)
        
        # Create performance logger
        performance_logger = logging.getLogger('performance')
        performance_logger.addHandler(performance_handler)
        performance_logger.setLevel(logging.INFO)
        performance_logger.propagate = False  # Don't propagate to root logger
        
        # Suppress noisy third-party loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('alpaca_trade_api').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        
        # Set APScheduler to WARNING to reduce noise
        logging.getLogger('apscheduler').setLevel(logging.WARNING)
        
        # Log startup message
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("AI TRADING APPLICATION LOGGING INITIALIZED")
        logger.info(f"Log Level: {config.LOG_LEVEL}")
        logger.info(f"Main Log File: {config.LOG_FILE}")
        logger.info(f"Trading Log: {trading_log_file}")
        logger.info(f"Error Log: {error_log_file}")
        logger.info(f"Performance Log: {performance_log_file}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        return False

def log_trade_execution(symbol: str, action: str, quantity: int, price: float, 
                       reasoning: str, confidence: float, order_id: str = None):
    """Log trade execution to performance log"""
    try:
        performance_logger = logging.getLogger('performance')
        
        trade_data = {
            'type': 'TRADE_EXECUTION',
            'symbol': symbol,
            'action': action.upper(),
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'reasoning': reasoning,
            'confidence': confidence,
            'order_id': order_id
        }
        
        performance_logger.info(f"TRADE: {trade_data}")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to log trade execution: {e}")

def log_portfolio_update(portfolio_value: float, cash: float, positions_count: int,
                        daily_return: float = None, spy_outperformance: float = None):
    """Log portfolio update to performance log"""
    try:
        performance_logger = logging.getLogger('performance')
        
        portfolio_data = {
            'type': 'PORTFOLIO_UPDATE',
            'portfolio_value': portfolio_value,
            'cash': cash,
            'positions_count': positions_count,
            'daily_return': daily_return,
            'spy_outperformance': spy_outperformance
        }
        
        performance_logger.info(f"PORTFOLIO: {portfolio_data}")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to log portfolio update: {e}")

def log_market_sentiment(sentiment: str, score: float, key_themes: list,
                        market_outlook: str, risk_level: str):
    """Log market sentiment analysis to performance log"""
    try:
        performance_logger = logging.getLogger('performance')
        
        sentiment_data = {
            'type': 'MARKET_SENTIMENT',
            'sentiment': sentiment,
            'score': score,
            'key_themes': key_themes,
            'market_outlook': market_outlook,
            'risk_level': risk_level
        }
        
        performance_logger.info(f"SENTIMENT: {sentiment_data}")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to log market sentiment: {e}")

def log_trading_session_start(session_id: str = None):
    """Log trading session start"""
    try:
        performance_logger = logging.getLogger('performance')
        
        session_data = {
            'type': 'SESSION_START',
            'session_id': session_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        performance_logger.info(f"SESSION_START: {session_data}")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to log session start: {e}")

def log_trading_session_end(session_id: str, duration_seconds: float,
                           trades_executed: int, status: str, error: str = None):
    """Log trading session end"""
    try:
        performance_logger = logging.getLogger('performance')
        
        session_data = {
            'type': 'SESSION_END',
            'session_id': session_id,
            'duration_seconds': duration_seconds,
            'trades_executed': trades_executed,
            'status': status,
            'error': error
        }
        
        performance_logger.info(f"SESSION_END: {session_data}")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to log session end: {e}")

def log_performance_metrics(total_return: float, sharpe_ratio: float,
                           max_drawdown: float, win_rate: float,
                           spy_outperformance: float, volatility: float):
    """Log performance metrics"""
    try:
        performance_logger = logging.getLogger('performance')
        
        metrics_data = {
            'type': 'PERFORMANCE_METRICS',
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'spy_outperformance': spy_outperformance,
            'volatility': volatility
        }
        
        performance_logger.info(f"METRICS: {metrics_data}")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to log performance metrics: {e}")

def get_log_summary(days: int = 7) -> dict:
    """Get summary of recent log activity"""
    try:
        import re
        from collections import defaultdict
        
        # Read recent performance log entries
        performance_log_file = config.LOG_FILE.replace('.log', '_performance.log')
        
        if not os.path.exists(performance_log_file):
            return {'status': 'no_log_file'}
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        summary = {
            'trades': 0,
            'portfolio_updates': 0,
            'sessions': 0,
            'errors': 0,
            'last_portfolio_value': None,
            'trading_symbols': set(),
            'sentiment_readings': 0
        }
        
        try:
            with open(performance_log_file, 'r') as f:
                for line in f:
                    try:
                        # Parse timestamp
                        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if not timestamp_match:
                            continue
                        
                        timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                        if timestamp < cutoff_date:
                            continue
                        
                        # Count different log types
                        if 'TRADE:' in line:
                            summary['trades'] += 1
                            # Extract symbol
                            symbol_match = re.search(r"'symbol': '([^']+)'", line)
                            if symbol_match:
                                summary['trading_symbols'].add(symbol_match.group(1))
                        
                        elif 'PORTFOLIO:' in line:
                            summary['portfolio_updates'] += 1
                            # Extract portfolio value
                            value_match = re.search(r"'portfolio_value': ([0-9.]+)", line)
                            if value_match:
                                summary['last_portfolio_value'] = float(value_match.group(1))
                        
                        elif 'SESSION_START:' in line:
                            summary['sessions'] += 1
                        
                        elif 'SENTIMENT:' in line:
                            summary['sentiment_readings'] += 1
                            
                    except Exception:
                        continue
        
        except Exception as e:
            summary['error'] = f"Error reading log file: {e}"
        
        # Convert set to list for JSON serialization
        summary['trading_symbols'] = list(summary['trading_symbols'])
        
        return summary
        
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

class TradingLogger:
    """Specialized logger for trading operations"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.performance_logger = logging.getLogger('performance')
    
    def log_trade(self, symbol: str, action: str, quantity: int, price: float,
                  reasoning: str, confidence: float, order_id: str = None):
        """Log a trade execution"""
        self.logger.info(f"TRADE EXECUTED: {action.upper()} {quantity} {symbol} @ ${price:.2f}")
        log_trade_execution(symbol, action, quantity, price, reasoning, confidence, order_id)
    
    def log_analysis(self, symbol: str, recommendation: str, confidence: float, reasoning: str):
        """Log stock analysis"""
        self.logger.info(f"ANALYSIS: {symbol} -> {recommendation.upper()} (confidence: {confidence:.2f})")
        self.logger.debug(f"ANALYSIS REASONING: {reasoning}")
    
    def log_portfolio_state(self, value: float, cash: float, positions: int):
        """Log current portfolio state"""
        self.logger.info(f"PORTFOLIO: Value=${value:,.2f}, Cash=${cash:,.2f}, Positions={positions}")
        log_portfolio_update(value, cash, positions)
    
    def log_market_condition(self, sentiment: str, spy_change: float, vix_level: float):
        """Log market conditions"""
        self.logger.info(f"MARKET: Sentiment={sentiment}, SPY={spy_change:+.2f}%, VIX={vix_level:.1f}")
    
    def log_error(self, operation: str, error: Exception):
        """Log operation error"""
        self.logger.error(f"ERROR in {operation}: {str(error)}")
    
    def log_session_summary(self, trades_count: int, portfolio_value: float, 
                           duration: float, status: str):
        """Log trading session summary"""
        self.logger.info(
            f"SESSION SUMMARY: {trades_count} trades, Portfolio=${portfolio_value:,.2f}, "
            f"Duration={duration:.1f}s, Status={status}"
        ) 