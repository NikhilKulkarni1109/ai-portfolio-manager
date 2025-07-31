#!/usr/bin/env python3
"""
AI-Powered US Stock Paper Trading Application

Main entry point for the AI trading system that uses LLM insights
to make automated trading decisions via Alpaca's paper trading API.
"""
import sys
import argparse
import signal
from datetime import datetime
import json

# Import our modules
from config import config
from logging_setup import setup_logging, TradingLogger
from trading_strategy import TradingStrategy
from portfolio_tracker import PortfolioTracker
from scheduler import TradingScheduler
from alpaca_client import AlpacaClient
from llm_analyzer import GeminiAnalyzer
from data_sources import DataSources

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def run_trading_session():
    """Run a single trading session"""
    try:
        logger = TradingLogger('main')
        logger.logger.info("=== MANUAL TRADING SESSION ===")
        
        # Initialize trading strategy
        strategy = TradingStrategy()
        
        # Execute trading session
        results = strategy.execute_trading_session()
        
        # Update portfolio tracking
        tracker = PortfolioTracker()
        tracker.update_performance()
        
        # Print results
        print("\n=== TRADING SESSION RESULTS ===")
        print(f"Status: {results.get('status', 'Unknown')}")
        
        if results.get('status') == 'completed':
            print(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
            
            sentiment = results.get('sentiment_analysis', {})
            print(f"Market Sentiment: {sentiment.get('overall_sentiment', 'Unknown')}")
            print(f"Sentiment Score: {sentiment.get('sentiment_score', 0):.2f}")
            
            executed_trades = results.get('executed_trades', [])
            successful_trades = [t for t in executed_trades if t.get('status') == 'submitted']
            print(f"Trades Executed: {len(successful_trades)}")
            
            for trade in successful_trades:
                trade_info = trade['trade']
                print(f"  - {trade_info['action'].upper()} {trade_info['shares']} {trade_info['symbol']} @ ~${trade_info['estimated_price']:.2f}")
            
            portfolio = results.get('portfolio_state', {})
            print(f"Portfolio Value: ${portfolio.get('portfolio_value', 0):,.2f}")
            
        elif results.get('status') == 'skipped':
            print(f"Reason: {results.get('reason', 'Unknown')}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        print(f"Error running trading session: {e}")
        return {'status': 'error', 'error': str(e)}

def run_portfolio_analysis():
    """Run portfolio analysis and generate report"""
    try:
        print("=== PORTFOLIO ANALYSIS ===")
        
        tracker = PortfolioTracker()
        
        # Update current performance
        tracker.update_performance()
        
        # Generate comprehensive report
        report = tracker.generate_performance_report(days=30)
        
        # Print summary
        portfolio_summary = report.get('portfolio_summary', {})
        benchmark_comparison = report.get('benchmark_comparison', {})
        
        if portfolio_summary:
            print(f"Current Value: ${portfolio_summary.get('current_value', 0):,.2f}")
            print(f"Total Return: {portfolio_summary.get('total_return_pct', 0):.2f}%")
            print(f"Win Rate: {portfolio_summary.get('win_rate', 0):.1f}%")
            print(f"Best Day: {portfolio_summary.get('best_day', 0):.2f}%")
            print(f"Worst Day: {portfolio_summary.get('worst_day', 0):.2f}%")
        
        if benchmark_comparison:
            print(f"SPY Return: {benchmark_comparison.get('spy_return', 0):.2f}%")
            print(f"Outperformance: {benchmark_comparison.get('outperformance', 0):.2f}%")
            print(f"Beta: {benchmark_comparison.get('beta', 1):.2f}")
        
        # Show recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        # Generate charts
        print("\nGenerating performance charts...")
        chart_files = tracker.create_performance_charts()
        for chart_file in chart_files:
            print(f"  Created: {chart_file}")
        
        return report
        
    except Exception as e:
        print(f"Error running portfolio analysis: {e}")
        return None

def run_market_analysis():
    """Run market analysis using LLM"""
    try:
        print("=== MARKET ANALYSIS ===")
        
        # Initialize components
        data_sources = DataSources()
        llm_analyzer = GeminiAnalyzer()
        
        # Gather market data
        print("Gathering market data and news...")
        market_news = data_sources.get_market_news(days_back=1)
        spy_data = data_sources.get_spy_data()
        vix_data = data_sources.get_vix_data()
        
        market_overview = {
            'spy_close': spy_data['current_price'],
            'spy_change_pct': spy_data['day_change_pct'],
            'vix_level': vix_data['level'],
            'vix_change_pct': vix_data['change_pct']
        }
        
        # Analyze sentiment
        print("Analyzing market sentiment with AI...")
        sentiment = llm_analyzer.analyze_market_sentiment(market_news, market_overview)
        
        # Print results
        print(f"\nMarket Sentiment: {sentiment.get('overall_sentiment', 'Unknown').upper()}")
        print(f"Sentiment Score: {sentiment.get('sentiment_score', 0):.2f}")
        print(f"Risk Level: {sentiment.get('risk_level', 'Unknown').upper()}")
        print(f"Volatility Expectation: {sentiment.get('volatility_expectation', 'Unknown').upper()}")
        
        key_themes = sentiment.get('key_themes', [])
        if key_themes:
            print(f"Key Themes: {', '.join(key_themes)}")
        
        print(f"\nMarket Outlook: {sentiment.get('market_outlook', 'No outlook available')}")
        print(f"Reasoning: {sentiment.get('reasoning', 'No reasoning available')}")
        
        # Sector insights
        sector_insights = sentiment.get('sector_insights', {})
        if sector_insights:
            print("\nSector Insights:")
            for sector, outlook in sector_insights.items():
                print(f"  {sector.title()}: {outlook.upper()}")
        
        return sentiment
        
    except Exception as e:
        print(f"Error running market analysis: {e}")
        return None

def check_system_status():
    """Check system status and configuration"""
    try:
        print("=== SYSTEM STATUS CHECK ===")
        
        # Check configuration
        print("Checking configuration...")
        try:
            config.validate_config()
            print("✓ Configuration valid")
        except Exception as e:
            print(f"✗ Configuration error: {e}")
            return False
        
        # Test Alpaca connection
        print("Testing Alpaca connection...")
        try:
            alpaca = AlpacaClient()
            account = alpaca.get_account_info()
            print(f"✓ Alpaca connected - Account ID: {account.get('id', 'Unknown')}")
            print(f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
            print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
        except Exception as e:
            print(f"✗ Alpaca connection failed: {e}")
        
        # Test Gemini LLM
        print("Testing Gemini LLM...")
        try:
            llm = GeminiAnalyzer()
            print("✓ Gemini LLM connected")
        except Exception as e:
            print(f"✗ Gemini LLM connection failed: {e}")
        
        # Test data sources
        print("Testing data sources...")
        try:
            data_sources = DataSources()
            spy_data = data_sources.get_spy_data()
            print(f"✓ Data sources working - SPY: ${spy_data.get('current_price', 0):.2f}")
        except Exception as e:
            print(f"✗ Data sources error: {e}")
        
        # Check market status
        try:
            alpaca = AlpacaClient()
            market_open = alpaca.is_market_open()
            print(f"Market Status: {'OPEN' if market_open else 'CLOSED'}")
        except Exception as e:
            print(f"Unable to check market status: {e}")
        
        print("\nSystem configuration:")
        print(f"  Max Positions: {config.MAX_POSITIONS}")
        print(f"  Trading Schedule: {config.TRADING_SCHEDULE}")
        print(f"  Stock Universe: {len(config.STOCK_UNIVERSE)} stocks")
        print(f"  Timezone: {config.TIMEZONE}")
        
        return True
        
    except Exception as e:
        print(f"Error checking system status: {e}")
        return False

def start_automated_trading():
    """Start automated trading scheduler"""
    try:
        print("=== STARTING AUTOMATED TRADING ===")
        
        # Check system status first
        if not check_system_status():
            print("System status check failed. Please fix issues before starting automated trading.")
            return False
        
        # Initialize and start scheduler
        scheduler = TradingScheduler()
        scheduler.setup_trading_schedule()
        
        print("\nStarting automated trading scheduler...")
        print("Press Ctrl+C to stop")
        
        scheduler.start_scheduler()  # This will block
        
    except KeyboardInterrupt:
        print("\nAutomated trading stopped by user")
    except Exception as e:
        print(f"Error starting automated trading: {e}")
        return False

def view_trade_journal():
    """View recent trade journal entries"""
    try:
        print("=== TRADE JOURNAL ===")
        
        strategy = TradingStrategy()
        journal = strategy.get_trade_journal()
        
        if not journal:
            print("No trade journal entries found")
            return
        
        print(f"Found {len(journal)} trade entries:")
        print()
        
        for entry in journal[-10:]:  # Show last 10 trades
            timestamp = entry.get('timestamp', 'Unknown')
            symbol = entry.get('symbol', 'Unknown')
            action = entry.get('action', 'Unknown')
            shares = entry.get('shares', 0)
            price = entry.get('estimated_price', 0)
            reasoning = entry.get('reasoning', 'No reasoning provided')
            confidence = entry.get('confidence', 0)
            
            print(f"Time: {timestamp}")
            print(f"Trade: {action.upper()} {shares} {symbol} @ ${price:.2f}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Reasoning: {reasoning}")
            print("-" * 50)
        
    except Exception as e:
        print(f"Error viewing trade journal: {e}")

def export_data():
    """Export performance data and reports"""
    try:
        print("=== EXPORTING DATA ===")
        
        tracker = PortfolioTracker()
        
        # Generate comprehensive report
        report = tracker.generate_performance_report(days=30)
        
        # Save report to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"performance_export_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Performance report exported to: {filename}")
        
        # Generate charts
        chart_files = tracker.create_performance_charts(output_dir='exported_charts')
        print(f"Generated {len(chart_files)} charts in 'exported_charts' directory")
        
        # Get trade journal
        strategy = TradingStrategy()
        journal = strategy.get_trade_journal()
        
        if journal:
            journal_filename = f"trade_journal_{timestamp}.json"
            with open(journal_filename, 'w') as f:
                json.dump(journal, f, indent=2, default=str)
            print(f"Trade journal exported to: {journal_filename}")
        
        print("Data export completed successfully")
        
    except Exception as e:
        print(f"Error exporting data: {e}")

def main():
    """Main application entry point"""
    setup_signal_handlers()
    
    parser = argparse.ArgumentParser(
        description='AI-Powered US Stock Paper Trading Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --trade            # Run single trading session
  python main.py --start            # Start automated trading
  python main.py --analyze          # Run portfolio analysis
  python main.py --market           # Run market analysis
  python main.py --status           # Check system status
  python main.py --journal          # View trade journal
  python main.py --export           # Export data and reports
        """
    )
    
    parser.add_argument('--trade', action='store_true',
                       help='Run a single trading session')
    parser.add_argument('--start', action='store_true',
                       help='Start automated trading scheduler')
    parser.add_argument('--analyze', action='store_true',
                       help='Run portfolio analysis and generate report')
    parser.add_argument('--market', action='store_true',
                       help='Run market sentiment analysis')
    parser.add_argument('--status', action='store_true',
                       help='Check system status and configuration')
    parser.add_argument('--journal', action='store_true',
                       help='View recent trade journal entries')
    parser.add_argument('--export', action='store_true',
                       help='Export performance data and reports')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    if not setup_logging():
        print("Failed to setup logging")
        sys.exit(1)
    
    # Show banner
    print("=" * 60)
    print("    AI-POWERED US STOCK PAPER TRADING APPLICATION")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: {len(config.STOCK_UNIVERSE)} stocks, ${config.INITIAL_CAPITAL:,.0f} capital")
    print()
    
    try:
        # Execute requested operation
        if args.trade:
            run_trading_session()
        elif args.start:
            start_automated_trading()
        elif args.analyze:
            run_portfolio_analysis()
        elif args.market:
            run_market_analysis()
        elif args.status:
            check_system_status()
        elif args.journal:
            view_trade_journal()
        elif args.export:
            export_data()
        else:
            # No specific action, show menu
            print("No action specified. Available options:")
            print("  --trade     Run single trading session")
            print("  --start     Start automated trading")
            print("  --analyze   Portfolio analysis")
            print("  --market    Market analysis")
            print("  --status    System status check")
            print("  --journal   View trade journal")
            print("  --export    Export data")
            print("\nUse --help for more information")
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 