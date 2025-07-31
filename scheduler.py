"""
Scheduler for automated trading sessions
"""
import logging
import time
import signal
import sys
from typing import Dict, List, Callable
from datetime import datetime, timedelta
import schedule
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from config import config
from trading_strategy import TradingStrategy
from portfolio_tracker import PortfolioTracker

logger = logging.getLogger(__name__)

class TradingScheduler:
    """Automated trading session scheduler"""
    
    def __init__(self):
        """Initialize the trading scheduler"""
        try:
            self.trading_strategy = TradingStrategy()
            self.portfolio_tracker = PortfolioTracker()
            
            # Set up timezone
            self.timezone = pytz.timezone(config.TIMEZONE)
            
            # Initialize scheduler
            self.scheduler = BlockingScheduler(timezone=self.timezone)
            
            # Track scheduler state
            self.is_running = False
            self.last_trading_session = None
            self.session_count = 0
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("Trading scheduler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading scheduler: {e}")
            raise
    
    def setup_trading_schedule(self):
        """Set up the automated trading schedule"""
        try:
            trading_hours = config.get_trading_hours()
            
            for hour_minute in trading_hours:
                try:
                    hour, minute = hour_minute.split(':')
                    hour = int(hour)
                    minute = int(minute)
                    
                    # Add trading session job
                    self.scheduler.add_job(
                        func=self.execute_scheduled_trading_session,
                        trigger=CronTrigger(
                            hour=hour,
                            minute=minute,
                            day_of_week='mon-fri'  # Only on weekdays
                        ),
                        id=f'trading_session_{hour}_{minute}',
                        name=f'Trading Session {hour:02d}:{minute:02d}',
                        max_instances=1,
                        coalesce=True
                    )
                    
                    logger.info(f"Scheduled trading session for {hour:02d}:{minute:02d} EST on weekdays")
                    
                except ValueError as e:
                    logger.error(f"Invalid time format in trading schedule: {hour_minute}")
                    continue
            
            # Add portfolio tracking job (once per hour during market hours)
            self.scheduler.add_job(
                func=self.update_portfolio_tracking,
                trigger=CronTrigger(
                    minute=0,
                    hour='9-16',  # Market hours
                    day_of_week='mon-fri'
                ),
                id='portfolio_tracking',
                name='Portfolio Tracking Update',
                max_instances=1,
                coalesce=True
            )
            
            # Add daily performance report (end of trading day)
            self.scheduler.add_job(
                func=self.generate_daily_report,
                trigger=CronTrigger(
                    hour=17,
                    minute=0,
                    day_of_week='mon-fri'
                ),
                id='daily_report',
                name='Daily Performance Report',
                max_instances=1,
                coalesce=True
            )
            
            # Add weekly performance report (Friday evening)
            self.scheduler.add_job(
                func=self.generate_weekly_report,
                trigger=CronTrigger(
                    hour=18,
                    minute=0,
                    day_of_week='fri'
                ),
                id='weekly_report',
                name='Weekly Performance Report',
                max_instances=1,
                coalesce=True
            )
            
            logger.info("Trading schedule setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up trading schedule: {e}")
            raise
    
    def start_scheduler(self):
        """Start the automated trading scheduler"""
        try:
            logger.info("=== STARTING AI TRADING SCHEDULER ===")
            logger.info(f"Timezone: {config.TIMEZONE}")
            logger.info(f"Trading Schedule: {config.TRADING_SCHEDULE}")
            logger.info(f"Stock Universe: {len(config.STOCK_UNIVERSE)} stocks")
            
            # Print scheduled jobs
            jobs = self.scheduler.get_jobs()
            logger.info(f"Scheduled {len(jobs)} jobs:")
            for job in jobs:
                next_run = job.next_run_time
                logger.info(f"  - {job.name}: Next run at {next_run}")
            
            self.is_running = True
            logger.info("Scheduler started - Press Ctrl+C to stop")
            
            # Start the scheduler (this will block)
            self.scheduler.start()
            
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler shutdown requested")
            self.stop_scheduler()
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
            raise
    
    def stop_scheduler(self):
        """Stop the scheduler gracefully"""
        try:
            if self.is_running:
                logger.info("Stopping trading scheduler...")
                self.scheduler.shutdown(wait=True)
                self.is_running = False
                logger.info("Scheduler stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    
    def execute_scheduled_trading_session(self):
        """Execute a scheduled trading session"""
        try:
            session_start = datetime.now(self.timezone)
            self.session_count += 1
            
            logger.info(f"=== SCHEDULED TRADING SESSION #{self.session_count} ===")
            logger.info(f"Start time: {session_start.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # Check if it's a trading day and market should be open
            if not self._is_trading_day():
                logger.info("Not a trading day, skipping session")
                return
            
            # Execute the trading session
            session_results = self.trading_strategy.execute_trading_session()
            
            # Update portfolio tracking
            self.portfolio_tracker.update_performance()
            
            # Log session summary
            self._log_session_summary(session_results)
            
            self.last_trading_session = session_start
            
            logger.info(f"=== TRADING SESSION #{self.session_count} COMPLETED ===")
            
        except Exception as e:
            logger.error(f"Error in scheduled trading session: {e}")
    
    def update_portfolio_tracking(self):
        """Update portfolio performance tracking"""
        try:
            logger.info("Updating portfolio performance tracking...")
            
            performance_data = self.portfolio_tracker.update_performance()
            
            if performance_data:
                current = performance_data.get('current', {})
                logger.info(f"Portfolio Value: ${current.get('portfolio_value', 0):,.2f}")
                
                spy_comparison = performance_data.get('historical', {})
                outperformance = spy_comparison.get('spy_outperformance', 0)
                logger.info(f"SPY Outperformance: {outperformance:.2f}%")
            
        except Exception as e:
            logger.error(f"Error updating portfolio tracking: {e}")
    
    def generate_daily_report(self):
        """Generate daily performance report"""
        try:
            logger.info("Generating daily performance report...")
            
            report = self.portfolio_tracker.generate_performance_report(days=1)
            
            # Log key metrics
            portfolio_summary = report.get('portfolio_summary', {})
            benchmark_comparison = report.get('benchmark_comparison', {})
            
            if portfolio_summary and benchmark_comparison:
                logger.info("=== DAILY PERFORMANCE REPORT ===")
                logger.info(f"Portfolio Value: ${portfolio_summary.get('current_value', 0):,.2f}")
                logger.info(f"Daily Return: {portfolio_summary.get('avg_daily_return', 0):.2f}%")
                logger.info(f"SPY Outperformance: {benchmark_comparison.get('outperformance', 0):.2f}%")
                
                recommendations = report.get('recommendations', [])
                for rec in recommendations[:3]:  # Show top 3 recommendations
                    logger.info(f"Recommendation: {rec}")
            
            # Save report to file
            self._save_report(report, 'daily')
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
    
    def generate_weekly_report(self):
        """Generate weekly performance report"""
        try:
            logger.info("Generating weekly performance report...")
            
            report = self.portfolio_tracker.generate_performance_report(days=7)
            
            # Generate performance charts
            chart_files = self.portfolio_tracker.create_performance_charts()
            
            # Log comprehensive summary
            self._log_weekly_summary(report)
            
            # Save report to file
            self._save_report(report, 'weekly')
            
            logger.info(f"Weekly report completed with {len(chart_files)} charts generated")
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
    
    def run_manual_session(self) -> Dict:
        """Run a manual trading session (for testing)"""
        try:
            logger.info("=== MANUAL TRADING SESSION ===")
            
            session_results = self.trading_strategy.execute_trading_session()
            self.portfolio_tracker.update_performance()
            
            return session_results
            
        except Exception as e:
            logger.error(f"Error in manual trading session: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_scheduler_status(self) -> Dict:
        """Get current scheduler status"""
        try:
            jobs = self.scheduler.get_jobs()
            
            job_info = []
            for job in jobs:
                job_info.append({
                    'id': job.id,
                    'name': job.name,
                    'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                    'func': job.func.__name__
                })
            
            return {
                'is_running': self.is_running,
                'session_count': self.session_count,
                'last_session': self.last_trading_session.isoformat() if self.last_trading_session else None,
                'scheduled_jobs': job_info,
                'timezone': str(self.timezone)
            }
            
        except Exception as e:
            logger.error(f"Error getting scheduler status: {e}")
            return {'status': 'error'}
    
    def _is_trading_day(self) -> bool:
        """Check if today is a trading day"""
        try:
            now = datetime.now(self.timezone)
            
            # Check if it's a weekday (Monday = 0, Friday = 4)
            if now.weekday() > 4:
                return False
            
            # Here you could add additional checks for holidays
            # For now, we'll just check weekdays
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trading day: {e}")
            return False
    
    def _log_session_summary(self, session_results: Dict):
        """Log trading session summary"""
        try:
            if session_results.get('status') == 'completed':
                executed_trades = session_results.get('executed_trades', [])
                successful_trades = [t for t in executed_trades if t.get('status') == 'submitted']
                
                sentiment = session_results.get('sentiment_analysis', {})
                portfolio = session_results.get('portfolio_state', {})
                
                logger.info(f"Session Duration: {session_results.get('duration_seconds', 0):.1f}s")
                logger.info(f"Market Sentiment: {sentiment.get('overall_sentiment', 'Unknown')}")
                logger.info(f"Trades Executed: {len(successful_trades)}")
                logger.info(f"Portfolio Value: ${portfolio.get('portfolio_value', 0):,.2f}")
                
            elif session_results.get('status') == 'skipped':
                logger.info(f"Session skipped: {session_results.get('reason', 'Unknown')}")
            else:
                logger.warning(f"Session failed: {session_results.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error logging session summary: {e}")
    
    def _log_weekly_summary(self, report: Dict):
        """Log weekly performance summary"""
        try:
            logger.info("=== WEEKLY PERFORMANCE SUMMARY ===")
            
            portfolio_summary = report.get('portfolio_summary', {})
            benchmark_comparison = report.get('benchmark_comparison', {})
            risk_metrics = report.get('risk_metrics', {})
            trading_analysis = report.get('trading_analysis', {})
            
            # Portfolio performance
            if portfolio_summary:
                logger.info(f"Current Value: ${portfolio_summary.get('current_value', 0):,.2f}")
                logger.info(f"Total Return: {portfolio_summary.get('total_return_pct', 0):.2f}%")
                logger.info(f"Win Rate: {portfolio_summary.get('win_rate', 0):.1f}%")
                logger.info(f"Best Day: {portfolio_summary.get('best_day', 0):.2f}%")
                logger.info(f"Worst Day: {portfolio_summary.get('worst_day', 0):.2f}%")
            
            # Benchmark comparison
            if benchmark_comparison:
                logger.info(f"SPY Return: {benchmark_comparison.get('spy_return', 0):.2f}%")
                logger.info(f"Outperformance: {benchmark_comparison.get('outperformance', 0):.2f}%")
                logger.info(f"Beta: {benchmark_comparison.get('beta', 1):.2f}")
            
            # Risk metrics
            if risk_metrics:
                logger.info(f"Volatility: {risk_metrics.get('volatility', 0):.2f}%")
                logger.info(f"VaR (95%): {risk_metrics.get('value_at_risk_95', 0):.2f}%")
            
            # Trading activity
            if trading_analysis:
                logger.info(f"Position Changes: {trading_analysis.get('position_changes', 0)}")
                logger.info(f"Symbols Traded: {trading_analysis.get('unique_symbols_traded', 0)}")
            
            # Recommendations
            recommendations = report.get('recommendations', [])
            logger.info("Key Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                logger.info(f"  {i}. {rec}")
            
            logger.info("=== END WEEKLY SUMMARY ===")
            
        except Exception as e:
            logger.error(f"Error logging weekly summary: {e}")
    
    def _save_report(self, report: Dict, report_type: str):
        """Save report to file"""
        try:
            import json
            import os
            
            # Create reports directory
            reports_dir = 'reports'
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{report_type}_report_{timestamp}.json"
            filepath = os.path.join(reports_dir, filename)
            
            # Save report
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Saved {report_type} report to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving {report_type} report: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop_scheduler()
        sys.exit(0)

def run_scheduler():
    """Main function to run the trading scheduler"""
    try:
        # Validate configuration
        config.validate_config()
        
        # Create and start scheduler
        scheduler = TradingScheduler()
        scheduler.setup_trading_schedule()
        scheduler.start_scheduler()
        
    except Exception as e:
        logger.error(f"Failed to run trading scheduler: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_scheduler() 