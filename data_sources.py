"""
Data sources for market data, news, and financial information
"""
import logging
import requests
import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
from newsapi import NewsApiClient

from config import config

logger = logging.getLogger(__name__)

class DataSources:
    """Unified data source manager for market and news data"""
    
    def __init__(self):
        """Initialize data sources"""
        self.news_client = None
        if config.NEWS_API_KEY:
            try:
                self.news_client = NewsApiClient(api_key=config.NEWS_API_KEY)
                logger.info("NewsAPI client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NewsAPI: {e}")
    
    def get_market_news(self, query: str = "stock market", 
                       sources: Optional[List[str]] = None,
                       days_back: int = 1) -> List[Dict]:
        """Get market news from various sources"""
        try:
            if not self.news_client:
                logger.warning("NewsAPI not available, using fallback news source")
                return self._get_fallback_news()
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Use configured sources or default
            news_sources = sources or config.NEWS_SOURCES
            sources_str = ','.join(news_sources)
            
            # Get top headlines
            headlines = self.news_client.get_top_headlines(
                q=query,
                sources=sources_str,
                language='en',
                page_size=50
            )
            
            # Get everything articles
            everything = self.news_client.get_everything(
                q=query,
                sources=sources_str,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='popularity',
                page_size=50
            )
            
            # Combine and deduplicate
            all_articles = headlines.get('articles', []) + everything.get('articles', [])
            
            # Process articles
            processed_news = []
            seen_titles = set()
            
            for article in all_articles:
                title = article.get('title', '')
                if title and title not in seen_titles:
                    processed_news.append({
                        'title': title,
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published_at': article.get('publishedAt', ''),
                        'content': article.get('content', ''),
                        'sentiment_score': self._estimate_news_sentiment(title + ' ' + article.get('description', ''))
                    })
                    seen_titles.add(title)
            
            logger.info(f"Retrieved {len(processed_news)} news articles")
            return processed_news[:30]  # Limit to top 30
            
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return self._get_fallback_news()
    
    def get_stock_news(self, symbols: List[str], days_back: int = 2) -> Dict[str, List[Dict]]:
        """Get news specific to stocks"""
        stock_news = {}
        
        for symbol in symbols:
            try:
                if self.news_client:
                    # Get news for specific stock
                    news = self._get_stock_specific_news(symbol, days_back)
                else:
                    news = []
                
                # Add yfinance news if available
                yf_news = self._get_yfinance_news(symbol)
                news.extend(yf_news)
                
                stock_news[symbol] = news[:10]  # Top 10 per stock
                
            except Exception as e:
                logger.error(f"Error getting news for {symbol}: {e}")
                stock_news[symbol] = []
        
        return stock_news
    
    def get_market_data(self, symbols: List[str], period: str = '5d') -> Dict[str, Dict]:
        """Get comprehensive market data for symbols"""
        market_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get historical data
                hist = ticker.history(period=period)
                if hist.empty:
                    logger.warning(f"No historical data for {symbol}")
                    continue
                
                # Get current info
                info = ticker.info
                
                # Calculate technical indicators
                technical_data = self._calculate_technical_indicators(hist)
                
                # Get current price data
                current_price = hist['Close'].iloc[-1] if not hist.empty else 0
                day_change = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
                
                market_data[symbol] = {
                    'price_data': {
                        'current_price': float(current_price),
                        'day_change_pct': float(day_change),
                        'volume': int(hist['Volume'].iloc[-1]) if not hist.empty else 0,
                        'week_52_high': float(info.get('fiftyTwoWeekHigh', 0)),
                        'week_52_low': float(info.get('fiftyTwoWeekLow', 0)),
                        'market_cap': info.get('marketCap', 0)
                    },
                    'technical_indicators': technical_data,
                    'fundamental_data': {
                        'pe_ratio': info.get('trailingPE', 0),
                        'forward_pe': info.get('forwardPE', 0),
                        'peg_ratio': info.get('pegRatio', 0),
                        'price_to_book': info.get('priceToBook', 0),
                        'debt_to_equity': info.get('debtToEquity', 0),
                        'roe': info.get('returnOnEquity', 0),
                        'revenue_growth': info.get('revenueGrowth', 0)
                    },
                    'historical_data': hist.to_dict('records')[-30:],  # Last 30 days
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown')
                }
                
                logger.debug(f"Retrieved market data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error getting market data for {symbol}: {e}")
                market_data[symbol] = self._get_default_market_data()
        
        return market_data
    
    def get_spy_data(self) -> Dict:
        """Get S&P 500 (SPY) benchmark data"""
        try:
            spy = yf.Ticker(config.SPY_SYMBOL)
            hist = spy.history(period='1mo')
            
            if hist.empty:
                return self._get_default_spy_data()
            
            current_price = hist['Close'].iloc[-1]
            day_change = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
            
            # Calculate returns
            returns = hist['Close'].pct_change().dropna()
            
            return {
                'current_price': float(current_price),
                'day_change_pct': float(day_change),
                'volatility': float(returns.std() * (252 ** 0.5)),  # Annualized volatility
                'volume': int(hist['Volume'].iloc[-1]),
                'historical_data': hist.to_dict('records')[-30:]
            }
            
        except Exception as e:
            logger.error(f"Error getting SPY data: {e}")
            return self._get_default_spy_data()
    
    def get_vix_data(self) -> Dict:
        """Get VIX (volatility index) data"""
        try:
            vix = yf.Ticker('^VIX')
            hist = vix.history(period='5d')
            
            if hist.empty:
                return {'level': 20.0, 'change_pct': 0.0}  # Default VIX level
            
            current_level = hist['Close'].iloc[-1]
            day_change = ((current_level - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
            
            return {
                'level': float(current_level),
                'change_pct': float(day_change)
            }
            
        except Exception as e:
            logger.error(f"Error getting VIX data: {e}")
            return {'level': 20.0, 'change_pct': 0.0}
    
    def get_economic_calendar(self) -> List[Dict]:
        """Get economic events (simplified version)"""
        # This is a simplified implementation
        # In production, you'd use a proper economic calendar API
        return [
            {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'event': 'Market Analysis',
                'importance': 'medium',
                'impact': 'neutral'
            }
        ]
    
    def _get_stock_specific_news(self, symbol: str, days_back: int) -> List[Dict]:
        """Get news specific to a stock symbol"""
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Search for company-specific news
            company_news = self.news_client.get_everything(
                q=f'{symbol} OR {self._get_company_name(symbol)}',
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=20
            )
            
            return [
                {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'sentiment_score': self._estimate_news_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
                }
                for article in company_news.get('articles', [])
            ]
            
        except Exception as e:
            logger.error(f"Error getting stock news for {symbol}: {e}")
            return []
    
    def _get_yfinance_news(self, symbol: str) -> List[Dict]:
        """Get news from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            return [
                {
                    'title': item.get('title', ''),
                    'description': item.get('summary', ''),
                    'url': item.get('link', ''),
                    'source': 'Yahoo Finance',
                    'published_at': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat(),
                    'sentiment_score': self._estimate_news_sentiment(item.get('title', '') + ' ' + item.get('summary', ''))
                }
                for item in news[:10]
            ]
            
        except Exception as e:
            logger.debug(f"No yfinance news for {symbol}: {e}")
            return []
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate basic technical indicators"""
        try:
            if data.empty or len(data) < 14:
                return self._get_default_technical_indicators()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Moving averages
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=min(50, len(data))).mean()
            
            # MACD (simplified)
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            
            return {
                'rsi': float(rsi.iloc[-1]) if not rsi.empty else 50.0,
                'sma_20': float(sma_20.iloc[-1]) if not sma_20.empty else 0.0,
                'sma_50': float(sma_50.iloc[-1]) if not sma_50.empty else 0.0,
                'macd': float(macd.iloc[-1]) if not macd.empty else 0.0,
                'macd_signal': float(signal.iloc[-1]) if not signal.empty else 0.0,
                'volume_sma': float(data['Volume'].rolling(window=20).mean().iloc[-1]) if len(data) >= 20 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return self._get_default_technical_indicators()
    
    def _estimate_news_sentiment(self, text: str) -> float:
        """Simple sentiment estimation for news"""
        if not text:
            return 0.0
        
        positive_words = ['gain', 'rise', 'up', 'bull', 'positive', 'growth', 'strong', 'beat', 'exceed']
        negative_words = ['loss', 'fall', 'down', 'bear', 'negative', 'decline', 'weak', 'miss', 'disappoint']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / max(total_words, 1)
        return max(-1.0, min(1.0, sentiment * 10))  # Scale and clamp between -1 and 1
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name for symbol (simplified mapping)"""
        company_names = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google Alphabet',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'NVDA': 'NVIDIA',
            'META': 'Meta Facebook',
            'NFLX': 'Netflix'
        }
        return company_names.get(symbol, symbol)
    
    def _get_fallback_news(self) -> List[Dict]:
        """Get fallback news when API is unavailable"""
        return [
            {
                'title': 'Market Analysis Unavailable',
                'description': 'News API not configured or unavailable',
                'url': '',
                'source': 'System',
                'published_at': datetime.now().isoformat(),
                'sentiment_score': 0.0
            }
        ]
    
    def _get_default_market_data(self) -> Dict:
        """Default market data structure"""
        return {
            'price_data': {
                'current_price': 0.0,
                'day_change_pct': 0.0,
                'volume': 0,
                'week_52_high': 0.0,
                'week_52_low': 0.0,
                'market_cap': 0
            },
            'technical_indicators': self._get_default_technical_indicators(),
            'fundamental_data': {
                'pe_ratio': 0.0,
                'forward_pe': 0.0,
                'peg_ratio': 0.0,
                'price_to_book': 0.0,
                'debt_to_equity': 0.0,
                'roe': 0.0,
                'revenue_growth': 0.0
            },
            'historical_data': [],
            'sector': 'Unknown',
            'industry': 'Unknown'
        }
    
    def _get_default_technical_indicators(self) -> Dict:
        """Default technical indicators"""
        return {
            'rsi': 50.0,
            'sma_20': 0.0,
            'sma_50': 0.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'volume_sma': 0.0
        }
    
    def _get_default_spy_data(self) -> Dict:
        """Default SPY data"""
        return {
            'current_price': 400.0,
            'day_change_pct': 0.0,
            'volatility': 0.2,
            'volume': 1000000,
            'historical_data': []
        } 