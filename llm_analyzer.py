"""
Gemini LLM analyzer for market insights and trading decisions
"""
import logging
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import google.generativeai as genai

from config import config

logger = logging.getLogger(__name__)

class GeminiAnalyzer:
    """Gemini LLM analyzer for trading decisions"""
    
    def __init__(self):
        """Initialize Gemini client"""
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.LLM_MODEL)
            
            # Test the connection
            response = self.model.generate_content("Hello, testing connection.")
            logger.info("Successfully connected to Gemini API")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def analyze_market_sentiment(self, news_data: List[Dict], 
                               market_data: Dict) -> Dict:
        """Analyze market sentiment from news and market data"""
        try:
            prompt = self._build_sentiment_prompt(news_data, market_data)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=config.LLM_TEMPERATURE,
                    max_output_tokens=config.MAX_TOKENS
                )
            )
            
            # Parse the response
            sentiment_analysis = self._parse_sentiment_response(response.text)
            
            logger.info(f"Market sentiment analysis completed: {sentiment_analysis.get('overall_sentiment', 'Unknown')}")
            
            return sentiment_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return self._get_default_sentiment()
    
    def generate_trading_decisions(self, sentiment_data: Dict, 
                                 portfolio_data: Dict,
                                 available_stocks: List[str]) -> Dict:
        """Generate trading decisions based on analysis"""
        try:
            prompt = self._build_trading_prompt(sentiment_data, portfolio_data, available_stocks)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=config.LLM_TEMPERATURE,
                    max_output_tokens=config.MAX_TOKENS
                )
            )
            
            # Parse trading decisions
            decisions = self._parse_trading_response(response.text)
            
            logger.info(f"Generated trading decisions for {len(decisions.get('trades', []))} potential trades")
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error generating trading decisions: {e}")
            return self._get_default_trading_decision()
    
    def analyze_individual_stock(self, symbol: str, stock_data: Dict, 
                               news_data: List[Dict]) -> Dict:
        """Analyze individual stock for trading potential"""
        try:
            prompt = self._build_stock_analysis_prompt(symbol, stock_data, news_data)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=config.LLM_TEMPERATURE,
                    max_output_tokens=config.MAX_TOKENS
                )
            )
            
            analysis = self._parse_stock_analysis_response(response.text)
            
            logger.info(f"Completed analysis for {symbol}: {analysis.get('recommendation', 'Unknown')}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {e}")
            return self._get_default_stock_analysis()
    
    def _build_sentiment_prompt(self, news_data: List[Dict], market_data: Dict) -> str:
        """Build prompt for market sentiment analysis"""
        news_summary = "\n".join([
            f"- {item.get('title', '')}: {item.get('description', '')[:200]}..."
            for item in news_data[:10]  # Limit to top 10 news items
        ])
        
        market_summary = f"""
        Market Data Summary:
        - SPY Close: ${market_data.get('spy_close', 'N/A')}
        - SPY Change: {market_data.get('spy_change_pct', 'N/A')}%
        - VIX Level: {market_data.get('vix_level', 'N/A')}
        - Volume Trend: {market_data.get('volume_trend', 'N/A')}
        """
        
        prompt = f"""
        As an expert financial analyst, analyze the current market sentiment based on the following data:

        RECENT NEWS HEADLINES:
        {news_summary}

        {market_summary}

        Please provide a comprehensive market sentiment analysis in JSON format with the following structure:
        {{
            "overall_sentiment": "bullish/bearish/neutral",
            "sentiment_score": -1.0 to 1.0 (negative to positive),
            "key_themes": ["theme1", "theme2", "theme3"],
            "market_outlook": "short analysis of market direction",
            "risk_level": "low/medium/high",
            "volatility_expectation": "low/medium/high",
            "sector_insights": {{
                "technology": "bullish/bearish/neutral",
                "healthcare": "bullish/bearish/neutral",
                "finance": "bullish/bearish/neutral",
                "energy": "bullish/bearish/neutral"
            }},
            "reasoning": "detailed explanation of sentiment analysis"
        }}

        Focus on actionable insights for day trading and swing trading decisions.
        """
        
        return prompt
    
    def _build_trading_prompt(self, sentiment_data: Dict, portfolio_data: Dict, 
                            available_stocks: List[str]) -> str:
        """Build prompt for trading decisions"""
        current_positions = portfolio_data.get('positions', [])
        portfolio_value = portfolio_data.get('portfolio_value', 100000)
        
        positions_summary = "\n".join([
            f"- {pos['symbol']}: {pos['qty']} shares, P&L: ${pos['unrealized_pl']:.2f}"
            for pos in current_positions
        ])
        
        prompt = f"""
        As an expert algorithmic trader, generate trading decisions based on the following analysis:

        MARKET SENTIMENT ANALYSIS:
        {json.dumps(sentiment_data, indent=2)}

        CURRENT PORTFOLIO:
        Portfolio Value: ${portfolio_value:,.2f}
        Current Positions:
        {positions_summary if positions_summary else "No current positions"}

        AVAILABLE STOCKS:
        {', '.join(available_stocks)}

        TRADING CONSTRAINTS:
        - Maximum 10 positions
        - Maximum 10% of portfolio per position
        - Paper trading only (no real money)
        - Focus on risk-adjusted returns

        Please provide trading recommendations in JSON format:
        {{
            "action": "buy/sell/hold",
            "trades": [
                {{
                    "symbol": "STOCK_SYMBOL",
                    "action": "buy/sell",
                    "quantity": number_of_shares,
                    "confidence": 0.0 to 1.0,
                    "reasoning": "detailed explanation",
                    "risk_level": "low/medium/high",
                    "target_price": estimated_price,
                    "stop_loss": stop_loss_price,
                    "time_horizon": "intraday/swing/position"
                }}
            ],
            "portfolio_allocation": {{
                "cash_percentage": 0-100,
                "equity_percentage": 0-100
            }},
            "overall_strategy": "detailed strategy explanation",
            "risk_assessment": "overall risk evaluation",
            "market_timing": "best execution time recommendation"
        }}

        Consider:
        1. Current market sentiment and volatility
        2. Individual stock momentum and news
        3. Risk management and position sizing
        4. Market timing and execution strategy
        """
        
        return prompt
    
    def _build_stock_analysis_prompt(self, symbol: str, stock_data: Dict, 
                                   news_data: List[Dict]) -> str:
        """Build prompt for individual stock analysis"""
        stock_news = "\n".join([
            f"- {item.get('title', '')}"
            for item in news_data if symbol.lower() in item.get('title', '').lower()
        ][:5])  # Top 5 relevant news items
        
        price_data = stock_data.get('price_data', {})
        
        prompt = f"""
        Analyze {symbol} for trading potential:

        STOCK DATA:
        - Current Price: ${price_data.get('current_price', 'N/A')}
        - Day Change: {price_data.get('day_change_pct', 'N/A')}%
        - Volume: {price_data.get('volume', 'N/A')}
        - 52-Week High: ${price_data.get('week_52_high', 'N/A')}
        - 52-Week Low: ${price_data.get('week_52_low', 'N/A')}
        - Market Cap: ${price_data.get('market_cap', 'N/A')}

        RECENT NEWS:
        {stock_news if stock_news else "No specific news found"}

        TECHNICAL INDICATORS:
        - RSI: {stock_data.get('rsi', 'N/A')}
        - MACD: {stock_data.get('macd', 'N/A')}
        - Moving Averages: {stock_data.get('moving_averages', 'N/A')}

        Provide analysis in JSON format:
        {{
            "recommendation": "strong_buy/buy/hold/sell/strong_sell",
            "confidence": 0.0 to 1.0,
            "target_price": estimated_price,
            "stop_loss": recommended_stop_loss,
            "risk_rating": "low/medium/high",
            "time_horizon": "intraday/short_term/medium_term",
            "key_factors": ["factor1", "factor2", "factor3"],
            "technical_analysis": "brief technical summary",
            "fundamental_analysis": "brief fundamental summary",
            "news_impact": "news impact assessment",
            "reasoning": "detailed explanation"
        }}
        """
        
        return prompt
    
    def _parse_sentiment_response(self, response_text: str) -> Dict:
        """Parse sentiment analysis response"""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return self._extract_sentiment_fallback(response_text)
        except Exception as e:
            logger.error(f"Error parsing sentiment response: {e}")
            return self._get_default_sentiment()
    
    def _parse_trading_response(self, response_text: str) -> Dict:
        """Parse trading decision response"""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._extract_trading_fallback(response_text)
        except Exception as e:
            logger.error(f"Error parsing trading response: {e}")
            return self._get_default_trading_decision()
    
    def _parse_stock_analysis_response(self, response_text: str) -> Dict:
        """Parse stock analysis response"""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._extract_stock_analysis_fallback(response_text)
        except Exception as e:
            logger.error(f"Error parsing stock analysis response: {e}")
            return self._get_default_stock_analysis()
    
    def _get_default_sentiment(self) -> Dict:
        """Get default sentiment when analysis fails"""
        return {
            "overall_sentiment": "neutral",
            "sentiment_score": 0.0,
            "key_themes": ["market_uncertainty"],
            "market_outlook": "uncertain market conditions",
            "risk_level": "medium",
            "volatility_expectation": "medium",
            "sector_insights": {
                "technology": "neutral",
                "healthcare": "neutral",
                "finance": "neutral",
                "energy": "neutral"
            },
            "reasoning": "Default sentiment due to analysis error"
        }
    
    def _get_default_trading_decision(self) -> Dict:
        """Get default trading decision when analysis fails"""
        return {
            "action": "hold",
            "trades": [],
            "portfolio_allocation": {
                "cash_percentage": 50,
                "equity_percentage": 50
            },
            "overall_strategy": "Conservative hold due to analysis uncertainty",
            "risk_assessment": "High uncertainty, maintaining current positions",
            "market_timing": "Wait for better analysis"
        }
    
    def _get_default_stock_analysis(self) -> Dict:
        """Get default stock analysis when analysis fails"""
        return {
            "recommendation": "hold",
            "confidence": 0.5,
            "target_price": 0,
            "stop_loss": 0,
            "risk_rating": "medium",
            "time_horizon": "medium_term",
            "key_factors": ["analysis_error"],
            "technical_analysis": "Unable to analyze",
            "fundamental_analysis": "Unable to analyze",
            "news_impact": "Unknown",
            "reasoning": "Default analysis due to error"
        }
    
    def _extract_sentiment_fallback(self, text: str) -> Dict:
        """Fallback sentiment extraction from text"""
        # Simple keyword-based sentiment extraction
        bullish_keywords = ['bullish', 'positive', 'up', 'gain', 'rise', 'buy']
        bearish_keywords = ['bearish', 'negative', 'down', 'loss', 'fall', 'sell']
        
        text_lower = text.lower()
        bullish_count = sum(1 for word in bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in bearish_keywords if word in text_lower)
        
        if bullish_count > bearish_count:
            sentiment = "bullish"
            score = 0.6
        elif bearish_count > bullish_count:
            sentiment = "bearish"
            score = -0.6
        else:
            sentiment = "neutral"
            score = 0.0
        
        return {
            "overall_sentiment": sentiment,
            "sentiment_score": score,
            "key_themes": ["fallback_analysis"],
            "market_outlook": f"Extracted sentiment: {sentiment}",
            "risk_level": "medium",
            "volatility_expectation": "medium",
            "sector_insights": {
                "technology": sentiment,
                "healthcare": sentiment,
                "finance": sentiment,
                "energy": sentiment
            },
            "reasoning": "Fallback text analysis used"
        }
    
    def _extract_trading_fallback(self, text: str) -> Dict:
        """Fallback trading decision extraction"""
        return self._get_default_trading_decision()
    
    def _extract_stock_analysis_fallback(self, text: str) -> Dict:
        """Fallback stock analysis extraction"""
        return self._get_default_stock_analysis() 