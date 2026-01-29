"""
Bitcoin Sentiment Data Collector - X (Twitter) API Stream
High-throughput streaming system for Bitcoin-related posts
"""

import tweepy
import json
import sqlite3
from datetime import datetime
import time
import logging
from typing import List, Dict
import queue
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('btc_stream.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BitcoinStreamCollector:
    """High-throughput Bitcoin sentiment data collector from X API"""
    
    def __init__(self, bearer_token: str, db_path: str = "bitcoin_sentiment.db"):
        """
        Initialize the collector
        
        Args:
            bearer_token: X API v2 Bearer Token
            db_path: Path to SQLite database for storing posts
        """
        self.bearer_token = bearer_token
        self.db_path = db_path
        self.client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
        
        # Data queue for batch processing
        self.data_queue = queue.Queue(maxsize=1000)
        
        # Bitcoin-specific keywords (optimized for relevance)
        self.keywords = [
            '$BTC', '#Bitcoin', 'Bitcoin',
            '$btc', '#bitcoin', 'bitcoin',
            'BTC/USD', 'BTCUSD',
            'Satoshi', 'sats',
            'crypto bull run', 'crypto crash',
            'bitcoin price', 'BTC price',
            'bitcoin halving', 'bitcoin etf'
        ]
        
        # Key influencer accounts (add Twitter user IDs here)
        self.key_accounts = {
            # Analysts
            'willy_woo': '2341876000',  # Willy Woo
            'woonomic': '2341876000',
            'PlanB_BTC': '1393879824',  # PlanB
            
            # Exchanges
            'binance': '877807935493033984',
            'coinbase': '574032254',
            'krakenfx': '1520214779',
            
            # Developers & Core Contributors
            'BTCTN': '1366945604',  # Bitcoin Twitter
            'lopp': '337265934',  # Jameson Lopp
            
            # News/Analysis
            'DocumentingBTC': '1366945604',
            'BitcoinMagazine': '1366945604',
            
            # Add more influential accounts here
            # To get user IDs: Use X API or tools like tweeterid.com
        }
        
        self.setup_database()
        
    def setup_database(self):
        """Create SQLite database with optimized schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tweets (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                author_id TEXT,
                author_username TEXT,
                text TEXT,
                metrics_retweet INTEGER DEFAULT 0,
                metrics_reply INTEGER DEFAULT 0,
                metrics_like INTEGER DEFAULT 0,
                metrics_quote INTEGER DEFAULT 0,
                is_influencer BOOLEAN DEFAULT 0,
                language TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                processed BOOLEAN DEFAULT 0,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON tweets(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed ON tweets(processed)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_influencer ON tweets(is_influencer)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment ON tweets(sentiment_label)')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def build_search_query(self) -> str:
        """
        Build optimized search query for X API v2
        Max query length: 1024 characters
        """
        # Combine keywords with OR operator
        keyword_query = ' OR '.join([f'"{kw}"' for kw in self.keywords[:15]])  # Limit to avoid exceeding max length
        
        # Add filters
        query = f"({keyword_query}) lang:en -is:retweet"
        
        logger.info(f"Search query: {query}")
        return query
    
    def start_filtered_stream(self):
        """
        Start filtered stream using X API v2
        This is the most efficient method for high-throughput collection
        """
        try:
            # Delete existing rules
            rules = self.client.get_rules()
            if rules.data:
                rule_ids = [rule.id for rule in rules.data]
                self.client.delete_rules(rule_ids)
                logger.info(f"Deleted {len(rule_ids)} existing rules")
            
            # Create new rule
            query = self.build_search_query()
            rule = tweepy.StreamRule(query)
            self.client.add_rules(rule)
            logger.info("Stream rules added successfully")
            
        except Exception as e:
            logger.error(f"Error setting up stream rules: {e}")
            raise
    
    def search_recent_tweets(self, max_results: int = 100) -> List[Dict]:
        """
        Search recent tweets (last 7 days) - useful for initial data collection
        
        Args:
            max_results: Number of tweets per request (10-100)
        """
        query = self.build_search_query()
        tweets_data = []
        
        try:
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'lang'],
                user_fields=['username', 'verified', 'public_metrics'],
                expansions=['author_id']
            )
            
            if not tweets.data:
                logger.warning("No tweets found")
                return tweets_data
            
            # Create user lookup
            users = {user.id: user for user in tweets.includes['users']}
            
            for tweet in tweets.data:
                author = users.get(tweet.author_id)
                is_influencer = tweet.author_id in self.key_accounts.values()
                
                tweet_data = {
                    'id': tweet.id,
                    'created_at': tweet.created_at,
                    'author_id': tweet.author_id,
                    'author_username': author.username if author else 'unknown',
                    'text': tweet.text,
                    'metrics_retweet': tweet.public_metrics['retweet_count'],
                    'metrics_reply': tweet.public_metrics['reply_count'],
                    'metrics_like': tweet.public_metrics['like_count'],
                    'metrics_quote': tweet.public_metrics['quote_count'],
                    'is_influencer': is_influencer,
                    'language': tweet.lang
                }
                
                tweets_data.append(tweet_data)
            
            logger.info(f"Collected {len(tweets_data)} tweets")
            
        except tweepy.TweepyException as e:
            logger.error(f"Error searching tweets: {e}")
        
        return tweets_data
    
    def monitor_key_accounts(self, lookback_hours: int = 24) -> List[Dict]:
        """
        Get recent tweets from key influencer accounts
        
        Args:
            lookback_hours: How far back to look for tweets
        """
        all_tweets = []
        
        for username, user_id in self.key_accounts.items():
            try:
                tweets = self.client.get_users_tweets(
                    id=user_id,
                    max_results=10,
                    tweet_fields=['created_at', 'public_metrics', 'lang'],
                    exclude=['retweets', 'replies']
                )
                
                if not tweets.data:
                    continue
                
                for tweet in tweets.data:
                    tweet_data = {
                        'id': tweet.id,
                        'created_at': tweet.created_at,
                        'author_id': user_id,
                        'author_username': username,
                        'text': tweet.text,
                        'metrics_retweet': tweet.public_metrics['retweet_count'],
                        'metrics_reply': tweet.public_metrics['reply_count'],
                        'metrics_like': tweet.public_metrics['like_count'],
                        'metrics_quote': tweet.public_metrics['quote_count'],
                        'is_influencer': True,
                        'language': tweet.lang
                    }
                    all_tweets.append(tweet_data)
                
                logger.info(f"Collected {len(tweets.data)} tweets from @{username}")
                time.sleep(1)  # Rate limit protection
                
            except tweepy.TweepyException as e:
                logger.error(f"Error collecting from @{username}: {e}")
                continue
        
        return all_tweets
    
    def save_tweets(self, tweets: List[Dict]):
        """Batch save tweets to database"""
        if not tweets:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for tweet in tweets:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO tweets 
                    (id, created_at, author_id, author_username, text, 
                     metrics_retweet, metrics_reply, metrics_like, metrics_quote,
                     is_influencer, language)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    tweet['id'], tweet['created_at'], tweet['author_id'],
                    tweet['author_username'], tweet['text'],
                    tweet['metrics_retweet'], tweet['metrics_reply'],
                    tweet['metrics_like'], tweet['metrics_quote'],
                    tweet['is_influencer'], tweet['language']
                ))
            except Exception as e:
                logger.error(f"Error saving tweet {tweet['id']}: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(tweets)} tweets to database")
    
    def run_continuous_collection(self, interval_minutes: int = 5):
        """
        Run continuous collection cycle
        
        Args:
            interval_minutes: Minutes between collection cycles
        """
        logger.info("Starting continuous collection...")
        
        while True:
            try:
                # Collect from search
                search_tweets = self.search_recent_tweets(max_results=100)
                self.save_tweets(search_tweets)
                
                # Collect from key accounts
                influencer_tweets = self.monitor_key_accounts()
                self.save_tweets(influencer_tweets)
                
                # Wait before next cycle
                logger.info(f"Sleeping for {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in collection cycle: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about collected data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total tweets
        cursor.execute("SELECT COUNT(*) FROM tweets")
        stats['total_tweets'] = cursor.fetchone()[0]
        
        # Influencer tweets
        cursor.execute("SELECT COUNT(*) FROM tweets WHERE is_influencer = 1")
        stats['influencer_tweets'] = cursor.fetchone()[0]
        
        # Unprocessed tweets
        cursor.execute("SELECT COUNT(*) FROM tweets WHERE processed = 0")
        stats['unprocessed_tweets'] = cursor.fetchone()[0]
        
        # Date range
        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM tweets")
        date_range = cursor.fetchone()
        stats['earliest_tweet'] = date_range[0]
        stats['latest_tweet'] = date_range[1]
        
        conn.close()
        return stats


# Example usage
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual X API Bearer Token
    BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAMWr4gEAAAAAkj%2BTAIrw5KkDRah0Sgsl9q3%2Bzd4%3DMlQmRuANIqheC5zBgrPqfulMNw13NkqwOAHxz2ReonyMvY9NtZ1"
    
    # Initialize collector
    collector = BitcoinStreamCollector(bearer_token=BEARER_TOKEN)
    
    print("\n=== Bitcoin Sentiment Data Collector ===\n")
    print("Options:")
    print("1. Collect recent tweets (one-time)")
    print("2. Monitor key accounts (one-time)")
    print("3. Start continuous collection (runs indefinitely)")
    print("4. Show collection statistics")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == "1":
        tweets = collector.search_recent_tweets(max_results=100)
        collector.save_tweets(tweets)
        print(f"\nCollected and saved {len(tweets)} tweets")
        
    elif choice == "2":
        tweets = collector.monitor_key_accounts()
        collector.save_tweets(tweets)
        print(f"\nCollected and saved {len(tweets)} influencer tweets")
        
    elif choice == "3":
        interval = int(input("Enter collection interval in minutes (e.g., 5): "))
        collector.run_continuous_collection(interval_minutes=interval)
        
    elif choice == "4":
        stats = collector.get_collection_stats()
        print("\n=== Collection Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    else:
        print("Invalid choice")