import time
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
from threading import Lock
from backend.logging_config import get_logger

logger = get_logger('cache')


class TTLCache:
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.time() < expiry:
                    # Move to end (LRU)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    logger.debug(f"Cache hit: {key}")
                    return value
                else:
                    # Expired, remove
                    del self._cache[key]
                    logger.debug(f"Cache expired: {key}")
            
            self._misses += 1
            logger.debug(f"Cache miss: {key}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        
        with self._lock:
            # Remove oldest items if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Cache evicted: {oldest_key}")
            
            self._cache[key] = (value, expiry)
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
    
    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache deleted: {key}")
                return True
            return False
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, (value, expiry) in self._cache.items():
                if current_time >= expiry:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': round(hit_rate, 2),
                'total_requests': total_requests
            }


class VelocityTracker:
    
    def __init__(self, bucket_size_seconds: int = 30):
        self.bucket_size = bucket_size_seconds
        self._data: Dict[Tuple[float, float], Dict[int, list]] = {}
        self._lock = Lock()
    
    def add_transaction(self, customer_id: float, account_no: float, amount: float) -> None:
        key = (customer_id, account_no)
        current_time = time.time()
        bucket = int(current_time // self.bucket_size)
        
        with self._lock:
            if key not in self._data:
                self._data[key] = {}
            
            if bucket not in self._data[key]:
                self._data[key][bucket] = []
            
            self._data[key][bucket].append({
                'amount': amount,
                'timestamp': current_time
            })
            
            # Cleanup old buckets (keep last 2 hours worth)
            cutoff_bucket = bucket - (7200 // self.bucket_size)
            old_buckets = [b for b in self._data[key].keys() if b < cutoff_bucket]
            for old_bucket in old_buckets:
                del self._data[key][old_bucket]
    
    def get_velocity_stats(self, customer_id: float, account_no: float) -> Dict[str, Any]:
        key = (customer_id, account_no)
        current_time = time.time()
        
        with self._lock:
            if key not in self._data:
                return {
                    'txn_count_30s': 0,
                    'txn_count_10min': 0,
                    'txn_count_1hour': 0,
                    'session_spending': 0,
                    'time_since_last': 3600
                }
            
            # Calculate counts for different time windows
            counts = {'30s': 0, '10min': 0, '1hour': 0}
            total_spending = 0
            all_transactions = []
            
            # Collect transactions from relevant buckets
            current_bucket = int(current_time // self.bucket_size)
            for window, seconds in [('30s', 30), ('10min', 600), ('1hour', 3600)]:
                buckets_to_check = max(1, seconds // self.bucket_size + 1)
                
                for i in range(int(buckets_to_check)):
                    bucket = current_bucket - i
                    if bucket in self._data[key]:
                        for txn in self._data[key][bucket]:
                            if current_time - txn['timestamp'] <= seconds:
                                counts[window] += 1
                                if window == '1hour':  # Only count spending once
                                    total_spending += txn['amount']
                                    all_transactions.append(txn)
            
            # Calculate time since last transaction
            time_since_last = 3600
            if all_transactions:
                latest_txn = max(all_transactions, key=lambda x: x['timestamp'])
                time_since_last = current_time - latest_txn['timestamp']
            
            return {
                'txn_count_30s': counts['30s'],
                'txn_count_10min': counts['10min'],
                'txn_count_1hour': counts['1hour'],
                'session_spending': total_spending,
                'time_since_last': time_since_last
            }
    
    def cleanup_old_data(self) -> int:
        current_time = time.time()
        current_bucket = int(current_time // self.bucket_size)
        cutoff_bucket = current_bucket - (7200 // self.bucket_size)  # 2 hours
        
        cleaned_count = 0
        with self._lock:
            for key in list(self._data.keys()):
                old_buckets = [b for b in self._data[key].keys() if b < cutoff_bucket]
                for bucket in old_buckets:
                    del self._data[key][bucket]
                    cleaned_count += 1
                
                # Remove empty account entries
                if not self._data[key]:
                    del self._data[key]
        
        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} old velocity buckets")
        
        return cleaned_count


# Global cache instances
account_cache = TTLCache(max_size=1000, default_ttl=300)  # 5 minutes
beneficiary_cache = TTLCache(max_size=500, default_ttl=600)  # 10 minutes
velocity_tracker = VelocityTracker(bucket_size_seconds=30)


def get_cached_account_stats(customer_id: float, account_no: float) -> Optional[Dict]:
    cache_key = f"account:{customer_id}:{account_no}"
    return account_cache.get(cache_key)


def set_cached_account_stats(customer_id: float, account_no: float, stats: Dict) -> None:
    cache_key = f"account:{customer_id}:{account_no}"
    account_cache.set(cache_key, stats)


def get_cached_beneficiary_stats(ben_id: float) -> Optional[Dict]:
    if ben_id is None or ben_id <= 0:
        return None
    cache_key = f"beneficiary:{ben_id}"
    return beneficiary_cache.get(cache_key)


def set_cached_beneficiary_stats(ben_id: float, stats: Dict) -> None:
    if ben_id is None or ben_id <= 0:
        return
    cache_key = f"beneficiary:{ben_id}"
    beneficiary_cache.set(cache_key, stats)


def get_cache_stats() -> Dict[str, Any]:
    return {
        'account_cache': account_cache.get_stats(),
        'beneficiary_cache': beneficiary_cache.get_stats(),
        'velocity_tracker_accounts': len(velocity_tracker._data)
    }


def cleanup_all_caches() -> Dict[str, int]:
    return {
        'account_expired': account_cache.cleanup_expired(),
        'beneficiary_expired': beneficiary_cache.cleanup_expired(),
        'velocity_cleaned': velocity_tracker.cleanup_old_data()
    }