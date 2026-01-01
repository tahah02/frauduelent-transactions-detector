import logging
import logging.config
import uuid
from contextvars import ContextVar
from typing import Optional
from backend.config import get_config

config = get_config()

# Context variable for request correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class CorrelationFilter(logging.Filter):
    
    def filter(self, record):
        record.correlation_id = correlation_id.get() or 'no-correlation-id'
        return True


class FraudDetectionFormatter(logging.Formatter):
    
    def format(self, record):
        # Add component name based on logger name
        component = record.name.split('.')[-1] if '.' in record.name else record.name
        record.component = component
        
        return super().format(record)


def setup_logging():
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                '()': FraudDetectionFormatter,
                'format': '%(asctime)s | %(levelname)-8s | %(component)-12s | %(correlation_id)s | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(name)s - %(message)s'
            }
        },
        'filters': {
            'correlation': {
                '()': CorrelationFilter,
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': config.LOG_LEVEL,
                'formatter': 'detailed',
                'filters': ['correlation'],
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filters': ['correlation'],
                'filename': 'fraud_detection.log',
                'mode': 'a'
            }
        },
        'loggers': {
            'backend': {
                'level': config.LOG_LEVEL,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'uvicorn': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': config.LOG_LEVEL,
            'handlers': ['console']
        }
    }
    
    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f'backend.{name}')


def set_correlation_id(corr_id: Optional[str] = None) -> str:
    if corr_id is None:
        corr_id = str(uuid.uuid4())[:8]
    
    correlation_id.set(corr_id)
    return corr_id


def get_correlation_id() -> Optional[str]:
    return correlation_id.get()


def log_transaction_start(customer_id: float, account_no: float, amount: float, transfer_type: str):
    logger = get_logger('transaction')
    logger.info(
        f"Transaction analysis started - Customer: {customer_id}, Account: {account_no}, "
        f"Amount: {amount}, Type: {transfer_type}"
    )


def log_transaction_result(status: str, risk_score: float, reasons: list):
    logger = get_logger('transaction')
    logger.info(
        f"Transaction analysis completed - Status: {status}, Risk Score: {risk_score:.4f}, "
        f"Reasons: {len(reasons)} flags"
    )


def log_model_performance(model_name: str, prediction_time_ms: float, success: bool):
    logger = get_logger('performance')
    if success:
        logger.info(f"{model_name} prediction completed in {prediction_time_ms:.2f}ms")
    else:
        logger.warning(f"{model_name} prediction failed after {prediction_time_ms:.2f}ms")


def log_system_health(component: str, status: str, details: dict = None):
    logger = get_logger('health')
    details_str = f" - {details}" if details else ""
    logger.info(f"Health check: {component} = {status}{details_str}")


def log_security_event(event_type: str, details: dict):
    logger = get_logger('security')
    logger.warning(f"Security event: {event_type} - {details}")


# Initialize logging on import
setup_logging()