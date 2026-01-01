class FraudDetectionError(Exception):
    
    def __init__(self, message: str, error_code: str = None, context: dict = None):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ModelError(FraudDetectionError):
    pass


class ModelLoadError(ModelError):
    pass


class PredictionError(ModelError):
    pass


class FeatureMismatchError(ModelError):
    pass


class DataError(FraudDetectionError):
    pass


class AccountNotFoundError(DataError):
    pass


class InvalidTransactionError(DataError):
    pass


class DataValidationError(DataError):
    pass


class FileNotFoundError(DataError):
    pass


class ConfigurationError(FraudDetectionError):
    pass


class SystemError(FraudDetectionError):
    pass


class TimeoutError(FraudDetectionError):
    pass


# Error code mappings for user-friendly messages
ERROR_MESSAGES = {
    'MODEL_LOAD_FAILED': 'System temporarily unavailable. Please try again later.',
    'PREDICTION_FAILED': 'Unable to analyze transaction. Please try again.',
    'ACCOUNT_NOT_FOUND': 'Account information not found. Please verify account details.',
    'INVALID_TRANSACTION': 'Invalid transaction details provided.',
    'DATA_VALIDATION_FAILED': 'Transaction data validation failed.',
    'FILE_NOT_FOUND': 'Required system files missing. Please contact support.',
    'CONFIGURATION_ERROR': 'System configuration error. Please contact support.',
    'SYSTEM_ERROR': 'System error occurred. Please try again later.',
    'TIMEOUT_ERROR': 'Request timeout. Please try again.',
    'FEATURE_MISMATCH': 'Feature validation failed. Please contact support.'
}


def get_user_friendly_message(error_code: str) -> str:
    return ERROR_MESSAGES.get(error_code, 'An unexpected error occurred. Please try again.')


def create_error_response(error: FraudDetectionError) -> dict:
    return {
        'error': True,
        'error_code': error.error_code,
        'message': get_user_friendly_message(error.error_code) if error.error_code else str(error),
        'technical_message': str(error),
        'context': error.context
    }