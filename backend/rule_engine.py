from backend.config import get_config

config = get_config()


def calculate_threshold(user_avg, user_std, transfer_type='O'):
    multiplier = config.TRANSFER_MULTIPLIERS.get(transfer_type, 3.0)
    floor = config.TRANSFER_MIN_FLOORS.get(transfer_type, 2000)
    return max(user_avg + multiplier * user_std, floor)


def calculate_all_limits(user_avg, user_std):
    return {t: calculate_threshold(user_avg, user_std, t) for t in ['S', 'I', 'L', 'Q', 'O']}


def check_rule_violation(amount, user_avg, user_std, transfer_type, 
                         txn_count_10min, txn_count_1hour, monthly_spending):
    reasons = []
    violated = False
    threshold = calculate_threshold(user_avg, user_std, transfer_type)

    # Velocity checks
    if txn_count_10min > config.MAX_VELOCITY_10MIN:
        violated = True
        reasons.append(
            f"Velocity limit exceeded: {txn_count_10min} transactions in last 10 minutes "
            f"(max allowed {config.MAX_VELOCITY_10MIN})"
        )

    if txn_count_1hour > config.MAX_VELOCITY_1HOUR:
        violated = True
        reasons.append(
            f"Hourly velocity limit exceeded: {txn_count_1hour} transactions in last 1 hour "
            f"(max allowed {config.MAX_VELOCITY_1HOUR})"
        )

    projected = monthly_spending + amount
    if projected > threshold:
        violated = True
        reasons.append(
            f"Monthly spending AED {projected:,.2f} exceeds limit AED {threshold:,.2f}"
        )

    return violated, reasons, threshold
