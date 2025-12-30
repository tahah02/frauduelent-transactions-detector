# backend/rule_engine.py

TRANSFER_MULTIPLIERS = {'S': 2.0, 'Q': 2.5, 'L': 3.0, 'I': 3.5, 'O': 4.0}
TRANSFER_MIN_FLOORS = {'S': 5000, 'Q': 3000, 'L': 2000, 'I': 1500, 'O': 1000}

MAX_VELOCITY_10MIN = 5
MAX_VELOCITY_1HOUR = 15


def calculate_threshold(user_avg, user_std, transfer_type='O'):
    multiplier = TRANSFER_MULTIPLIERS.get(transfer_type, 3.0)
    floor = TRANSFER_MIN_FLOORS.get(transfer_type, 2000)
    return max(user_avg + multiplier * user_std, floor)


def calculate_all_limits(user_avg, user_std):
    return {t: calculate_threshold(user_avg, user_std, t) for t in ['S', 'I', 'L', 'Q', 'O']}


def check_rule_violation(
    amount,
    user_avg,
    user_std,
    transfer_type,
    txn_count_10min,
    txn_count_1hour,
    monthly_spending
):
    reasons = []
    violated = False
    threshold = calculate_threshold(user_avg, user_std, transfer_type)

    # Velocity – HARD BLOCK
    if txn_count_10min > MAX_VELOCITY_10MIN:
        violated = True
        reasons.append(
            f"Velocity limit exceeded: {txn_count_10min} transactions in last 10 minutes "
            f"(max allowed {MAX_VELOCITY_10MIN})"
        )

    if txn_count_1hour > MAX_VELOCITY_1HOUR:
        violated = True
        reasons.append(
            f"Hourly velocity limit exceeded: {txn_count_1hour} transactions in last 1 hour "
            f"(max allowed {MAX_VELOCITY_1HOUR})"
        )

    # Monthly spending
    projected = monthly_spending + amount
    if projected > threshold:
        violated = True
        reasons.append(
            f"Monthly spending AED {projected:,.2f} exceeds limit AED {threshold:,.2f}"
        )

    return violated, reasons, threshold
