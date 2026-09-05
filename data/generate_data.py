import os
import logging
import numpy as np
import pandas as pd

NUM_TRANSACTIONS = 25_000
RANDOM_SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "transactions.csv")

PAYMENT_METHODS = ["UPI", "Card", "NetBanking", "Wallet"]
PAYMENT_METHOD_PROBS = [0.50, 0.30, 0.15, 0.05]

ORDER_CATEGORIES = ["Electronics", "Apparel", "Grocery", "Furniture", "Digital_Goods"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("data_generator")

def generate_transactions(n: int = NUM_TRANSACTIONS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    logger.info(f"Generating {n} synthetic transactions (seed={seed})...")

    transaction_ids = [f"TXN_{i + 1:05d}" for i in range(n)]
    merchant_ids = rng.randint(1, 201, size=n)
    customer_ids = rng.randint(1, 2001, size=n)

    raw_amount = rng.lognormal(mean=8.5, sigma=1.2, size=n)
    amount = np.clip(raw_amount, 100, 100_000).round(2)

    delivery_distance_km = np.clip(rng.exponential(scale=50, size=n), 0, 800).round(2)
    customer_tenure_days = np.clip(rng.exponential(scale=400, size=n), 0, 2000).astype(int)
    payment_failure_history = np.clip(rng.poisson(lam=1.0, size=n), 0, 8)
    hours_since_last_order = np.clip(rng.exponential(scale=48, size=n), 0, 720).round(2)
    transaction_velocity_5min = np.clip(rng.poisson(lam=1.5, size=n), 0, 20)
    device_age_days = rng.randint(0, 1501, size=n)
    attempted_payment_method_changes = np.clip(rng.poisson(lam=0.3, size=n), 0, 4)

    payment_method = rng.choice(PAYMENT_METHODS, size=n, p=PAYMENT_METHOD_PROBS)
    order_item_category = rng.choice(ORDER_CATEGORIES, size=n)

    is_foreign_ip = rng.random(size=n) < 0.05

    fraud_prob = np.full(n, 0.05)
    fraud_prob += (amount > 30_000) * 0.15
    fraud_prob += is_foreign_ip * 0.20
    fraud_prob += (transaction_velocity_5min > 5) * 0.10
    fraud_prob += (customer_tenure_days < 30) * 0.10
    fraud_prob += (payment_failure_history > 3) * 0.05
    fraud_prob = np.clip(fraud_prob, 0, 1)
    is_fraud = rng.binomial(1, fraud_prob)

    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "merchant_id": merchant_ids,
        "customer_id": customer_ids,
        "amount": amount,
        "payment_method": payment_method,
        "delivery_distance_km": delivery_distance_km,
        "customer_tenure_days": customer_tenure_days,
        "order_item_category": order_item_category,
        "payment_failure_history": payment_failure_history,
        "is_foreign_ip": is_foreign_ip,
        "hours_since_last_order": hours_since_last_order,
        "transaction_velocity_5min": transaction_velocity_5min,
        "device_age_days": device_age_days,
        "attempted_payment_method_changes": attempted_payment_method_changes,
        "is_fraud": is_fraud,
    })

    logger.info(f"Fraud rate: {is_fraud.mean() * 100:.2f}%")
    return df

def main() -> None:
    df = generate_transactions()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
