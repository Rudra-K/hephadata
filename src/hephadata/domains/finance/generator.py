from dataclasses import dataclass, field
import datetime
import random
import uuid
import pandas as pd
import numpy as np

@dataclass
class Account:
    """For a single user account with its their attributes."""
    user_id: str = field(default_factory=lambda: f"user_{uuid.uuid4().hex[:8]}")
    persona: str = 'salaried_professional'
    credit_score: int = 750
    account_created_at: datetime.datetime = None
    home_zone: tuple = None
    work_zone: tuple = None

@dataclass
class Transaction:
    """For a single financial transaction."""
    transaction_id: str = field(default_factory=lambda: f"txn_{uuid.uuid4().hex[:12]}")
    user_id: str = ""
    timestamp: datetime.datetime = None
    amount: float = 0.0
    transaction_type: str = "debit"
    merchant_name: str = ""
    merchant_category: str = ""
    transaction_lat: float = 0.0
    transaction_lon: float = 0.0
    is_fraud: int = 0

class FinancialGenerator:
    """
    Generates a dataset of financial transactions.
    """
    def __init__(self, start_date: str, end_date: str):
        self.start_date = datetime.datetime.fromisoformat(start_date)
        self.end_date = datetime.datetime.fromisoformat(end_date)
        self._persona_config = {
            'student': {
                'credit_score_range': (580, 680), 'avg_transactions_per_day': 2.5, 'merchant_preferences': 
                {'Food': ['Zomato', 'Swiggy'], 'Transport': ['Ola', 'Uber'], 'Shopping': ['Amazon']}
            },
            'salaried_professional': {
                'credit_score_range': (700, 850), 'avg_transactions_per_day': 1.5, 'merchant_preferences': 
                {'Groceries': ['DMart', 'Reliance Fresh'], 'Food': ['Barbeque Nation'], 'Finance': ['HDFC ATM'], 'Utilities': ['MSEB']}
            },
        }
        self._zones = {
            'residential': [(18.610, 18.580, 73.785, 73.750), (18.530, 18.500, 73.810, 73.780)],
            'commercial': [(18.590, 18.570, 73.740, 73.720), (18.535, 18.515, 73.860, 73.840)]
        }

    # --- Configuration Methods ---
    def update_persona(self, persona_name: str, config: dict):
        """Adds a new persona or updates an existing one."""
        if persona_name in self._persona_config: self._persona_config[persona_name].update(config)
        else: self._persona_config[persona_name] = config
    
    def remove_persona(self, persona_name: str):
        """Removes a persona from the configuration."""
        if persona_name in self._persona_config:
            del self._persona_config[persona_name]
            print(f"Persona '{persona_name}' removed.")
        else:
            print(f"Warning: Persona '{persona_name}' not found.")

    def set_zones(self, zone_type: str, zone_list: list[tuple]):
        self._zones[zone_type] = zone_list

    def add_zone(self, zone_type: str, bbox: tuple):
        if zone_type not in self._zones: self._zones[zone_type] = []
        self._zones[zone_type].append(bbox)

    # --- Simulation Logic ---
    def _create_user_population(self, n_users: int, persona_mix: dict) -> list[Account]:
        user_population = []
        for persona, percentage in persona_mix.items():
            num_users_for_persona = int(n_users * percentage)
            config = self._persona_config.get(persona)
            if not config: continue
            for _ in range(num_users_for_persona):
                if random.random() < 0.9: days_old = random.randint(365, 3650)
                else: days_old = random.randint(1, 364)
                created_at = self.start_date - datetime.timedelta(days=days_old)
                home_zone = random.choice(self._zones['residential'])
                work_zone = random.choice(self._zones['commercial']) if persona == 'salaried_professional' and 'commercial' in self._zones else None
                user_population.append(Account(
                    persona=persona, credit_score=random.randint(*config['credit_score_range']),
                    account_created_at=created_at, home_zone=home_zone, work_zone=work_zone
                ))
        random.shuffle(user_population)
        return user_population

    def _simulate_day_for_user(self, account: Account, date: datetime.date) -> list[Transaction]:
        daily_transactions = []
        persona_config = self._persona_config[account.persona]
        num_transactions = np.random.poisson(persona_config['avg_transactions_per_day'])
        for _ in range(num_transactions):
            timestamp = datetime.datetime.combine(date, datetime.time(random.randint(7, 22), random.randint(0, 59)))
            amount = round(np.random.lognormal(mean=4.0, sigma=1.5), 2)
            merchant_category = random.choice(list(persona_config['merchant_preferences'].keys()))
            merchant_name = random.choice(persona_config['merchant_preferences'][merchant_category])
            if account.work_zone and date.weekday() < 5: zone = account.work_zone
            else: zone = account.home_zone
            lat = random.uniform(zone[1], zone[0])
            lon = random.uniform(zone[3], zone[2])
            daily_transactions.append(Transaction(
                user_id=account.user_id, timestamp=timestamp, amount=amount,
                merchant_name=merchant_name, merchant_category=merchant_category,
                transaction_lat=round(lat, 6), transaction_lon=round(lon, 6)
            ))
        return daily_transactions

    def _inject_fraud(self, transactions: list[Transaction], accounts: list[Account], fraud_rate: float):
        """Selects transactions and turns them into fraud."""
        accounts_map = {acc.user_id: acc for acc in accounts}
        
        # 1. Get all legitimate transactions once
        legit_transactions = [t for t in transactions if not t.is_fraud]
        random.shuffle(legit_transactions) # Shuffle the list
        
        # 2. Calculate how many to make fraudulent
        num_fraud_transactions = int(len(transactions) * fraud_rate)
        
        # 3. Iterate through the top of the shuffled list
        for i in range(min(num_fraud_transactions, len(legit_transactions))):
            fraud_transaction = legit_transactions[i]
            
            # --- Pick a random fraud tactic ---
            tactic = random.choice(['location', 'amount', 'new_account'])
            
            if tactic == 'location':
                # Impossible location jump
                fraud_transaction.transaction_lat = random.uniform(34.0, 51.0)
                fraud_transaction.transaction_lon = random.uniform(-118.0, -0.1)
                
            elif tactic == 'amount':
                # Unusually high amount for this user
                fraud_transaction.amount *= random.uniform(10, 50)
                
            elif tactic == 'new_account':
                # "Bust-out" fraud: High value transaction on a new account
                account = accounts_map[fraud_transaction.user_id]
                days_since_creation = (fraud_transaction.timestamp - account.account_created_at).days
                if days_since_creation < 60:
                    fraud_transaction.amount = random.uniform(50000, 200000)
                else: # If account is old, just use the 'amount' tactic
                    fraud_transaction.amount *= random.uniform(10, 50)
            
            fraud_transaction.is_fraud = 1
        
        return transactions

    def generate(self, n_users: int, persona_mix: dict, fraud_rate: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
        accounts = self._create_user_population(n_users, persona_mix)
        all_transactions = []
        
        current_date = self.start_date.date()
        while current_date <= self.end_date.date():
            for account in accounts:
                daily_txns = self._simulate_day_for_user(account, current_date)
                all_transactions.extend(daily_txns)
            current_date += datetime.timedelta(days=1)
        
        if fraud_rate > 0:
            all_transactions = self._inject_fraud(all_transactions, accounts, fraud_rate)

        accounts_df = pd.DataFrame([vars(acc) for acc in accounts])
        transactions_df = pd.DataFrame([vars(txn) for txn in all_transactions])
        return accounts_df, transactions_df

# --- Testing Block ---
if __name__ == '__main__':
    generator = FinancialGenerator(start_date="2025-01-01", end_date="2025-01-31")
    
    # Let's remove the default 'student' persona for this simulation
    generator.remove_persona('student')

    accounts_data, transactions_data = generator.generate(
        n_users=1000,
        persona_mix={'salaried_professional': 1.0}, # Simulate only one persona
        fraud_rate=0.01
    )

    print(f"\n--- Generated {len(accounts_data)} accounts ---")
    print(f"--- Generated {len(transactions_data)} total transactions ---")
    
    fraud_txns = transactions_data[transactions_data['is_fraud'] == 1]
    print(f"\n--- Fraud Summary ---")
    print(f"Total fraudulent transactions: {len(fraud_txns)}")
    print(f"Percentage of fraud: {len(fraud_txns) / len(transactions_data):.2%}")
    print("\nSample of fraudulent transactions:")
    print(fraud_txns.head())