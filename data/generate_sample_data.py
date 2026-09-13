"""
generate_sample_data.py
------------------------
Creates a synthetic "Customer Churn" dataset matching the schema of the
popular Kaggle Telco Customer Churn dataset. Used so this repo works fully
out of the box, with no external downloads required.

For a real deployment, swap this for the actual Kaggle dataset:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
(just keep the column names: tenure, MonthlyCharges, Contract, Churn)
"""

import csv
import os
import random

random.seed(42)


def generate_row(customer_id):
    gender = random.choice(["Male", "Female"])
    senior_citizen = random.choice([0, 0, 0, 1])
    partner = random.choice(["Yes", "No"])
    dependents = random.choice(["Yes", "No"])
    tenure = random.randint(0, 72)
    contract = random.choices(
        ["Month-to-month", "One year", "Two year"],
        weights=[0.55, 0.25, 0.20]
    )[0]
    monthly_charges = round(random.uniform(18.0, 120.0), 2)

    churn_score = 0
    if contract == "Month-to-month":
        churn_score += 2
    elif contract == "One year":
        churn_score += 1

    if tenure < 12:
        churn_score += 2
    elif tenure < 24:
        churn_score += 1

    if monthly_charges > 80:
        churn_score += 1

    churn_prob = min(0.9, 0.1 + churn_score * 0.15)
    churn = "Yes" if random.random() < churn_prob else "No"

    return [
        f"CUST{customer_id:05d}", gender, senior_citizen, partner,
        dependents, tenure, monthly_charges, contract, churn,
    ]


def main(num_rows=1500, out_path=None):
    header = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "MonthlyCharges", "Contract", "Churn"
    ]
    out_path = out_path or os.path.join(os.path.dirname(__file__), "sample_customer_churn.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(1, num_rows + 1):
            writer.writerow(generate_row(i))

    print(f"Created {out_path} with {num_rows} rows.")


if __name__ == "__main__":
    main()
