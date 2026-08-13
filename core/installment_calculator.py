class InstallmentCalculator:
    """
    Handles financial business logic and installment payment plan calculations.
    """
    def __init__(self, default_interest_rate: float = 0.05):
        self.default_interest_rate = default_interest_rate

    def calculate_installment(self, price: float, months: int = 12, interest_rate: float = None) -> dict:
        """
        Calculates monthly payment, total interest, and total payable amount.
        """
        if price <= 0:
            return {
                "monthly_installment": 0.0,
                "total_amount": 0.0,
                "interest_amount": 0.0,
                "months": months,
                "error": "Price must be greater than zero."
            }
        
        rate = interest_rate if interest_rate is not None else self.default_interest_rate
        total_amount = price * (1 + rate)
        monthly_installment = total_amount / months if months > 0 else total_amount

        return {
            "original_price": round(price, 2),
            "months": months,
            "interest_rate_percent": round(rate * 100, 2),
            "interest_amount": round(total_amount - price, 2),
            "total_amount": round(total_amount, 2),
            "monthly_installment": round(monthly_installment, 2)
        }

    def get_all_plans(self, price: float) -> list:
        """
        Generates standard 3, 6, and 12-month installment breakdowns for a product price.
        """
        plan_terms = [
            (3, 0.10),   # 3 months, 10% interest
            (6, 0.15),   # 6 months, 15% interest
            (12, 0.20)   # 12 months, 20% interest
        ]
        return [self.calculate_installment(price, months=m, interest_rate=r) for m, r in plan_terms]
