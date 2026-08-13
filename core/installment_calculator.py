class InstallmentCalculator:
    
    # Available installment plans
    PLANS = {
        3:  {"months": 3,  "annual_rate": 0.10, "label": "3 months (10% interest)"},
        6:  {"months": 6,  "annual_rate": 0.15, "label": "6 months (15% interest)"},
        12: {"months": 12, "annual_rate": 0.20, "label": "12 months (20% interest)"},
    }
    
    def calculate(self, price, months=6):
        """Calculating the monthly installment with interest using the official EMI formula"""
        if months not in self.PLANS:
            return None
        
        plan = self.PLANS[months]
        monthly_rate = plan['annual_rate'] / 12
        
        # equation EMI (Equal Monthly Installment)
        # EMI = P × r × (1+r)^n / ((1+r)^n - 1)
        if monthly_rate == 0:
            emi = price / months
        else:
            emi = (price * monthly_rate * (1 + monthly_rate)**months) / \
                  ((1 + monthly_rate)**months - 1)
        
        total = emi * months
        interest_paid = total - price
        
        return {
            "original_price": price,
            "months": months,
            "monthly_payment": round(emi, 2),
            "total_amount": round(total, 2),
            "interest_paid": round(interest_paid, 2),
            "annual_rate": plan['annual_rate'],
            "plan_label": plan['label']
        }
    
    def get_all_plans(self, price: float) -> list:
        return [self.calculate(price, m) for m in self.PLANS]
