import ollama

class RiskAssistant:
    def __init__(self, model="mistral"):
        self.model = model

    def explain_credit(self, summary: dict) -> str:
        prompt = f"""
        You are a risk analyst.

        Explain this credit portfolio summary:
        {summary}

        Focus on risk insights, not technical jargon.
        """

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]

    def explain_fraud(self, alerts_df):
        prompt = f"""
        Explain the fraud alerts:
        {alerts_df.head().to_string()}

        What patterns do you see?
        """

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]