# simple currency converter tool
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
import os


def currency_converter(amount, from_currency, to_currency):
    rates = {"USD": 1.0, "EUR": 0.92, "INR": 83.0, "JPY": 146.0}

    if from_currency not in rates or to_currency not in rates:
        raise ValueError("Currency not supported")

    # Convert to USD first, then to target
    usd_amount = amount / rates[from_currency]
    converted = usd_amount * rates[to_currency]
    return round(converted, 2)


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

client.models.generate_content()
