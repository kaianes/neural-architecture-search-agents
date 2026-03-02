from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.models.generate_content(
    model=os.getenv("MODEL"),
    contents="Explain neural architecture search briefly."
)

print(response.text)