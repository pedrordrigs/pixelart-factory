
from google import genai
import os

api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

try:
    print("Checking generate_images...")
    if hasattr(client.models, 'generate_images'):
        print("client.models.generate_images exists")
    else:
        print("client.models.generate_images DOES NOT exist")
        
    print("Checking generate_image...")
    if hasattr(client.models, 'generate_image'):
        print("client.models.generate_image exists")
        
except Exception as e:
    print(e)

