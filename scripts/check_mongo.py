import os
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi

load_dotenv()

uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB", "nyayamitra")

try:
    client = MongoClient(uri, tlsCAFile=certifi.where())
    client.admin.command('ping')
    db = client[db_name]
    count = db.feedback.count_documents({})
    print(f"Connected to MongoDB. Database: {db_name}")
    print(f"Feedback entries so far: {count}")
except Exception as e:
    print("Could not connect to MongoDB:", e)