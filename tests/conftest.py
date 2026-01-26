"""
Test configuration for recsys-universe.
"""
import os
import sys

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set test environment variables
os.environ["OPENAI_API_KEY"] = "test_api_key"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
os.environ["RECSYS_DEBUG"] = "false"
