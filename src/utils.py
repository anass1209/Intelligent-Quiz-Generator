import os
from dotenv import load_dotenv

# --- Utility Functions ---
def load_api_keys():
    """
    Loads essential API keys and configuration variables from environment variables.

    This function attempts to load the following keys from a .env file (or the system environment):
    - GOOGLE_API_KEY: Required for accessing Google Generative AI models (Embeddings, Chat).
    - TAVILY_API_KEY: Optional key for enabling Tavily web search functionality.
    - FLASK_SECRET_KEY: Required for Flask session security and other cryptographic functions.

    It prioritizes variables found in a `.env` file located in the project's root
    or parent directories.

    Returns:
        tuple[str, str | None, str]: A tuple containing the loaded keys:
                                     (google_api_key, tavily_api_key, flask_secret_key).
                                     tavily_api_key will be None if not found.

    Raises:
        ValueError: If the mandatory GOOGLE_API_KEY or FLASK_SECRET_KEY is not defined
                    in the environment variables after attempting to load the .env file.
    """
    # Load environment variables from a .env file if it exists.
    load_dotenv()

    # Retrieve keys from the environment. os.getenv returns None if the key doesn't exist.
    google_api_key = os.getenv("GOOGLE_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY") # Optional key
    flask_secret_key = os.getenv("FLASK_SECRET_KEY") # Mandatory for Flask security

    # --- Validation ---
    if not google_api_key:
        raise ValueError("Google API Key (GOOGLE_API_KEY) is not defined in the environment or .env file.")

    if not flask_secret_key:
        raise ValueError("Flask Secret Key (FLASK_SECRET_KEY) is not defined in the environment or .env file.")

    # Handle the optional Tavily key. If missing, print a warning instead of raising an error,
    if not tavily_api_key:
        print("Warning: Tavily API Key (TAVILY_API_KEY) is not defined. Web search functionality will be disabled.")

    # Return the loaded keys.
    return google_api_key, tavily_api_key, flask_secret_key