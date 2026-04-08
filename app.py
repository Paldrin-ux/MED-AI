import os
from dotenv import load_dotenv

# Load .env before importing app factory so all modules see env vars.
load_dotenv()
from app import create_app
app = create_app(os.environ.get("FLASK_ENV", "development"))

import json

@app.template_filter('fromjson')
def fromjson_filter(value):
    try:
        return list(json.loads(value).items())
    except Exception:
        return []


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=app.config.get("DEBUG", False))