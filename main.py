# Vercel entry point - imports the FastAPI app from app/main.py
from app.main import app

# This is the app that Vercel will use
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
