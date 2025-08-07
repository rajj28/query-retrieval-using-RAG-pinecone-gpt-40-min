@echo off
echo 🚀 HackRx Vercel Deployment Script
echo ==================================

echo.
echo 📋 Prerequisites Check:
echo.

REM Check if git is initialized
if not exist ".git" (
    echo ❌ Git not initialized. Initializing now...
    git init
    git add .
    git commit -m "Initial commit - HackRx LLM System"
    echo ✅ Git initialized
) else (
    echo ✅ Git already initialized
)

echo.
echo 📤 Pushing to GitHub...
git add .
git commit -m "Deploy to Vercel - %date% %time%"
git push origin main

echo.
echo 🎯 Vercel Deployment Steps:
echo.
echo 1. 🌐 Go to https://vercel.com
echo 2. 👤 Sign up with GitHub
echo 3. ➕ Click "New Project"
echo 4. 📁 Import your hackrx-llm-system repository
echo 5. ⚙️ Configure settings:
echo    - Framework Preset: Other
echo    - Build Command: pip install -r requirements.txt
echo    - Output Directory: ./
echo 6. 🔑 Add Environment Variables:
echo    - OPENAI_API_KEY = your-openai-api-key
echo    - PINECONE_API_KEY = your-pinecone-api-key
echo    - PINECONE_ENVIRONMENT = your-pinecone-environment
echo    - PINECONE_INDEX_NAME = your-pinecone-index-name
echo    - ENVIRONMENT = production
echo    - DEBUG = false
echo    - LOG_LEVEL = INFO
echo 7. 🚀 Click "Deploy"
echo.
echo 🎉 Your API will be live at:
echo    https://hackrx-llm-system.vercel.app/api/v1/hackrx/run
echo.
echo 🧪 Test with:
echo    curl https://hackrx-llm-system.vercel.app/api/v1/hackrx/health
echo.
echo 📚 For detailed guide, see: VERCEL_DEPLOYMENT_GUIDE.md
echo.
pause

