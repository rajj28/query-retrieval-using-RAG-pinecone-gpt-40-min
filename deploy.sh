#!/bin/bash

# 🚀 HackRx Deployment Script
# This script helps you deploy HackRx to free hosting platforms

echo "🚀 HackRx LLM Query Retrieval System - Deployment Script"
echo "========================================================"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git repository not found. Please initialize git first:"
    echo "   git init"
    echo "   git add ."
    echo "   git commit -m 'Initial commit'"
    echo "   git remote add origin <your-github-repo-url>"
    exit 1
fi

# Check if remote is set
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "❌ Git remote not set. Please add your GitHub repository:"
    echo "   git remote add origin <your-github-repo-url>"
    exit 1
fi

echo "✅ Git repository found"
echo ""

# Push to GitHub
echo "📤 Pushing to GitHub..."
git add .
git commit -m "Deploy HackRx LLM System - $(date)"
git push origin main

echo "✅ Code pushed to GitHub"
echo ""

# Display deployment options
echo "🎯 Choose your deployment platform:"
echo ""
echo "1. 🚀 Render (Recommended - Free & Easy)"
echo "   - Go to: https://render.com"
echo "   - Sign up with GitHub"
echo "   - Click 'New +' → 'Web Service'"
echo "   - Connect your repository"
echo "   - Your URL: https://hackrx-llm-system.onrender.com/api/v1/hackrx/run"
echo ""

echo "2. ⚡ Fly.io (Very Fast - Free)"
echo "   - Install CLI: curl -L https://fly.io/install.sh | sh"
echo "   - Run: fly launch"
echo "   - Set secrets: fly secrets set OPENAI_API_KEY='your-key'"
echo "   - Deploy: fly deploy"
echo "   - Your URL: https://hackrx-llm-system.fly.dev/api/v1/hackrx/run"
echo ""

echo "3. 🚂 Railway (Free Tier)"
echo "   - Go to: https://railway.app"
echo "   - Sign up with GitHub"
echo "   - Click 'New Project' → 'Deploy from GitHub repo'"
echo "   - Your URL: https://hackrx-llm-system-production.up.railway.app/api/v1/hackrx/run"
echo ""

echo "4. ⚡ Vercel (Very Fast - Free)"
echo "   - Go to: https://vercel.com"
echo "   - Sign up with GitHub"
echo "   - Click 'New Project'"
echo "   - Import your repository"
echo "   - Your URL: https://hackrx-llm-system.vercel.app/api/v1/hackrx/run"
echo ""

echo "🔧 Required Environment Variables:"
echo "   OPENAI_API_KEY=your-openai-api-key"
echo "   PINECONE_API_KEY=your-pinecone-api-key"
echo "   PINECONE_ENVIRONMENT=your-pinecone-environment"
echo "   PINECONE_INDEX_NAME=your-pinecone-index-name"
echo ""

echo "🧪 Test your API once deployed:"
echo "   curl https://your-app-url/api/v1/hackrx/health"
echo ""

echo "📚 For detailed instructions, see: DEPLOYMENT_GUIDE.md"
echo ""

echo "🎉 Happy deploying! Your HackRx API will be live soon!"

