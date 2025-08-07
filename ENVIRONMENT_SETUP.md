# 🔐 Environment Variables Setup Guide

## **Local Development (.env file)**

Create a `.env` file in your project root with:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here

# Pinecone Vector Database Configuration
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=your-pinecone-environment
PINECONE_INDEX_NAME=your-pinecone-index-name

# Application Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Optional: Custom Settings
EMBEDDING_BATCH_SIZE=50
MAX_SEARCH_RESULTS=30
MAX_CONTEXT_LENGTH=60000
```

## **Vercel Deployment (Environment Variables)**

In Vercel dashboard, add these environment variables:

### **Required Variables:**
```
OPENAI_API_KEY = sk-your-openai-api-key-here
PINECONE_API_KEY = your-pinecone-api-key-here
PINECONE_ENVIRONMENT = your-pinecone-environment
PINECONE_INDEX_NAME = your-pinecone-index-name
```

### **Optional Variables:**
```
ENVIRONMENT = production
DEBUG = false
LOG_LEVEL = INFO
EMBEDDING_BATCH_SIZE = 50
MAX_SEARCH_RESULTS = 30
MAX_CONTEXT_LENGTH = 60000
```

## **How to Get API Keys:**

### **1. OpenAI API Key:**
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up/Login
3. Go to "API Keys"
4. Create new secret key
5. Copy the key (starts with `sk-`)

### **2. Pinecone API Key:**
1. Go to [pinecone.io](https://pinecone.io)
2. Sign up/Login
3. Go to "API Keys"
4. Create new API key
5. Note your environment (e.g., `us-east-1-aws`)
6. Create an index and note the index name

## **Security Notes:**
- ✅ `.env` file is in `.gitignore` (not uploaded to GitHub)
- ✅ Environment variables in Vercel are encrypted
- ✅ Never commit API keys to git
- ✅ Use different keys for development and production

## **Testing Environment Variables:**
```bash
# Check if variables are loaded
python -c "import os; print('OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

