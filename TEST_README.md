# Comprehensive System Testing Guide

This guide explains how to run comprehensive tests on your LLM Query Retrieval System to evaluate each component's performance.

## Test Scripts

### 1. Comprehensive System Test (`test_comprehensive_system.py`)

This script tests each component individually and shows detailed performance metrics for every step in the system.

**What it tests:**
- 🔧 Component initialization
- 🏥 Health checks for all services
- 📄 Document processing (IntelligentDocumentProcessor)
- 🧠 Embedding creation (EmbeddingManager)
- 🗄️ Vector store operations (VectorStore)
- 🔍 Dual index manager operations (DualIndexManager)
- 🤖 LLM client operations (ContextAwareLLMClient)
- 🏷️ Namespace manager operations (NamespaceManager)
- 🔄 End-to-end processing
- 🌐 API endpoint (if server is running)

**What you'll see:**
- Detailed timing for each component
- Success/failure status for each test
- Sample outputs and metadata
- Performance metrics and statistics
- Complete answers from the LLM
- Error details if any component fails

### 2. Simple API Test (`test_api_endpoint.py`)

This script tests the actual API endpoint with the exact request you provided.

**What it tests:**
- The `/hackrx/run` endpoint
- Your exact document URL and questions
- Authentication with your bearer token
- Complete response processing

**What you'll see:**
- All 10 answers from the LLM
- Processing statistics
- Service health information
- Confidence scores and reasoning
- Complete response saved to file

## How to Run the Tests

### Prerequisites

1. Make sure your virtual environment is activated:
```bash
# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

2. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

3. Make sure your environment variables are set (API keys, etc.)

### Running the Comprehensive Test

```bash
python test_comprehensive_system.py
```

This will:
- Test each component individually
- Show detailed logs for each step
- Save results to JSON and text files
- Display performance metrics

**Output files:**
- `comprehensive_test_results_YYYYMMDD_HHMMSS.json` - Complete test results
- `test_summary_YYYYMMDD_HHMMSS.txt` - Human-readable summary
- `test_results.log` - Detailed log file

### Running the API Test

First, start your server:
```bash
python -m app.main
```

Then in another terminal:
```bash
python test_api_endpoint.py
```

This will:
- Send your exact request to the API
- Display all answers from the LLM
- Show processing statistics
- Save the complete response

**Output files:**
- `api_response_YYYYMMDD_HHMMSS.json` - Complete API response

## What Each Component Does

### 1. IntelligentDocumentProcessor
- Downloads and parses PDF documents
- Extracts text, tables, and metadata
- Creates intelligent chunks based on document structure
- Identifies policy UIN, company name, policy type
- Extracts tables and converts to markdown

### 2. EmbeddingManager
- Creates embeddings for queries and document chunks
- Uses OpenAI's text-embedding-ada-002 model
- Caches embeddings for performance
- Manages embedding dimensions and model settings

### 3. VectorStore (Pinecone)
- Stores document embeddings in Pinecone
- Manages namespaces for different documents
- Provides vector similarity search
- Handles index operations and health checks

### 4. DualIndexManager
- Manages both vector and metadata databases
- Performs hybrid search combining vector and metadata
- Stores document statistics and metadata
- Handles search filters and ranking

### 5. ContextAwareLLMClient
- Processes queries with context awareness
- Uses OpenAI's GPT models for answer generation
- Provides confidence scores and reasoning
- Handles different query types and domains

### 6. NamespaceManager
- Creates unique namespaces for documents
- Manages document tracking and processing status
- Ensures document isolation and organization

## Expected Results

### Sample Output from Comprehensive Test

```
🚀 Starting comprehensive system testing...
================================================================================
🔧 Initializing system components...
✅ Components initialized in 2.34s
🏥 Testing component health checks...
✅ Health check completed in 1.23s - Status: healthy
📄 Testing document processing...
✅ Document processing completed in 15.67s
   - Created 247 chunks
   - Tables extracted: 8
   - Policy UIN: CHOTGDP23004V012223
   - Company: Cholamandalam MS General Insurance
🧠 Testing embedding creation...
✅ Embedding creation completed in 0.45s
   - Embedding dimensions: 1536
   - Model: text-embedding-ada-002
🗄️ Testing vector store operations...
✅ Vector store test completed in 0.78s
   - Health status: healthy
   - Index: hackrx-documents
🔍 Testing dual index manager...
✅ Dual index manager test completed in 0.12s
   - Initialization: True
   - Database: data/cache/metadata_index.db
🤖 Testing LLM client...
✅ LLM client test completed in 3.45s
   - Health status: healthy
   - Model: gpt-4
   - Test answer: Insurance is a financial product that provides protection...
🏷️ Testing namespace manager...
✅ Namespace manager test completed in 0.01s
   - Generated namespace: chola_ms_gen_ins_chotgdp23004v012223_20250105
🔄 Testing end-to-end processing...
✅ End-to-end test completed in 45.23s
   - Questions processed: 10
   - Successful queries: 10
   - Failed queries: 0
   - Document chunks: 247
   - Tables extracted: 8
   - Q1: A grace period of thirty days is provided for premium payment... (confidence: 0.92)
   - Q2: There is a waiting period of thirty-six (36) months... (confidence: 0.89)
   - Q3: Yes, the policy covers maternity expenses... (confidence: 0.94)
📊 Calculating performance metrics...
📊 Performance metrics calculated:
   - Total test time: 69.25s
   - Component success rate: 100.0%
   - End-to-end success: True
   - API endpoint success: True
================================================================================
🎉 Comprehensive testing completed!
💾 Test results saved to: comprehensive_test_results_20250105_143022.json
📝 Test summary saved to: test_summary_20250105_143022.txt
```

### Sample Output from API Test

```
🚀 Starting API endpoint test...
URL: http://localhost:8000/hackrx/run
Documents: 1
Questions: 10
📤 Sending request...
📥 Response received in 42.15s
HTTP Status: 200
Content-Type: application/json
Response keys: ['answers', 'processing_stats', 'document_metadata', 'service_stats', 'request_id']
✅ Received 10 answers

📋 Question 1: What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?
💡 Answer: A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.
🎯 Confidence: 0.92

📋 Question 2: What is the waiting period for pre-existing diseases (PED) to be covered?
💡 Answer: There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.
🎯 Confidence: 0.89

[... more answers ...]

📊 Processing Statistics:
   - Total questions: 10
   - Successful queries: 10
   - Failed queries: 0
   - Processing time: 42.15s
   - Document chunks: 247
   - Tables extracted: 8

🔧 Service Statistics:
   - Embedding service: healthy
   - LLM service: healthy
   - Vector store: healthy

💾 Full response saved to: api_response_20250105_143022.json
✅ API test completed successfully!
🎉 All tests passed!
```

## Troubleshooting

### Common Issues

1. **Connection errors**: Make sure your server is running on `http://localhost:8000`
2. **Authentication errors**: Check your bearer token in settings
3. **API key errors**: Ensure your OpenAI and Pinecone API keys are set
4. **Import errors**: Make sure you're in the correct directory and virtual environment is activated

### Performance Expectations

- **Component initialization**: 1-3 seconds
- **Document processing**: 10-30 seconds (depends on document size)
- **Embedding creation**: 0.5-2 seconds per query
- **LLM processing**: 2-5 seconds per query
- **End-to-end processing**: 30-60 seconds for 10 questions
- **API response**: 30-60 seconds total

### Success Criteria

- All components should show "success" status
- Component success rate should be 100%
- End-to-end processing should complete successfully
- All questions should receive answers with confidence > 0.7
- Processing time should be reasonable (< 60 seconds for 10 questions)

## File Structure

After running tests, you'll have:

```
llm-query-retrieval-system/
├── test_comprehensive_system.py          # Comprehensive test script
├── test_api_endpoint.py                  # API test script
├── TEST_README.md                        # This file
├── comprehensive_test_results_*.json     # Complete test results
├── test_summary_*.txt                    # Human-readable summary
├── api_response_*.json                   # API response data
└── test_results.log                      # Detailed logs
```

This comprehensive testing approach will give you complete visibility into each component's performance and the overall system behavior. 