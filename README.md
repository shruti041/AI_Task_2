# TASK – 2
## RAG-Based Semantic Quote Retrieval and Structured QA with Model Training
## 1. Data Preparation
1. **Dataset:** Uses the HuggingFace dataset "Abirate/english_quotes", which includes quotes, authors, and tags. <br>
2. **Cleaning:**
   - Lowercasing all text ensures consistency during embedding.
   - Regex is used to remove special characters that may affect model learning.
   - Missing author values are filled with "unknown".
3. **Tokenization:** Quotes are split into word tokens to allow further analysis or NLP tasks. <br>
4. **Context Construction:** For each quote, a combined quote - author string is built. This serves as input for contrastive training.
## 2. Model Fine-Tuning
### Sentence-BERT (SBERT)
- SBERT converts sentences into fixed-size dense vectors that preserve semantic meaning.
- Pretrained model used: all-MiniLM-L6-v2 (compact, efficient, 384-dim vectors).
### Fine-Tuning with MultipleNegativesRankingLoss
1. This contrastive loss improves how well the model embeds similar sentence pairs.
2. **Training Input:** Pairs of quote and quote-context ([quote, quote - author]).
3. The model is trained to:
   - Pull similar sentence pairs closer in vector space.
   - Push dissimilar ones apart.
4. This makes future semantic search more accurate.
## 3. Build the RAG Pipeline
1. **Query Embedding:** The user query is converted into a dense vector.
2. **Quote Retrieval:** FAISS searches for the top-k quotes closest to the query vector.
3. **Prompt Construction:** The retrieved quotes are combined into a prompt.
4. **Generation:** The prompt is passed to the FLAN-T5 model to generate a relevant, quote-based response.
## 4. RAG Evaluation
1. **ROUGE-L Score**
   - Measures the longest common subsequence between generated and ground truth text.
   - Captures fluency and semantic overlap.
   - **output:** 'avg_rougeL': 1.0
2. **Cosine Similarity**
   - Measures angle similarity between two vectors (generated vs. ground truth).
   - Closer to 1.0 = higher semantic similarity.
   - **output:** 'avg_cosine_similarity': np.float32(1.0)
Both metrics help evaluate whether the generated responses are meaningful and contextually correct.
## 5. Streamlit Application
### Features:
1. Allows natural language input (e.g., “quotes about resilience by women”).
2. Top-K slider lets users control how many quotes are retrieved for generation.
3. Displays:
   - Retrieved quote context.
   - Generated quote-based answer.
   - JSON-style output for structured results.
### Purpose:
Makes the RAG pipeline interactive and usable for general audiences without needing code execution.
## Challenges:
1. Low Retrieval Relevance
   - Solved by fine-tuning Sentence-BERT with contrastive loss.
2. LLM Generating Irrelevant Responses
   - Controlled by crafting clear, context-rich prompts.
3. Missing or Incomplete Data
   - Handled by cleaning text and filling missing authors as "unknown".
4. Slow Embedding and Indexing
   - Optimized with FAISS and batch encoding.
5. Limited Evaluation Coverage
   - Current metrics (ROUGE, cosine) are based on small test cases; needs scaling.
6. Model Resource Constraints
   - Chose FLAN-T5-small for CPU compatibility.


