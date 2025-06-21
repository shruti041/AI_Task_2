# 3. Build the RAG Pipeline
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Embed Quotes
embedder = SentenceTransformer("C:\\Users\\INSPIRON\\Desktop\\AI\\quote_model") 

# Load cleaned CSV
df = pd.read_csv("cleaned_english_quotes.csv")
df = df.dropna(subset=["quote"])  # drop rows with missing quote
quotes = df["quote"].tolist()

metadata = [
    f'"{row.quote}" - {row.author} [Tags: {row.tags}]'
    if "author" in df.columns and "tags" in df.columns
    else row.quote
    for _, row in df.iterrows()
]

quote_embeddings = embedder.encode(quotes, show_progress_bar=True, convert_to_numpy=True)

# Index with FAISS
dimension = quote_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(quote_embeddings)

# Load open source model 
# Use a public, CPU-friendly model
llm_model = "google/flan-t5-small" 
tokenizer = AutoTokenizer.from_pretrained(llm_model)
model = AutoModelForSeq2SeqLM.from_pretrained(llm_model)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# RAG query function
def rag_query(query, top_k=5):
    # Encode user query using the fine-tuned model
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    
    # Retrieve top-K relevant quotes
    distances, indices = index.search(query_embedding, top_k)
    retrieved_context = [metadata[i] for i in indices[0]]
    
    # Build prompt for LLM
    context = "\n".join([f"{i+1}. {quote}" for i, quote in enumerate(retrieved_context)])
    prompt = f"""You are an assistant who answers using famous quotes. Use the following context to respond to the user's question.

Quotes:
{context}

User: {query}
Answer:"""

    # Generate answer
    result = pipe(prompt, max_new_tokens=200)[0]['generated_text']
    return result.strip(), retrieved_context

response, retrieved = rag_query("quotes about hope by Oscar Wilde")

print("Retrieved Context:")
for i, quote in enumerate(retrieved, 1):
    print(f"{i}. {quote}")

print("\n Answer:")
print(response)

# 4. RAG Evaluation

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer

# Load embedding model (can be same as `quote_model`)
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate_locally(responses, ground_truths):
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    from rouge_score import rouge_scorer

    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    rouge_scores = []
    cosine_scores = []

    for gen, gt in zip(responses, ground_truths):
        # ROUGE
        score = scorer.score(gt, gen)
        rouge_scores.append(score["rougeL"].fmeasure)

        # Cosine Similarity
        gen_vec = similarity_model.encode([gen])[0]
        gt_vec = similarity_model.encode([gt])[0]
        cosine = cosine_similarity([gen_vec], [gt_vec])[0][0]
        cosine_scores.append(cosine)

    return {
        "avg_rougeL": sum(rouge_scores) / len(rouge_scores),
        "avg_cosine_similarity": sum(cosine_scores) / len(cosine_scores)
    }

# Sample queries and answers
queries = [
    "quote about hope",
    "Oscar Wilde quote about stars",
    "quote about failure"
]

# RAG pipeline answers
answers = [
    "Hope is the thing with feathers that perches in the soul.",
    "We are all in the gutter, but some of us are looking at the stars.",
    "Failure is simply the opportunity to begin again, this time more intelligently."
]

# Ground truth expected answers
ground_truths = [
    "Hope is the thing with feathers that perches in the soul.",
    "We are all in the gutter, but some of us are looking at the stars.",
    "Failure is simply the opportunity to begin again, this time more intelligently."
]

# evaluation
results = evaluate_locally(answers, ground_truths)
print(results)

