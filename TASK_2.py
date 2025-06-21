# 1. Data Preparation
from datasets import load_dataset
import pandas as pd
import re
from sentence_transformers import InputExample

print("Loading dataset from Hugging Face...")
dataset = load_dataset("Abirate/english_quotes")

df = pd.DataFrame(dataset['train'])
df.dropna(subset=['quote'], inplace=True) # we drop rows with missing quotes to avoid errors and ensure our model trains only on meaningful, non-empty examples.

print("Sample data:")
print(df.head())
print("\nColumns:", df.columns)

# lowercasing
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\'\":.,-]', '', text)
    return text.strip()

df['quote'] = df['quote'].apply(clean)
df['author'] = df['author'].fillna("unknown").apply(str.lower)

# tokenization
df['tokens'] = df['quote'].apply(lambda x: x.split())

df.to_csv("cleaned_english_quotes.csv", index=False)

df['context'] = df.apply(lambda row: f"{row['quote']} - {row['author']}", axis=1)

train_samples = [InputExample(texts=[row['quote'], row['context']]) for _, row in df.iterrows()]

# 2. Model Fine-Tuning
from sentence_transformers import SentenceTransformer, losses, models
from torch.utils.data import DataLoader

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')  

# Create DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)      #Efficiently feed batches of training data to the model during fine-tuning.

# Define loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,            # Increase for better results
    warmup_steps=100,
    show_progress_bar=True
)

# Save the model
model.save("quote_model")
print("Fine-tuned model saved to: quote_model")

