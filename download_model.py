from sentence_transformers import SentenceTransformer

# You can use:
# model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('intfloat/e5-small-v2')
model = SentenceTransformer('all-mpnet-base-v2')


model.save('./local_model')
print("✅ Model downloaded and saved to ./local_model")