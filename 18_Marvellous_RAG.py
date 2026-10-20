from sentence_transformers import SentenceTransformer,util
import chromadb
from ollama import chat

# load pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# connect to existing chromsDB
client = chromadb.PersistentClient(path="./MarvellousDB")

collection = client.get_collection(name="Marvellous_knowledge")

# accept question
querry  = input("Enter your Question")


# Generate querry embedding
querry_embedding = model.encode(querry,convert_to_tensor=True)

# Retrive top 2 resutls
results = collection.query(
    query_embeddings = [querry_embedding.tolist()],
    n_results=2
)

# create context from retrived documents 
documents = results["documents"][0]

context = "\n".join(documents)

print("Retrived context :")
print(context)
