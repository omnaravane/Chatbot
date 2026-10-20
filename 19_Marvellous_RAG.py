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

# create prompt for LLM

prompt = f"""
    Answer the question based on the given context and more in 4 5 lines    .

    Context : {context}.

    Question : {querry}

    
"""                                                         # using only the given context.
# If the Answer is not available,
#    say "Information not available".




# Send the context and question to ollama(llama)

respone = chat(
    model="llama3.2:1b",
    messages=[
        {
            "role" : "user",
            "content" : prompt 
        }
    ]
)

answer = respone.message.content

print("Marvellous GenAI : RAG answer")
print(answer)
