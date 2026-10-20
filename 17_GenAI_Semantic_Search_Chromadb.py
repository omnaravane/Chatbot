from sentence_transformers import SentenceTransformer,util
import chromadb

# load pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# knowledge base
documents = [
    "Python is used for machine learning",
    "Java is used for enterprise Application",
    "SQL is used to manage Databases",
    "HTML is used to create webpages",
    "Tensorflow is used  for deep learning",
    "JavaScript is used to build interactive websites",
    "C++ is used for game development",
    "React is used to build user interfaces",
    "Docker is used to containerize applications",
    "Kubernetes is used for container orchestration",
    "Git is used to track source code changes",
    "Swift is used for iOS app development",
    "Kotlin is used for Android application development",
    "Pandas is used for data manipulation and analysis",
    "PyTorch is used for building neural networks",
    "Redis is used for caching data in memory",
    "Linux is used to manage server operating systems",
    "Node.js is used to execute JavaScript on the server",
    "Ansible is used for IT configuration automation",
    "MongoDB is used to store unstructured data"

]

# Generate Document embedding
document_embedding = model.encode(documents,convert_to_tensor=True)

#create chromadb client
client = chromadb.PersistentClient(path="./MarvellousDB")

# create collection
collection = client.get_or_create_collection(
    name="Marvellous_knowledge",
    configuration={"hnsw":{"space":"cosine"}},

) #hnsw : Hierarchical navigable small world

# store documents and embeddings
collection.upsert(
    ids = ["doc1","doc2","doc3","doc4","doc5"],
    documents = documents,
    embeddings=document_embedding.tolist()
)

print("Total documents :",collection.count())


# accept question
querry  = input("Enter your Question")


# Generate querry embedding
querry_embedding = model.encode(querry,convert_to_tensor=True)

# calculate cosine similarity
scores = util.cos_sim(querry_embedding,document_embedding)[0]

# find top 2 matching documents
Top_result = scores.topk(2)

print("Top 2 relavant documents :")

for index in Top_result.indices:
    print(documents[index.item()])
    