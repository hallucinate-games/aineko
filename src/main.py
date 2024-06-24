import chromadb

from aineko import AinekoEmbeddingFunction

# Use this for persistent client
#chroma_client = chromadb.PersistentClient(path="../demo-db")
chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(name="aineko-demo", embedding_function=AinekoEmbeddingFunction())
# Uncomment these for persistent client
#chroma_client.delete_collection("aineko-demo")
#collection = chroma_client.get_or_create_collection(name="aineko-demo", embedding_function=AinekoEmbeddingFunction())

collection.add(
        ids=['id1', 'id2'],
        documents=[
            "Pineapples are so delicious it's worth the effort to eat them. Just don't eat too much or your tongue will go numb.",
            "Pizza is not an actual food, it's a framework for producing new instances of delicious flavor combos on flatbread."
        ]
)

query1 = "Fire baked dough with tomato sauce, cheese, and toppings."
query2 = "Fruit with yellow flesh and a spiky exterior."
query_results = collection.query(
        query_texts=[query1, query2],
        n_results=2,
)

from pprint import pprint
pprint(query_results)

