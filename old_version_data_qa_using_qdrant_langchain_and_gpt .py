# !pip install langchain
# !pip install PyPDF2
# !pip install qdrant_client
# !pip install langchain_openai
# !pip install pypdf
# !pip install openai

import os
import uuid

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain
from qdrant_client import QdrantClient,models
from qdrant_client.http.models import PointStruct
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""
os.environ["QDRANT_URL"]=""
os.environ["QDRANT_API_KEY"]=""

#create new cluseter in qdrant
record=0

connection = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
)


connection.recreate_collection(
    collection_name="second_project",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
)
print("Create collection reponse:", connection)

info = connection.get_collection(collection_name="second_project")

print("Collection info:", info)
for get_info in info:
  print(get_info)

# #funcrion for read data from pdf
def load_and_split_documents():
    pdf_path = 'baby.pdf'

    text = ""  # for storing the extracted text
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    # chunks = text_splitter.split_documents(documents)
    return chunks

def ask_question_with_context(qa, question, chat_history):

    query = ""
    result = qa({"question": question, "chat_history": chat_history})
    print("answer:", result["answer"])
    chat_history = [(query, result["answer"])]
    return chat_history

#code for convert chunks into embeddings

def get_embedding(text_chunks, model_id="text-embedding-ada-002"):
    openai = OpenAI()
    api_key = os.environ.get("OPENAI_API_KEY"),
    points = []

    for idx, chunk in enumerate(text_chunks):
        response = openai.embeddings.create(
            input=chunk,
            model=model_id
        )

        embeddings = response.data[0].embedding

        point_id = str(uuid.uuid4())  # Generate a unique ID for the point

        points.append(PointStruct(id=point_id, vector=embeddings, payload={"text": chunk}))
    return points

def get_document_store(points):
    operation_info = connection.upsert(
    collection_name="second_project",
    wait=True,
    points=points
    )
    print("Operation info:", operation_info)

def create_answer_with_context(query):
    openai = OpenAI()
    response = openai.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    embeddings = response.data[0].embedding

    search_result = connection.search(
        collection_name="second_project",
        query_vector=embeddings,
        limit=5
    )

    prompt = "Context:\n"
    for result in search_result:
        prompt += result.payload['text'] + "\n---\n"
    prompt += "Question:" + query + "\n---\n" + "Answer:"

    print("----PROMPT START----")
    print(":", prompt)
    print("----PROMPT END----")

    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        )

    return completion.choices[0].message.content

def main():
    docs = load_and_split_documents()
    embeddings = get_embedding(docs)

    doc_store = get_document_store(embeddings)

    chat_history = []
    while True:
        query = input('you: ')
        if query == 'q':
            break
        answer = create_answer_with_context(query)
        print(answer)


if __name__ == "__main__":
    main()
