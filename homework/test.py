import requests
from minsearch import Index
from gitsource import GithubRepositoryDataReader
from dotenv import load_dotenv
load_dotenv()
from gitsource import chunk_documents

from openai import OpenAI
client = OpenAI()


reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()


documents = []

for file in files:
    doc = file.parse()
    documents.append(doc)

#print("Number of documents:", len(documents))
#print("Sample document:", documents[0])

chunked_documents = chunk_documents(
    documents,
    size=2000,
    step=1000
)

print(f"Chunks: {len(chunked_documents)}")