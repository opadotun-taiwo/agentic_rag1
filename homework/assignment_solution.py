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

def build_context(documents):
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )
    index.fit(documents)
    return index

question = "How does the agentic loop keep calling the model until it stops?"

index = build_context(chunked_documents)
#index_results = index.search(question)

def search(question):
    search_results = index.search(question, 
            boost_dict={"content": 2.0},
            num_results=3
            )
    return search_results

def build_context(search_results):
    lines = []

    for doc in search_results:
        lines.append(doc["content"])
        lines.append("")

    return "\n".join(lines)
    
context = build_context(search(question))

USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""

#print(USER_PROMPT_TEMPLATE)

def build_prompt(question, search_results):
    context = build_context(search_results)
    prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)
    return prompt.strip()

#print(build_prompt(question, search(question)))

INSTRUCTIONS = """Your task is to answer questions from the contents
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."""


def llm(user_prompt, instructions, model="gpt-5.4-mini"):
    message_history = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt}
    ]

    response = client.responses.create(
            model=model,
            input=message_history
        )
    
    output_text = response.output_text
    prompt_tokens = response.usage.input_tokens

    return output_text, prompt_tokens


def rag(query, model="gpt-5.4-mini"):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer, token = llm(prompt, INSTRUCTIONS, model=model)
    return answer, token

answer, token = rag(question)


print("Number of documents:", len(documents))
print(f"Chunks: {len(chunked_documents)}")
print("Answer:", answer)
print("Prompt Tokens:", token)
