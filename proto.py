"""
title: Espoo 
author: Silas Rech
date: 27/04/2025
version: 0.1
license: MIT
description: A pipeline for retrieving relevant information from a knowledge graph from Espoo.
requirements: typing, pydantic, langchain, langchain-community, langchain_groq, numpy, pandas, langchain-huggingface, faiss-cpu
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, BaseRetriever  # Updated import for BaseRetriever
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

from pydantic import Field
from typing import List
from langchain.prompts import PromptTemplate

import numpy as np
import pandas as pd
import re


class Pipeline:
    class Valves(BaseModel):
        pass

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "pipeline_example"

        # The name of the pipeline.
        self.name = "KG RAG ESPOO"
        self.vectorstore = None
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,

        )
        self.chunks = None
        self.base_retriever = None

        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")

        try:
            df_graph = pd.read_csv('/app/pipelines/kg.csv')
            df_graph = df_graph.dropna()
            # Create document list from dataset
            documents = [(df_graph.text.iloc[num], df_graph.url.iloc[num]) for num in range(df_graph.shape[0])]
            print('Successfully loaded graph.')
        except:
            print('Could not load KG - not found.')
            documents = ''

        # Initialize embedding model
        embedding_model_name = "multi-qa-mpnet-base-cos-v1"
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        self.chunks = create_chunks(documents)
        vectorstore = FAISS.from_documents(self.chunks, embeddings)
        self.base_retriever = vectorstore.as_retriever(search_type="mmr",
                                                       search_kwargs={'k': 50, 'fetch_k': 200, 'lambda_mult': 0.25})


    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    async def inlet(self, body: dict, user: dict) -> dict:
        # This function is called before the OpenAI API request is made. You can modify the form data before it is sent to the OpenAI API.
        print(f"inlet:{__name__}")

        print(body)
        print(user)


        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        # This function is called after the OpenAI API response is completed. You can modify the messages after they are received from the OpenAI API.
        print(f"outlet:{__name__}")

        print(body)
        print(user)

        return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.

        try:
            filtered_chain, filtered_retriever = get_retreival_chain(user_message, self.base_retriever, self.chunks,
                                                                     self.llm)
            response = filtered_chain.run(user_message)
            # Extract links of sources used in the response
            relevant_docs = filtered_retriever.get_relevant_documents(user_message)
            source_links = [doc.metadata["link"] for doc in relevant_docs]
            response += f"These are the sources for my answer: {source_links}"
        except Exception as e:
            response = f'Could not complete knowledge search, ERROR: {e}'
            print("ERROR retrieving")

        return response


class FilteredRetriever(BaseRetriever):
    retriever: BaseRetriever = Field(...)
    original_documents: List[Document] = Field(...)
    index_filter: List[int] = Field(...)
    embedding_model_name: str = Field("all-MiniLM-L6-v2")
    top_k_initial: int = Field(50)  # Number of top documents to initially retrieve
    k: int = Field(10)  # Final number of filtered documents to return

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Step 1: Retrieve top `top_k_initial` documents based on relevance
        initial_relevant_docs = self.retriever.get_relevant_documents(query)[:self.top_k_initial]
        # print(len(initial_relevant_docs))

        # Step 2: Filter these documents by `index_filter`
        filtered_docs = [
            doc for doc in initial_relevant_docs if doc.metadata["original_index"] in self.index_filter
        ]

        # Step 3: Return top `k` documents after filtering
        # print(len(filtered_docs))
        return filtered_docs[:self.k]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


def create_query(row):
    answer_length = ' The answer should be maximum ' + str(np.mean([len(row['answer_1']), len(row['answer_2']), len(row['answer_3'])])) + ' characters.'
    query=  row['question'] + answer_length
    return query


# Step 1: Define document chunking function
def create_chunks(documents, max_chunk_size=500):
    chunks = []
    for i, (text, link) in enumerate(documents):
        # Split document text into smaller chunks if necessary, and include link in metadata
        for j in range(0, len(text), max_chunk_size):
            chunk_text = text[j:j + max_chunk_size]
            chunks.append(Document(page_content=chunk_text, metadata={"original_index": i, "link": link}))
    return chunks


def is_relevant(node, output_list):
    for item in output_list:
        # Create word-based patterns for both node and output item
        node_pattern = r"\b" + r"\b.*\b".join(re.escape(word) for word in node.split()) + r"\b"
        item_pattern = r"\b" + r"\b.*\b".join(re.escape(word) for word in item.split()) + r"\b"

        # Check for partial or full match in either direction
        if re.search(node_pattern, item, re.IGNORECASE) or re.search(item_pattern, node, re.IGNORECASE):
            return True
    return False


def find_relevant_edges(graph, output_list):
    relevant_edges = []

    # Iterate over edges in the graph
    for edge in graph._graph.edges(data=True):
        node1, node2 = edge[0], edge[1]

        # Check if both nodes of the edge are relevant
        if is_relevant(node1, output_list) and is_relevant(node2, output_list):
            relevant_edges.append(edge)

        # if node1 in output_list and node2 in output_list:
        #    relevant_edges.append(edge)

    return relevant_edges


def get_relevant_docindex(df, relevant_edges):
    relevant_docindexs = []

    for edge in relevant_edges:
        node1, node2 = edge[0], edge[1]

        # Check if both nodes of the edge are relevant, appned the df text and url to the relevant_documents list
        if df[(df.ent1 == node1) & (df.ent2 == node2)].shape[0] > 0:
            rel_index = df[(df.ent1 == node1) & (df.ent2 == node2)]
            relevant_docindexs += rel_index.index.tolist()
        elif df[(df.ent1 == node2) & (df.ent2 == node1)].shape[0] > 0:
            rel_index = df[(df.ent1 == node2) & (df.ent2 == node1)]
            relevant_docindexs += rel_index.index.tolist()
    return relevant_docindexs

def get_filtered_index(question, df_graph):
    print(question)

    relevant_doc_indexes = []
    for t in question.split('\n'):
        t = t.lower()
        t = t[1:-1]
        list_output = (t.split(', '))
        if len(list_output) == 3:
            print(list_output)
            relevant_edges = find_relevant_edges(df_graph, list_output)
            relevant_doc_indexes += get_relevant_docindex(df_graph, relevant_edges)

    doc_index = list(set(relevant_doc_indexes))
    return doc_index


def get_retreival_chain(query, base_retriever, chunks, llm):
    # Indices of documents to keep
    index_filter = get_filtered_index(query)  # Only include filtered documents
    # print(index_filter)

    # Create a filtered retriever with index filtering
    # filtered_retriever = FilteredRetriever(retriever=base_retriever, index_filter=index_filter)
    filtered_retriever = FilteredRetriever(
        retriever=base_retriever,
        original_documents=chunks,
        index_filter=index_filter,
        top_k_initial=50,  # Set the initial retrieval to 20
        k=10  # Final filtered top 10 documents
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],

        template="""You are an assistant for question-answering tasks.
            Use the following documents to answer the question about migration in Finland.
            Be consise as much as possible.
            Do not include phrases about provided documents in the answer.

            Question: {question}
            Documents: {context}
            Answer:
            """)

    # Create the retrieval chain with the filtered retriever
    retrieval_chain = RetrievalQA.from_llm(llm=llm, retriever=filtered_retriever, prompt=prompt_template)

    return retrieval_chain, filtered_retriever


