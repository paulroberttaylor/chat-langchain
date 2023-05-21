import datetime
import pickle
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms.fake import FakeListLLM
from langchain.llms import GPT4All

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

model_name = 'gpt-3.5-turbo'
temperature = 0.1
llm = ChatOpenAI(model_name=model_name, temperature=temperature, verbose=True, openai_api_key="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")

# embeddings = OpenAIEmbeddings()
# new_db = FAISS.load_local("vectorstore.pkl", embeddings)

# query = "How do I use FAISS vector stores?"
# docs = new_db.similarity_search(query)

from langchain.vectorstores.base import VectorStoreRetriever


with open("vectorstore.pkl", "rb") as f:
    global vectorstore
    vectorstore = pickle.load(f)
    print(vectorstore.index)

retriever = VectorStoreRetriever(vectorstore=FAISS(vectorstore.index))
retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)    


# question_gen_llm = OpenAI(
#     temperature=0,
#     verbose=True
# )

# streaming_llm = OpenAI(
#     streaming=True,
#     verbose=True,
#     temperature=0,
# )

# question_generator = LLMChain(
#     llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT
# )
# doc_chain = load_qa_chain(
#     streaming_llm, chain_type="stuff", prompt=QA_PROMPT
# )

# qa = ConversationalRetrievalChain(
#     vectorstore=vectorstore,
#     combine_docs_chain=doc_chain,
#     question_generator=question_generator
# )