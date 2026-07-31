
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM

def get_qa_chain():
    # Placeholder LLM for GitHub execution without API key limits.
    # Replace with ChatOpenAI(model='gpt-3.5-turbo') for real-world usage.
    llm = FakeListLLM(responses=["Here is the contextually retrieved answer based on the provided document.", 
                                 "I don't know the answer to that."])
    
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return llm, PROMPT

def run_retrieval(vector_store, query):
    docs = vector_store.similarity_search(query, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    llm, prompt = get_qa_chain()
    prompt_formatted = prompt.format(context=context, question=query)
    response = llm.invoke(prompt_formatted)
    return response, docs
