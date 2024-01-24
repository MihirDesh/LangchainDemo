import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

os.environ['OPENAI_API_KEY'] = apikey

st.title('🦜️🔗 YouTube GPT Creator')
prompt = st.text_input("Plug in your prompt here")

title_template = PromptTemplate(
    input_variables = ['topic'],
    template='Write me a YouTube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title'],
    template='Write me a YouTube video script on this title TITLE: {title}'
)

llm = OpenAI(temperature=0.6)

title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script')
sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

if prompt:
    response = sequential_chain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])
