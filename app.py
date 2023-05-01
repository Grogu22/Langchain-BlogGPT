# Importing dependencies
import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

st.title('🦜🔗 BlogGPT📝')
st.subheader('Creating blogs with the power of Langchain and OpenAI !!!')
st.caption('Want to write a blog? Confused about what to write about? 🤡')
st.caption('Fear no more... 😌')
st.caption('Choose a domain.. we will choose a cool topic for you and write a blog on it 😎')
api_key = st.text_input("Enter your OpenAI API key", type="password")
if (api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    prompt = st.text_input("Enter the domain to write a blog on:")

    # customizations
    with st.sidebar:
        add_radio = st.radio(
            "Choose number of tokens for your blog post",
            ("Simple😁🤟(300-400 tokens)","Semi-pro😎🙌(400-500 tokens)","Pro🤡💪(500-600 tokens)")
        )

    # Prompt templates
    title_template = PromptTemplate(
        input_variables = ['topic'],
        template='Suggest me a sample blog title about {topic}'
    )

    literature_template = PromptTemplate(
        input_variables = ['title'],
        template='Write me a blog on having the title: {title}'
    )

    # Memory
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    literature_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

    # LLMs
    if add_radio == "Simple😁🤟(300-400 tokens)":
        llm = OpenAI(temperature=0.7, max_tokens=300)
    elif add_radio == "Semi-pro😎🙌(400-500 tokens)":
        llm = OpenAI(temperature=0.7, max_tokens=400)
    elif add_radio == "Pro🤡💪(500-600 tokens)":
        llm = OpenAI(temperature=0.7, max_tokens=500)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    literature_chain = LLMChain(llm=llm, prompt=literature_template, verbose=True, output_key='literature', memory=literature_memory)

    # Show output
    with get_openai_callback() as cb:
        if prompt:
            title = title_chain.run(prompt)
            literature = literature_chain.run(title=title)

            st.write(title)
            st.write(literature)

            with st.expander('Title History'):
                st.info(title_memory.buffer)
        
            with st.expander('Literature History'):
                st.info(literature_memory.buffer)
                
            with st.expander('Token usage'):    
                st.info(f'Spent a total of {cb.total_tokens} tokens')

