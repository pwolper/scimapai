#!/usr/bin/env python3

import os, sys, time
import json
from typing import List
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import streamlit.components.v1 as components
from pyvis.network import Network
from pyvis import network as net
from PyPDF2 import PdfReader

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import faiss


### Functions ###

def get_text():
    text_box=st.empty()
    text_box.text_area(label="Text Input",
                       # label_visibility='collapsed',
                       placeholder="Paste a scientific text...", key="text_input",
                       height=200)
    if st.session_state.text_input != "":
        if st.button("New Input"):
            del st.session_state.text_input
            st.rerun()
        text_box.info(str("Prompt: \"" +st.session_state.text_input+"\""))
    return


def get_pdf():
    pdf = st.file_uploader("PDF upload")
    if pdf is not None:
        pdf = PdfReader(pdf)
        text = " ".join(page.extract_text() for page in pdf.pages)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len
        )

        chunks = text_splitter.create_documents([text])

        st.success("The file has been split into " + str(len(chunks)) + " chunks.")
        return(chunks)
    else:
        return(pdf==None)


#@st.cache_data(show_spinner="Generating vectorstore from embeddings and text...")
def create_vectorstore(chunks):
        embeddings = OpenAIEmbeddings()
        vectorstore = faiss.FAISS.from_documents(chunks, embeddings)
        return(vectorstore)

def get_api_key():
    if "openai_api_key" not in st.session_state:
        if not os.getenv("OPENAI_API_KEY"):
            openai_input_field=st.empty()
            openai_input_field.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...",
                                          key="openai_api_key_input", type="password",
                                          help="Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)")
            if st.session_state.openai_api_key_input != "":
                st.session_state.openai_api_key=st.session_state.openai_api_key_input
                openai_input_field.success("API key saved...")
                time.sleep(.5)
                openai_input_field.empty()
        else:
            st.session_state.openai_api_key=os.getenv("OPENAI_API_KEY")
            return

def main_menu():
    option = st.selectbox("What you  you like to do?", options=("Input a text", "Upload a PDF file", "Chat with the AI"),
                          index=None, placeholder="Select an option...")
    if option == "Input a text":
        get_text()
    if option == "Upload a PDF file":
        chunks = get_pdf()

        if chunks is not None:
            summary = summary_chain(chunks, st.session_state.openai_api_key)
            summary_text = summary["output_text"]


            st.markdown("**Summary:**")
            st.write(summary_text)

            mapping_output = llm_network_call(summary_text, st.session_state.openai_api_key)
            nodes, edges = json_parsing(mapping_output)
            source_code=pyvis_graph(nodes, edges)
            st.markdown("**Knowledge Graph:**")
            components.html(source_code, height=550,width=1350)
            download=st.download_button("Download HTML", data=source_code, file_name="knowledge_graph.html")

        #vectorstore = create_vectorstore(chunks)

        #st.write(chunks)
    return


@st.cache_data(show_spinner="Generating concept map from text...")
def llm_network_call(text_input, openai_api_key):
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=openai_api_key)

    mapping_template = """
    You are an scientific assistant, tasked with extracting the main concepts and key words from articles and abstracts. I want to create a concept map to enhance understanding of the text.
    Breaking down the text hierarchically and semantically, I want you to choose high-level concepts as nodes, avoiding overlap between concepts. Keep the number of nodes to a minimum.
    Concepts should be as atomistic as possible, but chosen so that there is high connectivity in the graph.
    Return nodes and edges for the concept map and come up with a sentence explaining each edge as well.
    Text: {text}

    Strictly return a list of json objects, with the following fields:
    "node_1", "node_2" and "edge". Where "node_1" and "node_2", represent two nodes and "edge" is a string containing a sentence describing the relationship between the nodes.
    Do not wrap the output in ```json ```.
    """

    mapping_prompt = ChatPromptTemplate.from_messages(
        [("system", mapping_template),
         ("human", "{text}"),
         ])

    messages = mapping_prompt.format_messages(text=text_input)
    answer = llm(messages)
    output = answer.content
    return(output)

@st.cache_data(show_spinner="Summarizing the text...")
def llm_summary_call(text_input, openai_api_key):
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.5, openai_api_key=openai_api_key)

    summary_template = """
    You are an scientific assistant, tasked with summarizing a scientific text. When summarizing make sure to mention, the research aims, methods and results.
    Also mention how the results can be interpreted in the broader picture of the research area. Return a short summary of the text.
    Text: {text}
    """

    summary_prompt = ChatPromptTemplate.from_messages(
        [("system", summary_template),
         ("human", "{text}"),
         ])

    summary_message = summary_prompt.format_messages(text=text_input)
    answer = llm(summary_message)
    output = answer.content
    return(output)

#@st.cache_data(show_spinner="Summarizing the document...")
def summary_chain(chunks, openai_api_key):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.5, openai_api_key=openai_api_key)
    chain = load_summarize_chain(llm=llm, chain_type="stuff")
    summarization = chain.invoke(chunks)
    return(summarization)


def json_parsing(mapping_output):
    output_dict = json.loads(mapping_output)

    nodes = []
    edges = []

    for dict in output_dict:
        node_1 = dict['node_1']
        node_2 = dict['node_2']
        edge = dict['edge']

        nodes.append(node_1)
        nodes.append(node_2)

        edges.append((node_1, node_2, edge))
    return(nodes, edges)

def pyvis_graph(nodes, edges):
    nt = Network(directed=False,
                 notebook=True,height="540px",width="1349px",
                #height="480px",
                #width="620px",
                #width="940px",
                heading='')

    for n in nodes:
        nt.add_node(n,
                    title=n,
                    size=15)

    for source, target, label in edges:
        nt.add_edge(source,
                    target,
                    title=label)

    # nt.barnes_hut()
    nt.show('pyvis_knowledge_graph.html')
    html_file = open('./pyvis_knowledge_graph.html', 'r', encoding='utf-8')
    source_code = html_file.read()
    return(source_code)


### Streamlit page starts here ###

st.set_page_config(page_title="SciMapAI", page_icon=":books:", layout="wide", initial_sidebar_state="collapsed")
st.header("SciMapAI: Creating text-based knowledge graphs using AI")
st.write("Exploring scientific texts and concepts through interactive knowledge graphs generated by AI.")

st.sidebar.markdown("**Description**")
st.sidebar.markdown("SciMapAI uses OpenAIs GPT-4 model, to extract concepts and their relationships from scientific texts. It aims to provide a visual way of understanding complex texts, by leveraging knowlegde graphs.")
st.sidebar.markdown("**Usage**")
st.sidebar.markdown("After pasting your OpenAI API key, paste any scientific text you wish to extract a map of concepts from into the text box. Alternative ask for explanation of a topic. You can then browse the resulting graph of concepts, by dragging and clicking on nodes and edges.")
st.sidebar.markdown("**Contact**")
st.sidebar.markdown("Created by Philip Wolper [phi.wolper@gmail.com](phi.wolper@gmail.com). Code is available on [GitHub](https://github.com/pwolper/scimapai.git) here. Feeback is very welcome.")
st.sidebar.markdown("**Knowledge Graph Options**")
debug=st.sidebar.checkbox("Show debugging information")

get_api_key()

if "openai_api_key" in st.session_state:
    main_menu()

if "text_input" in st.session_state:
    if st.session_state.text_input != "":
        mapping_output = llm_network_call(st.session_state.text_input, st.session_state.openai_api_key)
        nodes, edges = json_parsing(mapping_output)
        source_code=pyvis_graph(nodes, edges)
        st.markdown("**Knowledge Graph:**")
        components.html(source_code, height=550,width=1350)
        download=st.download_button("Download HTML", data=source_code, file_name="knowledge_graph.html")

        st.markdown("**Summary:**")
        summary = llm_summary_call(st.session_state.text_input, st.session_state.openai_api_key)
        st.markdown(summary)
    else:
        st.stop()

if debug:
    with st.expander("**LLM output and Data Structure**"):
        st.markdown("**LLM output:**")
        st.markdown(mapping_output)
        st.markdown("**Nodes:**")
        st.write(nodes)
        st.markdown("**Edges:**")
        st.write(edges)
