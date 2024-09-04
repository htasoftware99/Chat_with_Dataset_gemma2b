import pandas as pd
import streamlit as st
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama

st.set_page_config(
    page_title="Data Frame Chat",
    page_icon="💬",
    layout="centered"
)

def read_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx") or file.name.endswith(".xls"):
        return pd.read_excel(file)
    else:
        return None

st.title("🤖 DataFrame ChatBot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "df" not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    st.session_state.df = read_data(uploaded_file)
    if st.session_state.df is not None:
        st.write("DataFrame Preview:")
        st.dataframe(st.session_state.df.head())
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Ask me anything about the dataset...")

if user_prompt and st.session_state.df is not None:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role":"user","content": user_prompt})

    llm = ChatOllama(model="gemma:2b", temperature=0)

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        st.session_state.df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )

    messages = [
        {"role":"system", "content": "You are a helpful assistant"},
        *st.session_state.chat_history
    ]

    response = pandas_df_agent.invoke(messages)

    assistant_response = response["output"]

    st.session_state.chat_history.append({"role":"assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
elif user_prompt:
    st.warning("Please only ask questions about the dataset.")
