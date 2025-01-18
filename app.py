# Importando bibliotecas essenciais para o funcionamento do script
import os
import streamlit as st # Importando a biblioteca Streamlit para criação de aplicações web
from langchain.llms import OpenAI # Importando o modelo de linguagem da OpenAI
from langchain.prompts import PromptTemplate # Importando a classe para templates de prompt
from langchain.chains import LLMChain, SequentialChain # Importando classes para criar cadeias de LLM
from langchain.memory import ConversationBufferMemory # Importando memória de buffer de conversação
import requests # Importando a biblioteca para realizar requisições HTTP

# Definindo uma classe para pesquisar no Google usando a API Serper.dev
class SerperAPIWrapper:
    def __init__(self, api_key):
        self.api_key = api_key # Inicializa a classe com a chave da API
        self.url = "https://google.serper.dev/search" # Define a URL para a API

    # Método para executar a pesquisa
    def run(self, query):
        headers = {
            "X-API-KEY": self.api_key, # Cabeçalhos com a chave da API
            "Content-Type": "application/json"
        }
        payload = {
            "q": query # Payload com a consulta de pesquisa
        }
        response = requests.post(self.url, headers=headers, json=payload) # Faz uma requisição POST
        if response.status_code == 200:
            results = response.json() # Obtém os resultados da resposta JSON
            snippets = [item['snippet'] for item in results.get('organic', [])] # Extrai snippets dos resultados orgânicos
            return "\n".join(snippets) # Retorna os snippets como uma string
        else:
            return "No results found." # Retorna mensagem de erro se a requisição falhar

# Configurando o título da aplicação no Streamlit
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Escreva aqui o tema do conteúdo') # Campo de entrada para o usuário escrever o tema
openai_api_key = st.text_input("Enter your OpenAI API Key", type='password')
serper_api_key = st.text_input("Enter your Serper API Key", type='password')

# Definindo templates de prompt para o título do vídeo e o roteiro
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Escreva um título para um vídeo do YouTube sobre... {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'google_research'], 
    template='Escreva um roteiro de vídeo do YouTube baseado neste título : {title} enquanto aproveita esta pesquisa do Google: {google_research}'
)

# Configurando memória para armazenar o histórico de conversação
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Inicializando o modelo de linguagem com uma temperatura de 0.9
llm = OpenAI(temperature=0.9)
# Configurando a cadeia de LLM para gerar títulos
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
# Configurando a cadeia de LLM para gerar roteiros
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

# Inicializando o wrapper da API Serper.dev
google_search = SerperAPIWrapper(api_key=serper_api_key)

# Mostrando os resultados na tela se houver um prompt
if prompt: 
    # Configurando a chave de API da OpenAI no ambiente
    os.environ['OPENAI_API_KEY'] = openai_api_key

    title = title_chain.run(prompt) # Gera o título do vídeo
    google_research = google_search.run(prompt) # Realiza a pesquisa no Google
    script = script_chain.run(title=title, google_research=google_research) # Gera o roteiro do vídeo

    st.write(title) # Exibe o título gerado
    st.write(script) # Exibe o roteiro gerado

    with st.expander('Title History'): 
        st.info(title_memory.buffer) # Exibe o histórico dos títulos gerados

    with st.expander('Script History'): 
        st.info(script_memory.buffer) # Exibe o histórico dos roteiros gerados

    with st.expander('Google Research'): 
        st.info(google_research) # Exibe os resultados da pesquisa no Google
