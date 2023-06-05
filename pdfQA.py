import openai
import os
from typing import Dict
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.callbacks import get_openai_callback
from llama_index.prompts.prompts import RefinePrompt
from langchain.prompts.chat import (
  AIMessagePromptTemplate,
  ChatPromptTemplate,
  HumanMessagePromptTemplate
)

from llama_index import download_loader
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, QuestionAnswerPrompt, ServiceContext, LLMPredictor, PromptHelper

@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}

def main():
    st.title("Transcribe Audio with OpenAI")

    # APIキーを設定します
    openai.api_key = st.text_input("openAIのAPIキーを入力してください。", value=openai.api_key)
    
    ##==== Download
    # pdf_directory = "/content/drive/MyDrive/Playlist/00.Desk/14.渡部諒/Model"
    # pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith(".pdf")]
    documents = []
    PDFReader = download_loader("CJKPDFReader")
    uploaded_pdf_file = ''
    # Upload the audio file
    static_store = get_static_store()
    
    uploaded_pdf_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True, type=["pdf"])

    with st.form(key='pdfqa'):
        question = st.text_input("質問内容を入力してください。（例）有給休暇について教えてください。",value="有給休暇について教えてください。")
                
        for uploaded_file in uploaded_pdf_files:
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
        if uploaded_pdf_files is None:
            st.warning("Please upload an audio file.")

        for filename in uploaded_pdf_files:
            st.write(filename)

            ##==== Load & Split
            loader = PDFReader()
            if documents:
                documents += loader.load_data(filename)
            else:
                documents = loader.load_data(filename)

        # text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=10)
        print(documents)
        # pages = text_splitter.split_documents(documents)
        # print(pages)

        ##==== Chroma
        # 使用するLLMやパラメータをカスタマイズする
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
        max_input_size = 4096
        num_output = 256
        chunk_overlap_ratio = 0.2
        prompt_helper = PromptHelper(max_input_size, num_output, chunk_overlap_ratio)
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist()

        ##==== QA
        print(uploaded_pdf_files, "\n")

        if (documents is not None) and (uploaded_pdf_files != []):
            submit_btn = st.form_submit_button('送信')
        if submit_btn:
            with get_openai_callback() as cb:
                # プロンプトのテンプレートをカスタマイズする
                QA_PROMPT_TMPL = (
                    """
                    You are an assistant AI that summarizes Japanese sentences.
                    Please summarize the text entered by the user within 300 characters.
                    Be sure to answer only the content of the question.
                    If the keyword for your question is not in the source, please refrain from answering.

                    "Also, related information is below."
                    "---------------------\n"
                    "{context_str}"
                    "\n---------------------\n"
                    "{query_str}\n"
                    If you have a specific answer, and if you have a citation source, please provide it in the following format.
                    Answer :
                      ***\n\n
                    - Source : ***\n
                    --item1 : ***\n
                    --item2 : ***\n
                    --item3 : ***\n

                    Answering is in Japanese.
                    """
                )

                CHAT_REFINE_PROMPT_TMPL_MSGS = [
                    HumanMessagePromptTemplate.from_template("{query_str}"),
                    AIMessagePromptTemplate.from_template("{existing_answer}"),
                    HumanMessagePromptTemplate.from_template(
                        """
                        以下の情報を参照してください。 \n"
                        "---------------------\n"
                        "{context_msg}"
                        "\n---------------------\n"
                        この情報が回答の改善に役立つようならこの情報を使って回答を改善してください。
                        この情報が回答の改善に役立たなければ元の回答を日本語で返してください。
                        """
                    )
                ]

                CHAT_REFINE_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_PROMPT_TMPL_MSGS)

                QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
                CHAT_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_PROMPT_LC)
                query_engine = index.as_query_engine(
                    text_qa_template=QA_PROMPT,
                    refine_template=CHAT_PROMPT,
                    similarity_top_k=3
                )
                
                # ユーザーの入力文に関連する部分を抽出し、プロンプトに追加した上でユーザーの入力文をChatGPTに渡す
                response = query_engine.query(question)
                print("Question:")
                print(question, "\n")
                print("Answer:")
                print(f"{response}", "\n")
                st.write(f"{response}", "\n",f"{response.get_formatted_sources()}")
                print("--------")
                print("total_tokens: ", cb.total_tokens)
                print("prompt_tokens: ", cb.prompt_tokens)
                print("completion_tokens: ", cb.completion_tokens)
                print("total_cost: ", cb.total_cost)

main()
