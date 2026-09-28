import streamlit as st
import ollama
st.title("Friendly AI Bot")
st.write("Ask me anything !")
question = st.text_input("Enter your question here:")
if st.button("Ask AI"):
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are a friendly assistant. Explain things in simple language.Be helpful and encouraging."""},
            {"role": "user", "content": question}   
        ]
    )
    answer = response["message"]["content"]
    st.write(" AI Response")
    st.write(answer)

import streamlit as st
import ollama
import chromadb
from pypdf import PdfReader
from streamlit_mic_recorder import speech_to_text


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="centered"
)


# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
    min-height: 100vh;
}

.block-container {
    max-width: 900px;
    padding-top: 35px;
}

.title {
    text-align: center;
    color: white;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #eeeeee;
    font-size: 17px;
    margin-bottom: 30px;
}

.user-message {
    background: white;
    color: #333333;
    padding: 15px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 12px 0 12px 80px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.ai-message {
    background: rgba(255,255,255,0.18);
    color: white;
    padding: 15px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 12px 80px 12px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.status-box {
    background: rgba(255,255,255,0.15);
    padding: 15px;
    border-radius: 15px;
    color: white;
    margin: 15px 0;
}

.footer {
    text-align: center;
    color: #eeeeee;
    margin-top: 30px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# TITLE
# =====================================================

st.markdown(
    '<div class="title">🤖 AI PDF Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Chat with your PDF using AI + Voice'
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""


# =====================================================
# CHROMADB
# =====================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="pdf_documents"
)


# =====================================================
# CREATE EMBEDDING
# =====================================================

def create_embedding(text):

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )

    return response["embedding"]


# =====================================================
# SPLIT TEXT
# =====================================================

def split_text(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# =====================================================
# PDF UPLOAD
# =====================================================

st.subheader("📄 Upload your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# =====================================================
# PROCESS PDF
# =====================================================

if uploaded_file is not None:

    if st.button("🔍 Process PDF"):

        try:

            with st.spinner("📖 Reading PDF..."):

                reader = PdfReader(uploaded_file)

                full_text = ""

                for page in reader.pages:

                    page_text = page.extract_text()

                    if page_text:
                        full_text += page_text + "\n"


            if not full_text.strip():

                st.error(
                    "❌ No readable text found in this PDF."
                )

            else:

                with st.spinner(
                    "🧠 Creating PDF embeddings..."
                ):

                    # Delete old collection
                    try:
                        client.delete_collection(
                            "pdf_documents"
                        )
                    except Exception:
                        pass

                    collection = client.create_collection(
                        name="pdf_documents"
                    )

                    # Split text
                    chunks = split_text(
                        full_text
                    )

                    # Add chunks
                    for index, chunk in enumerate(chunks):

                        embedding = create_embedding(
                            chunk
                        )

                        collection.add(
                            ids=[str(index)],
                            embeddings=[embedding],
                            documents=[chunk]
                        )

                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name

                st.success(
                    "✅ PDF processed successfully!"
                )

                st.info(
                    f"📄 File: {uploaded_file.name} | "
                    f"📑 Pages: {len(reader.pages)} | "
                    f"🧩 Chunks: {len(chunks)}"
                )

        except Exception as e:

            st.error(
                f"❌ Error processing PDF: {e}"
            )


# =====================================================
# PDF STATUS
# =====================================================

if st.session_state.pdf_processed:

    st.markdown(
        f"""
        <div class="status-box">
        🟢 <b>PDF Ready</b><br>
        📄 {st.session_state.pdf_name}<br>
        You can now ask questions using text or voice.
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="user-message">
                👤 <b>You</b><br>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="ai-message">
                🤖 <b>AI</b><br>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )


# =====================================================
# VOICE INPUT
# =====================================================

st.subheader("🎤 Ask by Voice")

voice_text = speech_to_text(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    language="en",
    use_container_width=True,
    key="voice_input"
)


# =====================================================
# QUESTION INPUT
# =====================================================

if voice_text:

    question = voice_text

    st.success(
        f"🎤 You said: {question}"
    )

else:

    question = st.text_input(
        "💬 Or type your question:",
        placeholder="Ask something about your PDF..."
    )


# =====================================================
# ASK AI
# =====================================================

if st.button("✨ Ask AI"):

    if not question.strip():

        st.warning(
            "⚠️ Please enter or speak a question."
        )

    elif not st.session_state.pdf_processed:

        st.warning(
            "⚠️ Please upload and process a PDF first."
        )

    else:

        # Add user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        try:

            # =========================================
            # QUESTION EMBEDDING
            # =========================================

            with st.spinner(
                "🔎 Searching your PDF..."
            ):

                question_embedding = create_embedding(
                    question
                )

                results = collection.query(
                    query_embeddings=[
                        question_embedding
                    ],
                    n_results=4
                )


            # =========================================
            # GET RELEVANT TEXT
            # =========================================

            documents = results.get(
                "documents",
                [[]]
            )[0]

            context = "\n\n".join(
                documents
            )


            # =========================================
            # RAG PROMPT
            # =========================================

            prompt = f"""
You are a helpful AI PDF assistant.

Answer the user's question using ONLY
the information found in the PDF context below.

If the answer cannot be found in the PDF,
say:

"I couldn't find that information in the PDF."

Explain the answer in simple and clear language.

PDF CONTEXT:
-------------------------
{context}
-------------------------

USER QUESTION:
{question}
"""


            # =========================================
            # LLAMA 3.2
            # =========================================

            with st.spinner(
                "🤖 Generating answer..."
            ):

                response = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a friendly AI PDF assistant. "
                                "Give accurate and simple answers."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )


            answer = response["message"]["content"]


            # =========================================
            # SAVE ANSWER
            # =========================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            st.rerun()


        except Exception as e:

            st.error(
                f"❌ AI Error: {e}"
            )


# =====================================================
# CLEAR CHAT
# =====================================================

if st.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    '<div class="footer">'
    'Powered by Ollama • Llama 3.2 • '
    'ChromaDB • RAG 🚀'
    '</div>',
    unsafe_allow_html=True
)