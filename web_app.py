import streamlit as st
from main import OllamaChatBot  # Import your main class

st.title("ESG / Document Chatbot")
st.markdown("Upload PDFs and chat with your documents using local Ollama")

# Initialize the bot (only once)
if "bot" not in st.session_state:
    st.session_state.bot = OllamaChatBot()

bot = st.session_state.bot

# File uploader
uploaded_file = st.file_uploader("Upload PDF document", type="pdf")

if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Processing PDF..."):
            # Save uploaded file temporarily
            temp_path = "temp_uploaded.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Use your existing function
            chunks = bot.accepting_user_file_adapted(temp_path)  # We'll add this method
            if chunks:
                bot.process_uploaded_documents(chunks)  # You may need to adjust this
                st.success(f"✅ Processed {len(chunks)} chunks!")
            else:
                st.error("Failed to process PDF")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the document"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = bot.ask_question(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})