import ollama,os,json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Document_work import accepting_user_file
from pypdf import PdfReader
from Building_model import answer_with_rag,build_prompt
from Embeddings_work import embed_file_chunks,embed_user_query,calculate_cosine_similarity_then_retrieval
class OllamaChatBot:
    def __init__(self,model="mistral",memory_file="memory.json"):
        self.model = model
        self.memory_file = memory_file
        self.chat_history = []
        self.for_retrieval_list = []
        self.chunk_embeddings = None
        self._load_memory()
    """Load chat history from memory.json"""
    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file,"r",encoding="utf-8") as f:
                    self.chat_history = json.load(f)
                print(f"✅ Loaded {len(self.chat_history)} messages")
            except Exception as e:
                print(f"⚠️  Couldn't load memory file: {e}")
                self.chat_history = []

    """Save chat history to memory.json"""
    def _save_memory(self):
        try:
            with open(self.memory_file,"w",encoding="utf-8") as f:
                json.dump(self.chat_history,f,indent=4)
        except Exception as e:
            print(f"❌ Error saving memory:{e}")
    """Clear all memory"""
    def _clear_memory(self):
        self.chat_history = []
        self.for_retrieval_list = []
        self.chunk_embeddings = None
        self._save_memory()
        print("✅  Memory cleared")
    """Process uploaded document chunks"""
    def process_uploaded_documents(self,chunks):
        if not chunks:
            print("❌ No chunks uploaded")
            return
        self.for_retrieval_list.extend(chunks)
        print(f"✅ Added {len(chunks)} chunks")
        # Create embeddings (cached)
        if self.chunk_embeddings is None:
            print("🔄 Creating embeddings...")
            self.chunk_embeddings = embed_file_chunks(self.for_retrieval_list)
            print("✅ Embeddings created!")

        print("\n💬 Ask questions (type 'exit' to stop):")

        while True:
            question = input("You: ")
            if question.lower() == "exit":
                break

            # Get answer
            answer = self.ask_question(question)
            print(f"Assistant: {answer}")

            # Save to memory
            self.chat_history.append({'role': 'user', 'content': question})
            self.chat_history.append({'role': 'assistant', 'content': answer})
            self._save_memory()

    def accepting_user_file_adapted(self, file_path):
        """Adapted version for web upload - no easygui"""
        try:
            from Document_work import accepting_user_file  # fallback
            # Try to reuse logic but with direct file path
            # For simplicity, we'll call the core extraction part
            reader = PdfReader(file_path)  # You'll need to import at top if not already
            full_text = ""
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    full_text += f"[Page {page_num}] {page_text}\n\n"

            if not full_text.strip():
                return None

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len
            )
            return splitter.split_text(full_text)
        except Exception as e:
            print(f"Error processing file: {e}")
            return None
    def ask_question(self, question):
        """Ask a question about the document"""
        if not self.for_retrieval_list:
            return "Please upload a document first using 'yes'"

        # Get query embedding
        q_emb = embed_user_query(question)

        # Retrieve relevant chunks
        top_indices = calculate_cosine_similarity_then_retrieval(
            self.chunk_embeddings, q_emb
        )
        relevant_memories = [self.for_retrieval_list[i] for i in top_indices[:3]]
        # noinspection PyTypeChecker
        relevant_memories = '\n'.join(relevant_memories)

        # Generate answer
        prompt = build_prompt(question, relevant_memories)
        return answer_with_rag(prompt)

    def general_chat(self, message):
        """Handle general chat (no document needed)"""
        # Add user message
        self.chat_history.append({
            'role': 'user',
            'content': message
        })

        # Get response
        response = ollama.chat(
            model=self.model,
            messages=self.chat_history
        )
        assistant = response['message']

        # Add assistant response
        self.chat_history.append(assistant)

        # Save to retrieval list for future context
        self.for_retrieval_list.append(assistant['content'])

        # Save memory
        self._save_memory()

        return assistant['content']

    def run(self):
        """Main chat loop"""
        print("\n" + "=" * 50)
        print("🤖 Welcome to Ollama ChatBot!")
        print("Commands: 'yes' (upload doc) | 'no' (general chat)")
        print("          'clear' (reset) | 'quit' (exit)")
        print("=" * 50 + "\n")

        while True:
            command = input("You: ").lower()

            if command == "quit":
                print("\n👋 Goodbye!")
                break

            elif command == "clear":
                self._clear_memory()

            elif command == "yes":
                print("\n📄 Upload your PDF...")
                chunks = accepting_user_file()
                if chunks:
                    self.process_uploaded_documents(chunks)
                else:
                    print("❌ Failed to load document")

            elif command == "no":
                print("\n💬 General chat (type 'exit' to stop):")
                while True:
                    user_input = input("You: ")
                    if user_input.lower() == "exit":
                        break
                    response = self.general_chat(user_input)
                    print(f"Assistant: {response}")

            else:
                print("❌ Unknown command")

# Run
if __name__ == "__main__":
    bot = OllamaChatBot()
    bot.run()