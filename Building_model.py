import ollama
def build_prompt(query,relevant_context=""):
    prompt=f"""
   You are a helpful assistant. Use ONLY the following documents to answer.
    If the answer is not contained,
    respond 'I don't know' or 'No answer in provided documents'. Do not guess"
    Context:{relevant_context}
    Question:{query}
    """
    return prompt
def answer_with_rag(prompt):
    try:
        response=ollama.chat(
            model="mistral",
            messages = [{"role":"user","content":prompt}],
            options={
             "temperature":0.6,
              "top_k":40,
            }
        )
        return response["message"]["content"]
    except Exception as e:
        print(e)
        return "Error calling Mistral"