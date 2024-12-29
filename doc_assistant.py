from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Carregar variáveis de ambiente
load_dotenv()

app = Flask(__name__)

# Carregar chaves de acesso das variáveis de ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Inicializar o cliente da OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Inicializar Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index(PINECONE_INDEX)

@app.route('/doc-assistant', methods=['POST'])
def doc_assistant():
    data = request.json
    question = data.get('question', '')

    # Verificar se a pergunta foi fornecida
    if not question:
        return jsonify({'error': 'A pergunta não foi fornecida.'}), 400

    # Gerar o embedding da pergunta
    question_embedding = generate_embedding(question)

    # Fazer busca semântica no Pinecone para obter contextos relevantes
    search_results = semantic_search(question_embedding)
    print('Resultados da busca:', search_results)

    # Formatar o contexto obtido do Pinecone
    additional_context = "\n\n".join(
        [match["metadata"].get("documentation", "") for match in search_results]
    )

    # Gerar a resposta usando a OpenAI
    response = generate_response(question, additional_context)
    print('Resposta gerada:', response)

    return jsonify({'response': response}), 200

def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return embedding

def semantic_search(embedding, top_k=3):
    response = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    results = [
        {"id": match["id"], "score": match["score"], "metadata": match.get("metadata", {})}
        for match in response["matches"]
    ]
    return results

def generate_response(question, context):
    prompt = f"""
<Persona>
Você é um assistente especializado em ajudar desenvolvedores com documentações técnicas. Sua resposta deve ser clara, concisa e útil, utilizando apenas o contexto fornecido como referência nas respostas.
</Persona>

<Context>
{context}
</Context>

<Question>
{question}
</Question>
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Você é um assistente responsável por ajudar desenvolvedores com dúvidas sobre documentações."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content

if __name__ == '__main__':
    app.run(port=5001)