# webhook_server.py

from flask import Flask, request, jsonify
import requests
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

app = Flask(__name__)

# Carregar chaves de acesso das variáveis de ambiente
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Inicializar o cliente da OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Inicializar Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in pinecone.list_indexes().names():
  raise ValueError(f"O índice '{PINECONE_INDEX}' não foi encontrado no Pinecone.")
index = pinecone.Index(PINECONE_INDEX)

@app.route("/webhook", methods=["POST"])
def webhook():
  # Receber os dados do webhook
  data = request.json

  # Verificar se a pull request é do tipo opened ou reopened e se é para a branch main
  if data["action"] not in ["opened", "reopened"] or data["pull_request"]["base"]["ref"] != "main":
    return jsonify({}), 204
  print("action:", data["action"])

  # Dados da pull request para acessar a API do GitHub
  user = data["pull_request"]["user"]["login"]
  repo = data["pull_request"]["head"]["repo"]["name"]
  pr_number = data["pull_request"]["number"]
  api_url = f"https://api.github.com/repos/{user}/{repo}/pulls/{pr_number}/files"

  headers = {
    "Authorization": f"token {GITHUB_TOKEN}"
  }

  # Buscar os arquivos da pull request
  response = requests.get(api_url, headers=headers)
  if response.status_code == 200:
    files = response.json()

    # Processar os patches dos arquivos
    result = []
    for file in files:
      patch = file.get("patch", "")
      if patch:
        clean_patch = "\n".join(
          line for line in patch.split("\n") if not line.startswith("@@")
        )
      result.append(f"Nome do arquivo: {file['filename']}\n\n{clean_patch}\n\n")

    formatted_result = "\n".join(result)
    print("Resultado Processado:")
    print(formatted_result)

    # Gerar o embedding do código
    embedding = generate_embedding(formatted_result)

    # Fazer busca semântica no Pinecone para obter exemplos de documentações parecidas 
    search_results = semantic_search(embedding)

    # Formatar o contexto obtido do Pinecone
    additional_context = "\n\n".join(
      [match["metadata"].get("documentation", "") for match in search_results]
    )
    print("additional_context:", additional_context)

    # Gerar a documentação
    documentation = generate_documentation(formatted_result, additional_context)
    print("documentation:", documentation)

    embbeded_documentation = generate_embedding(documentation)

    # Enviar o embbeded_documentation para o Pinecone com id com nome da branch
    response = index.upsert(
      vectors=[
        {"id": data["pull_request"]["head"]["ref"], "values": embbeded_documentation}
      ]
    )
    print("Documentação enviada para o Pinecone:", response)

    # Enviar a documentação como comentário na pull request
    comment_url = f"https://api.github.com/repos/{user}/{repo}/issues/{pr_number}/comments"
    comment_data = {
      "body": documentation
    }
    response = requests.post(comment_url, headers=headers, json=comment_data)
    if response.status_code == 201:
      print("Comentário com documentação enviado com sucesso!")
    else:
      print("Erro ao enviar comentário com documentação:", response.status_code, response.text)

    return jsonify({
      "message": "Documentação gerada com sucesso!",
      "documentation": documentation
    }), 200

  else:
    print("Erro ao buscar arquivos:", response.status_code, response.text)
    return jsonify({"error": "Erro ao buscar arquivos"}), response.status_code

# Geração de embeddings com a OpenAI
def generate_embedding(text):
  """Gera um embedding do texto usando a API da OpenAI"""
  response = client.embeddings.create(
    input=text,
    model="text-embedding-ada-002"
  )
  embedding = response.data[0].embedding
  return embedding

# Busca semântica no Pinecone
def semantic_search(embedding, top_k=3):
  """Faz uma busca semântica no Pinecone"""
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

# Geração de documentação com a OpenAI
def generate_documentation(code, context):
  prompt = f"""
<Persona>
Você é um especialista em gerar documentações técnicas detalhadas, claras e completas para códigos. Sua documentação deve ser focada na explicação passo a passo, com exemplos práticos de uso, detalhamento de parâmetros e tipos de retorno, além de especificar como tratar exceções e erros comuns. A documentação deve ser acessível para desenvolvedores iniciantes e experientes.
</Persona>

<ToDo>
- Crie uma documentação profissional e completa para os arquivos dentro do Código, incluindo uma visão geral, exemplos de uso, instruções de utilização, detalhes sobre parâmetros e tipos de retorno, e tratamento de exceções.
- Para cada função ou módulo, forneça um exemplo de código funcional, teste unitário, e a explicação do comportamento esperado em diferentes cenários.
- Escreva a documentação de forma que facilite o onboarding de novos desenvolvedores.
</ToDo>

<Context>
{context}
</Context>

<Código>
{code}
</Código>
"""

  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "Você é um assistente responsável por gerar documentações técnicas."},
      {"role": "user", "content": prompt}
    ],
    temperature=0.7
  )

  return response.choices[0].message.content


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)