# Exemplos de uso da API Laya

Comece pelos exemplos **06–07** de [requests.http](requests.http) para escolher uma
categoria em uma lista ou em um mapa com descrições. O cliente
[classificar_categorias.py](classificar_categorias.py) utiliza um dicionário
`dict[str, str]`, com o nome da categoria como chave e sua descrição como valor.
Os exemplos acessam a API HTTP deste projeto; o cliente não carrega modelos localmente.

## Executar o exemplo Python

O exemplo assume que o Docker já está rodando e que a API está disponível.
URL e timeout ficam explícitos na chamada `self._client.post` do arquivo Python.
A autenticação é HTTP Basic, com as credenciais lidas de `BASIC_AUTH` no `.env`
da raiz do projeto. O caminho é resolvido a partir do script, independentemente
do diretório de execução; não é necessário configurar variáveis no terminal.

1. Mantenha em `.env` a configuração `BASIC_AUTH=usuario:senha` com as credenciais
   aceitas pelo contêiner. Em `LayaCategoryClient.classify`, ajuste a URL
   `http://localhost:8000/predict` se necessário. O exemplo não utiliza API key.
2. Execute na raiz do repositório:

   ```bash
   uv run --script exemplos/classificar_categorias.py
   ```

   O arquivo declara suas dependências (`httpx`, `pydantic` e `python-dotenv`) para o `uv` preparar
   um ambiente isolado. O cliente requer Python 3.14+, como o projeto.

3. Edite `self.categories` em `LayaCategoryClient.__init__` para utilizar suas categorias e
   [documentos.yml](documentos.yml) para utilizar seus textos. Os placeholders `{razao_social}`,
   `{cnpj}` e `{endereco}` são preenchidos com os valores de `ClassificationExample.PLACEHOLDERS`.

Se `BASIC_AUTH` contiver vários pares separados por vírgula, o cliente usa o
primeiro par válido. Dois-pontos dentro da senha são preservados. A leitura não
expande referências a variáveis dentro dos valores do `.env`. Se não houver um
par válido, o script informa o problema antes de enviar qualquer requisição.
As credenciais são enviadas por `auth=httpx.BasicAuth(...)`, sem API key e sem
imprimi-las na saída. Se alterar as credenciais do servidor em `.env`, recrie o
serviço com `make up` para que o contêiner receba a nova configuração.

O script faz uma chamada a `POST /predict` para cada texto. O dicionário completo,
incluindo as descrições, vai em `questions.categoria.criteria`; a pergunta usa
`type: "choice"` para selecionar uma categoria. O resultado está em
`answers.categoria.choice`.

`classifier.classify(text)` usa as categorias padrão do cliente. Para uma chamada
específica, passe outro dicionário em `classifier.classify(text, categories)`.
Um dicionário vazio também utiliza as categorias padrão.

São exibidas a categoria escolhida, a confiança e as probabilidades retornadas pela
API. Os valores dependem da inferência real; não há resultados predefinidos no script.
O modelo fixado é `multilingual`, pois as mensagens estão em português. O servidor
precisa conseguir carregar esse checkpoint; a primeira chamada pode demorar mais.

O cliente usa timeout de 120 segundos. Respostas HTTP de erro, como `401`, `422` e
`503`, geram uma exceção por `raise_for_status()`. O código não converte falhas em
categorias fictícias.

## Executar os requests

Abra [requests.http](requests.http) com a extensão REST Client do VS Code.
Ajuste `baseUrl` se necessário e execute cada bloco separadamente. As rotas
protegidas usam HTTP Basic com `BASIC_AUTH` lido do `.env` na raiz; a senha
não fica no arquivo de requests.

| Exemplos | O que demonstram                                              |
| -------- | ------------------------------------------------------------- |
| 01–05    | Saúde, modelos, presets, idioma e preparação de email         |
| 06–07    | Categoria escolhida em uma lista ou em um mapa com descrições |
| 08–10    | `choice`, `score`, `noul`, conversas e objetos JSON           |
| 11–13    | Presets e lotes com configurações compartilhadas ou por item  |
| 14       | Compatibilidade SystemOne / TypeSafe                          |

`POST /email/state` apenas prepara o conteúdo. Para classificá-lo, copie o objeto
retornado para o campo `state` de um request a `/predict`.

## Exemplo com curl

Com o Docker já rodando, ajuste a URL e a chave diretamente no comando:

```bash
curl --fail-with-body --request POST 'http://localhost:8000/predict' \
  --header 'X-API-Key: substitua-pela-sua-chave' \
  --header 'Content-Type: application/json' \
  --data '{
    "state": "Preciso de ajuda com uma cobrança duplicada.",
    "model": "multilingual",
    "questions": {
      "categoria": {
        "type": "choice",
        "instructions": "Qual categoria descreve melhor esta mensagem?",
        "criteria": ["financeiro", "suporte_tecnico", "comercial", "outros"]
      }
    }
  }'
```

Para autenticação Bearer, substitua o header de chave por
`--header 'Authorization: Bearer substitua-pela-sua-chave'`.
Para HTTP Basic, substitua-o por `--user 'usuario:senha'`, usando uma conta
configurada em `BASIC_AUTH` no servidor. O script Python já utiliza HTTP Basic
e lê esse par diretamente do `.env`.

## Interpretar e adaptar

- `choice`: seleciona uma categoria; `outros` pode representar casos fora das opções
  específicas. A presença dessa opção não garante que o modelo reconheça todo caso desconhecido.
- `score`: retorna um valor esperado na escala iniciada em zero. Com quatro níveis,
  a escala vai de 0 a 3 e o resultado pode ser fracionário.
- `noul`: retorna um valor numérico para a decisão binária, não um booleano pronto.
- Em `/predict`, use `questions` **ou** `preset`. No lote, use `states` **ou** `items`.
  Para perguntas por item, siga o exemplo 13: um preset compartilhado tem precedência
  sobre as perguntas dos itens no comportamento atual da API.
- O lote preserva a ordem de entrada. Examine `results[i].ok` antes de acessar
  `results[i].result`; erros individuais aparecem em `results[i].error`.
