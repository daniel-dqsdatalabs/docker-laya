# docker-laya

API de predição com FastAPI e [Laya](https://huggingface.co/convaiinnovations/laya),
executável com Docker ou Python. Classifica textos, objetos JSON e conversas por
meio de perguntas com opções, escalas ou decisões binárias.

O projeto permite escolher um checkpoint por requisição ou delegar o roteamento
à biblioteca. Também oferece processamento em lote, presets de perguntas e
compatibilidade com SystemOne / TypeSafe.

## Início rápido com Docker Compose

Execute os comandos na raiz do repositório. É necessário ter Docker com Compose.

1. Crie o arquivo de configuração, caso ele ainda não exista:

   ```bash
   test -f .env || cp .env.example .env
   ```

2. Edite `.env`. Para começar com os exemplos em português, configure:

   ```dotenv
   PORT=8000
   API_KEYS=substitua-por-uma-chave-propria
   BASIC_AUTH=
   MODELS=multilingual
   MODEL_SUBFOLDER=multilingual
   MAX_BULK_ITEMS=256
   DEVICE=cpu
   ```

   Os valores de autenticação de `.env.example` são demonstrativos. Use uma chave
   própria. Os exemplos prontos autenticam com HTTP Basic: defina
   `BASIC_AUTH=usuario:senha` antes de executá-los.

3. Construa e inicie o serviço:

   ```bash
   make up
   ```

   O Compose constrói `laya-api:latest` com o código local. Os checkpoints
   configurados são carregados antes de a aplicação começar a atender. Na primeira
   execução, o servidor pode precisar baixá-los. Acompanhe com `make logs`.

4. Confira a saúde e abra a documentação:

   ```bash
   curl --fail-with-body http://localhost:8000/healthz
   ```

   Verifique `model_loaded` e `models` na resposta. O endpoint retorna HTTP 200
   quando atendido; o campo `model_loaded` informa se há um router disponível.
   Se alterar `PORT`, ajuste também as URLs utilizadas nos exemplos.

| Recurso             | Endereço padrão                      |
| ------------------- | ------------------------------------ |
| Swagger UI          | <http://localhost:8000/docs>         |
| ReDoc               | <http://localhost:8000/redoc>        |
| OpenAPI em execução | <http://localhost:8000/openapi.json> |
| Contrato versionado | [openapi.json](openapi.json)         |

## Exemplos prontos

A pasta [exemplos/](exemplos/README.md) contém 14 requisições HTTP e um cliente
Python para selecionar uma categoria em um dicionário de nomes e descrições.
Os exemplos só chamam a API; não carregam checkpoints.

| Arquivo                                                         | Conteúdo                                                                          |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [requests.http](exemplos/requests.http)                         | Saúde, modelos, presets, predições, lotes e SystemOne                             |
| [classificar_categorias.py](exemplos/classificar_categorias.py) | Cliente HTTP tipado que classifica três mensagens usando categorias configuráveis |
| [Guia dos exemplos](exemplos/README.md)                         | Adaptação das categorias e interpretação dos resultados                           |

### Antes de executar

1. No `.env` da raiz, defina a conta usada pelos exemplos:

   ```dotenv
   BASIC_AUTH=usuario:senha
   PORT=8000
   MODELS=multilingual
   ```

   Para [requests.http](exemplos/requests.http), use um único par `usuario:senha`.
   Se houver vários pares separados por vírgula, o cliente Python usa só o primeiro.
   Dois-pontos dentro da senha são preservados, e a leitura não expande variáveis.
   Os exemplos não usam `API_KEYS`.

2. Inicie a API. `make up` e `make run` leem esse `.env`:

   ```bash
   make up
   ```

   Sem Docker:

   ```bash
   make run
   ```

3. Confira o health check público. O caminho é `/healthz`, não `/health`:

   ```bash
   curl --fail-with-body http://localhost:8000/healthz
   ```

   Espere `"status":"ok"` e `"model_loaded":true` antes de classificar textos.
   Se mudar `PORT` ou `BASIC_AUTH`, reinicie o processo. No Compose, recrie o
   serviço com `make up`. Se a porta não for `8000`, use a mesma URL nos exemplos.

### Requests HTTP

1. Instale a extensão REST Client no VS Code ou no Cursor.
2. Abra [exemplos/requests.http](exemplos/requests.http).
3. O arquivo aponta `@baseUrl` para `http://localhost:8000` e lê `BASIC_AUTH` do
   `.env` da raiz. A senha não fica no arquivo. Ajuste `@baseUrl` se a porta mudar.
4. Execute um bloco por vez, pelo link **Send Request** acima dele.

O bloco 01 (`GET /healthz`) não envia credenciais. Os demais enviam
`Authorization: Basic` com o par do `.env`. Comece pelos blocos 06 e 07 para
escolher uma categoria em uma lista ou em um mapa com descrições.

### Cliente Python

Na raiz do repositório, com a API já respondendo em `/healthz`:

```bash
uv run --script exemplos/classificar_categorias.py
```

O arquivo declara `httpx`, `pydantic` e `python-dotenv`. O `uv` prepara um ambiente
separado do servidor e exige Python 3.14 ou superior. Não é preciso exportar
variáveis no terminal nem instalar o Laya no cliente.

O script localiza o `.env` a partir do próprio arquivo, lê o primeiro par válido
de `BASIC_AUTH` e chama `POST http://localhost:8000/predict` para cada texto.
A saída mostra a categoria, a confiança e as probabilidades. Para outros textos
ou categorias, edite `ClassificationExample.TEXTS` e `self.categories`. A URL
fica na chamada `self._client.post`, dentro de `LayaCategoryClient.classify`.

## Endpoints e autenticação

| Método | Caminho         | Autenticação configurada | Finalidade                                                       |
| ------ | --------------- | ------------------------ | ---------------------------------------------------------------- |
| `GET`  | `/healthz`      | Não exigida              | Saúde, dispositivo, métodos de autenticação e modelos residentes |
| `GET`  | `/models`       | Exigida                  | Checkpoints disponíveis e carregados                             |
| `GET`  | `/presets`      | Exigida                  | Perguntas dos presets disponíveis                                |
| `POST` | `/detect`       | Exigida                  | Detecção de escrita e idioma do estado enviado                   |
| `POST` | `/email/state`  | Exigida                  | Limpeza e estruturação de email para posterior predição          |
| `POST` | `/predict`      | Exigida                  | Predição sobre um estado                                         |
| `POST` | `/predict/bulk` | Exigida                  | Predições em lote, com resultado ou erro por item                |
| `POST` | `/v1/systemone` | Exigida                  | Predição no formato SystemOne / TypeSafe                         |
| `POST` | `/systemone`    | Exigida                  | Alias de compatibilidade, omitido do OpenAPI                     |

A aplicação exige autenticação quando `API_KEYS` ou `BASIC_AUTH` está configurado.
Qualquer método habilitado pode autorizar a chamada; não é necessário combinar
API key e Basic. `/healthz`, `/docs`, `/redoc` e `/openapi.json` são públicos.

| Método de autenticação  | Opção para curl                                             |
| ----------------------- | ----------------------------------------------------------- |
| Chave no header         | `--header 'X-API-Key: substitua-pela-sua-chave'`            |
| Chave como bearer token | `--header 'Authorization: Bearer substitua-pela-sua-chave'` |
| HTTP Basic              | `--user 'usuario:senha'`                                    |

Credenciais ausentes ou inválidas resultam em HTTP 401. A aplicação aceita acesso
sem autenticação quando ambos os conjuntos de credenciais estão vazios, mas o
**Compose atual aplica `API_KEYS=change-me` quando a variável está ausente ou vazia**.
Para desabilitar autenticação nessa forma de execução, é necessário ajustar esse
padrão no [docker-compose.yml](docker-compose.yml), além de manter `BASIC_AUTH` vazio.

## Predições e categorias

Envie `state` e exatamente uma fonte de perguntas: `questions` ou `preset`.
`state` aceita texto, um objeto JSON ou uma lista de turnos de conversa.

Este exemplo escolhe uma categoria entre as opções fornecidas. Com o Docker já
rodando, ajuste a URL e a chave diretamente no comando:

```bash
curl --fail-with-body --request POST 'http://localhost:8000/predict' \
  --header 'X-API-Key: substitua-pela-sua-chave' \
  --header 'Content-Type: application/json' \
  --data '{
    "state": "Minha fatura veio com uma cobrança duplicada. Preciso de estorno.",
    "model": "multilingual",
    "questions": {
      "categoria": {
        "type": "choice",
        "instructions": "Qual categoria descreve melhor esta mensagem?",
        "criteria": ["financeiro", "suporte_tecnico", "comercial", "cancelamento", "outros"]
      }
    }
  }'
```

Também é possível usar um mapa em `criteria`, associando cada categoria a uma
descrição. Isso permite explicar a diferença entre opções semelhantes, como no
exemplo 07 de [requests.http](exemplos/requests.http).

| Tipo de pergunta | Critérios                                        | Resultado principal                                |
| ---------------- | ------------------------------------------------ | -------------------------------------------------- |
| `choice`         | Lista de opções ou mapa de rótulos e descrições  | `choice`: opção escolhida                          |
| `score`          | Lista ordenada do menor para o maior nível       | `score`: valor esperado na escala iniciada em zero |
| `noul`           | Mapa opcional com descrições de `false` e `true` | `noul`: valor numérico da decisão binária          |

Um `score` pode ser fracionário; com quatro níveis, a escala vai de 0 a 3. O valor
`noul` não é um booleano pronto. Os critérios podem conter valores JSON, conforme
os schemas de cada tipo de pergunta.

Na resposta nativa de `/predict`, consulte:

| Campo                                            | Significado                                                    |
| ------------------------------------------------ | -------------------------------------------------------------- |
| `answers.categoria.choice`                       | Categoria selecionada no exemplo acima                         |
| `answers.categoria.confidence` e `probabilities` | Confiança e probabilidades retornadas pelo modelo              |
| `answers.categoria.action.act_probability`       | Probabilidade de o modelo ter tomado uma ação                  |
| `usage.input_tokens` e `usage.output_tokens`     | Contagens de tokens                                            |
| `model` e `routing`                              | Modelo informado pelo Laya e metadados opcionais de roteamento |

Os valores dependem da inferência. A resposta nativa preserva campos adicionais
retornados pela biblioteca e omite campos opcionais nulos.

### Modelos e presets

Os nomes aceitos em `model` na API nativa são `english`, `multilingual` e
`typed-decisions`. Sem esse campo, o Laya decide o roteamento. `MODELS` define os
checkpoints pré-carregados na inicialização; não restringe os modelos aceitos nas
requisições. Uma chamada pode exigir o carregamento de outro checkpoint.

Os presets são `triage`, `email`, `guard`, `moderation` e `router`. Consulte suas
perguntas em `GET /presets`. Para utilizá-los, envie `preset` no lugar de `questions`:

```json
{
  "state": "I was billed twice this month. Please refund the duplicate charge.",
  "preset": "triage"
}
```

`POST /email/state` somente prepara o conteúdo do email. Use o objeto retornado
como `state` em uma chamada posterior a `/predict` para classificá-lo.

### Processamento em lote

`POST /predict/bulk` aceita dois formatos exclusivos:

1. `states`: lista de estados com `questions` ou `preset` compartilhado.
2. `items`: lista de objetos com `state` e, opcionalmente, `questions` e `model`
   específicos para cada item.

As listas precisam conter pelo menos um item. O limite configurado em
`MAX_BULK_ITEMS` se aplica a ambos os formatos.

O resultado mantém a ordem da entrada e contém `count` e `results`. Cada item
bem-sucedido possui `ok: true` e `result`; cada falha possui `ok: false` e `error`.
Uma falha individual não descarta os demais resultados.

No comportamento atual, um `preset` compartilhado tem precedência sobre as
perguntas de cada item. Para personalizar perguntas por item, use o formato do
exemplo 13, sem preset compartilhado. Um mapa de perguntas vazio no item usa as
perguntas compartilhadas como alternativa.

O processamento é sequencial. O lote mantém o lock de inferência até terminar,
impedindo que outras predições se intercalem nesse mesmo processo.

### Compatibilidade SystemOne / TypeSafe

Use `POST /v1/systemone` com `state`, `model` e `questions`. O alias `/systemone`
continua disponível para compatibilidade.

Os nomes explícitos de modelo incluem `laya-english`, `laya-multilingual` e
`laya-typed-decisions`, além dos nomes da API nativa. O adaptador também normaliza
os prefixos `typesafe/` e `laya/`. Prefira esses nomes conhecidos: a compatibilidade
mantém regras de fallback para nomes desconhecidos, em vez de rejeitá-los.

A resposta inclui `id`, `model`, `provider`, `answers` e `usage`. O campo `model`
preserva o identificador enviado pelo cliente. As respostas SystemOne não incluem
`action`; respostas `noul` também não incluem `confidence`. `session_id` e `user`
são aceitos como metadados opcionais, mas não são usados pelo serviço de inferência.

### Erros esperados

| Situação                                                           | Comportamento                                      |
| ------------------------------------------------------------------ | -------------------------------------------------- |
| Credenciais ausentes ou inválidas                                  | HTTP 401 com os métodos de autenticação aceitos    |
| Corpo inválido, modelo nativo desconhecido ou lote acima do limite | HTTP 422                                           |
| Router indisponível em uma operação que depende dele               | HTTP 503                                           |
| Falha de inferência ou de validação em um item do lote             | Erro dentro de `results`, mantendo os demais itens |
| Falha de inferência em uma predição individual                     | Propagada como erro do servidor                    |

## Configuração

[Settings](app/config.py) lê as variáveis de ambiente uma vez por instância da
aplicação. Na inicialização sem configuração injetada, o processo também carrega
o `.env` da raiz, sem sobrescrever variáveis já definidas. A execução direta e o
Compose têm alguns padrões diferentes:

| Variável          | Padrão da aplicação      | Padrão do Compose | Finalidade                                                                |
| ----------------- | ------------------------ | ----------------- | ------------------------------------------------------------------------- |
| `API_KEYS`        | Vazio                    | `change-me`       | Chaves separadas por vírgula                                              |
| `BASIC_AUTH`      | Vazio                    | Vazio             | Pares `usuario:senha` separados por vírgula                               |
| `MAX_BULK_ITEMS`  | Sem limite               | Sem limite        | Máximo de itens por lote; vazio ou `0` desabilita o limite                |
| `MODELS`          | `english`                | `multilingual`    | Checkpoints pré-carregados, separados por vírgula                         |
| `MODEL_ID`        | `convaiinnovations/laya` | Mesmo valor       | Repositório ou caminho dos checkpoints                                    |
| `MODEL_SUBFOLDER` | Sem sobrescrita          | `multilingual`    | Subpasta usada para a entrada `english` quando há sobrescrita das origens |
| `DEVICE`          | `cpu`                    | `cpu`             | Dispositivo repassado ao router do Laya                                   |

A configuração do Compose usa `${VAR:-padrão}`: valores ausentes **ou vazios**
recebem o padrão. Por isso, `MODEL_SUBFOLDER=` em `.env` ainda resulta em
`multilingual` dentro do contêiner. No adaptador, essa variável altera a origem da
entrada `english`; as outras entradas usam as subpastas com seus próprios nomes.
Para restaurar a ausência de sobrescrita, ajuste o padrão no Compose.

O arquivo `.env.example` contém valores demonstrativos, incluindo `MODELS=english`
e `MAX_BULK_ITEMS=256`; copiá-lo não equivale a usar todos os padrões da tabela.
Revise os valores conforme o modelo que deseja carregar.

| Variável de execução | Comportamento                                                                                        |
| -------------------- | ---------------------------------------------------------------------------------------------------- |
| `PORT`               | Padrão `8000` no Docker e no Makefile; o Compose publica essa mesma porta no host e no contêiner     |
| `HF_HOME`            | Padrão `/data/hf` na imagem; o volume `hf-cache` do Compose persiste esse diretório                  |
| `HF_TOKEN`           | Lido do `.env` na execução local e injetado pelo Compose; autentica os downloads no Hugging Face Hub |
| `PRELOAD_MODEL`      | Argumento de construção; `1` baixa os checkpoints durante a criação da imagem                        |

`DEVICE` não instala suporte ao dispositivo escolhido. O projeto configura o
índice de wheels de CPU do PyTorch; habilitar CUDA requer uma instalação e uma
imagem compatíveis, além de alterar a variável.

## Desenvolvimento local

O projeto requer **Python 3.14 ou superior** e utiliza **uv** para dependências.
Instale as versões registradas no lockfile:

```bash
uv sync --locked
```

Para carregar explicitamente o `.env` e iniciar o servidor local:

```bash
uv run --env-file .env uvicorn app.main:app --reload --port 8000
```

Nesse comando, a porta é o argumento `--port`; ajuste-o se necessário.
`make run` também carrega o `.env` da raiz: a aplicação lê esse arquivo ao subir,
sem sobrescrever variáveis que já existam no processo. Para alterar a porta, use
`make run PORT=8080`.

A aplicação carrega os checkpoints durante o lifespan. Um erro de carregamento
interrompe a inicialização. Cada instância possui seu próprio router, lock e cache
de presets; o lock é compartilhado entre predições nativas, lotes e SystemOne.
Processos diferentes possuem recursos e locks independentes.

### Organização do código

```text
app/
├── main.py                  # Entrada ASGI: app.main:app
├── application.py           # Factory, composição e aplicação FastAPI
├── api.py                   # Rotas HTTP e autenticação das rotas protegidas
├── authentication.py        # Strategies para API key e HTTP Basic
├── backend.py               # Adapter e contratos tipados da biblioteca Laya
├── config.py                # Configuração imutável por instância
├── schemas.py               # Validação de requisições e respostas com Pydantic
├── openapi.py               # Geração e cache do schema com autenticação documentada
└── services/
    ├── runtime.py           # Ciclo de vida do router e lock de inferência
    ├── questions.py         # Serialização de perguntas e resolução de presets
    ├── prediction.py        # Predição individual e processamento em lote
    ├── metadata.py          # Saúde e inventário de modelos
    └── systemone.py         # Adapter de compatibilidade SystemOne
exemplos/                    # Requisições e cliente Python de classificação
tests/                       # Testes dos serviços e contratos HTTP
```

As rotas delegam o trabalho aos serviços. A composição usa Factory; autenticação
usa Strategy; os limites com Laya e SystemOne usam Adapter. O Laya é importado sob
demanda, permitindo gerar o OpenAPI sem carregar o runtime de modelos.

### Qualidade, testes e OpenAPI

```bash
make check
make test
make openapi
```

`make check` executa Ruff, conferência de formatação, mypy estrito e Pyright.
Ruff verifica também as docstrings. Mypy e Pyright estão configurados para `app/`
e `tests/`; para conferir o cliente de exemplo, execute adicionalmente:

```bash
uv run mypy exemplos
uv run pyright exemplos
```

`make test` executa testes de configuração, autenticação, serviços, concorrência,
validação e contratos HTTP com backend simulado. Esses testes não baixam checkpoints
nem comprovam a qualidade das predições de um modelo real. O ambiente instalado
por `uv sync`, contudo, inclui as dependências do servidor.

`make openapi` regenera [openapi.json](openapi.json). O teste de contrato compara
esse arquivo com o schema produzido pela aplicação. Ao alterar schemas ou
informações de documentação expostas pela API, regenere o arquivo e execute os
testes. O contrato pode ser usado por ferramentas de geração de clientes.

| Comando                               | Finalidade                                               |
| ------------------------------------- | -------------------------------------------------------- |
| `make run`                            | Servidor local com recarga automática                    |
| `make format` / `make fix`            | Formatação / correções automáticas de lint com Ruff      |
| `make check` / `make test`            | Verificações estáticas / testes automatizados            |
| `make openapi`                        | Atualização do contrato OpenAPI                          |
| `make up` / `make down` / `make logs` | Construção e inicialização / parada / logs do Compose    |
| `make build`                          | Construção da imagem local `laya-api:latest`             |
| `make build-model MODEL=multilingual` | Construção com o checkpoint escolhido incluído na imagem |

## Imagens e operação

O [Dockerfile](Dockerfile) executa o serviço como `appuser`, UID `10001`, e define
um healthcheck que consulta `/healthz`. Se utilizar um diretório do host como
cache, ele precisa permitir escrita por esse usuário.

Por padrão, a construção não inclui os checkpoints. Eles são carregados na
inicialização a partir do cache ou baixados quando necessários. Para incluí-los
na imagem local:

```bash
make build-model MODEL=multilingual
```

`make up` também faz uma construção. Para incluir modelos nessa construção pelo
Compose, configure `PRELOAD_MODEL=1` e `MODELS` em `.env` antes de executá-lo.
A construção precisa acessar os arquivos dos modelos. Incluir um checkpoint não
elimina o tempo de carregamento em memória nem garante operação offline para
requisições que precisem de outros checkpoints.

O workflow configura imagens para `linux/amd64` e `linux/arm64`, publicadas em
`ghcr.io/<proprietário-do-repositório>/laya`. As variantes previstas são:

| Variante          | Checkpoints incluídos na construção |
| ----------------- | ----------------------------------- |
| `latest`          | Nenhum                              |
| `english`         | `english`                           |
| `multilingual`    | `multilingual`                      |
| `typed-decisions` | `typed-decisions`                   |
| `all`             | Os três checkpoints                 |

O workflow também define tags de versão nas publicações de releases. Para executar
as alterações deste checkout, utilize a construção local: uma imagem publicada
separadamente não incorpora automaticamente mudanças locais.

Não há garantia fixa de tamanho de imagem ou tempo por predição. Esses valores
dependem das dependências, checkpoints, hardware e conteúdo processado.

### Integração contínua

O [workflow de publicação](.github/workflows/publish.yml) possui estas verificações:

| Execução                                 | Verificações configuradas                                                             |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| Job `lint`                               | Lockfile, Ruff, formatação e geração do OpenAPI                                       |
| Pull requests                            | Construção de imagem e importação da aplicação para gerar o OpenAPI no contêiner      |
| Agendamento, execução manual e tags `v*` | Construção com checkpoint e chamadas reais a `/healthz`, `/predict` e `/v1/systemone` |

O workflow atual ainda não chama `make test`, mypy ou Pyright. A publicação depende
do job `lint`; o teste de inferência real é um job separado, não uma condição
obrigatória de todas as publicações.

## Licença

Distribuído sob a [licença MIT](LICENSE). O projeto original está disponível em
[chneau/docker-laya](https://github.com/chneau/docker-laya).
