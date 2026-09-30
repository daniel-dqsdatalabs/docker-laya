"""Classifique mensagens em categorias usando a API HTTP deste projeto."""

from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from dotenv import dotenv_values
from pydantic import BaseModel


class CategoryPrediction(BaseModel):
    """Categoria escolhida e valores de confiança retornados pela API."""

    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class PredictionResponse(BaseModel):
    """Parte da resposta necessária para consultar a categoria prevista."""

    answers: dict[str, CategoryPrediction]


class LayaCategoryClient:
    """Adapter que transforma categorias e descrições em uma pergunta do tipo choice."""

    def __init__(self, client: httpx.Client, auth: httpx.BasicAuth) -> None:
        """Reutilize a conexão HTTP e a autenticação Basic entre as mensagens."""
        self._client = client
        self._auth = auth
        # Laya truncates each option to ~26 tokens when there are nine of them, so descriptions stay short
        # and lead with the words that tell categories apart.
        self.categories: dict[str, str] = {
            "Certidões": "Certidão (nada consta, negativa, criminal, cível, de débitos), certifica-se.",
            "Ações Judiciais": "Ação ou processo em andamento: autor, réu, valor da causa, petição, sentença.",
            "Endividamento Bancário": "Cédula de crédito bancário, empréstimo, financiamento, parcelas, juros.",
            "Endividamento Fornecedor": "Dívida com fornecedor: duplicatas, boletos, notas fiscais em aberto.",
            "Endividamento Trabalhista": "Dívida com empregados: rescisão, salários atrasados, FGTS.",
            "Bens dos Sócios": "Bens pessoais dos sócios: imóveis, veículos em nome de pessoa física.",
            "Bens Empresa": "Bens da empresa: imóveis, veículos, máquinas da pessoa jurídica.",
            "Extratos Bancários": "Extrato bancário: agência, conta corrente, período, débito, crédito, saldo.",
            "Passivo Tributário": "Dívida de impostos: débito fiscal, dívida ativa, auto de infração, DARF.",
        }

    def classify(self, text: str, categories: dict[str, str] | None = None) -> CategoryPrediction:
        """Classifique com o dicionário informado ou com as categorias padrão do cliente."""
        text = text.lower().strip()
        categories = categories if categories else self.categories

        response = self._client.post(
            "http://localhost:8000/predict",
            auth=self._auth,
            json=self._payload(text, categories),
            timeout=120.0,
        )
        response.raise_for_status()
        prediction = PredictionResponse.model_validate(response.json()).answers["categoria"]
        return prediction

    @staticmethod
    def _payload(text: str, categories: dict[str, str]) -> dict[str, Any]:
        """Envie cada categoria e sua descrição no mapa de critérios da pergunta."""
        return {
            "state": text,
            "model": "multilingual",
            "questions": {
                "categoria": {
                    "type": "choice",
                    "instructions": "Qual a categoria que melhor representa o texto do documento?",
                    "criteria": categories,
                }
            },
        }


class SampleDocument(BaseModel):
    """Documento de exemplo com o nome do arquivo e o texto extraído."""

    filename: str
    content: str


class SampleDocumentRepository:
    """Repository que lê os documentos de exemplo do YAML e preenche os placeholders."""

    def __init__(self, source: Path, placeholders: dict[str, str]) -> None:
        """Guarde o arquivo de origem e os valores que substituem os placeholders."""
        self._source = source
        self._placeholders = placeholders

    def load(self) -> list[SampleDocument]:
        """Leia todos os documentos, falhando se o texto usar um placeholder desconhecido."""
        raw = yaml.safe_load(self._source.read_text(encoding="utf-8"))
        return [self._render(SampleDocument.model_validate(item)) for item in raw["documentos"]]

    def _render(self, document: SampleDocument) -> SampleDocument:
        """Substitua os placeholders do conteúdo pelos valores fictícios."""
        return document.model_copy(update={"content": document.content.format_map(self._placeholders)})


class ClassificationExample:
    """Exemplo executável usando o dicionário de categorias documentais do cliente."""

    DOCUMENTS_FILE = Path(__file__).with_name("documentos.yml")
    PLACEHOLDERS: dict[str, str] = {
        "cnpj": "76.674.138/0001-38",
        "razao_social": "Vanguarda Comércio S.A.",
        "endereco": "Rua das Flores, 123, Bairro Jardim, Cidade Recife, Estado Pernambuco, CEP 50000-000",
    }

    @classmethod
    def run(cls) -> None:
        """Classifique as mensagens assumindo que a API já está rodando no Docker."""
        auth = cls._authentication(Path(__file__).resolve().parents[1] / ".env")
        documents = SampleDocumentRepository(cls.DOCUMENTS_FILE, cls.PLACEHOLDERS).load()
        with httpx.Client() as client:
            classifier = LayaCategoryClient(client, auth)
            for document in documents:
                prediction = classifier.classify(document.content)
                print(f"\nDocumento: {document.filename}")
                print(f"Categoria: {prediction.choice}")
                print(f"Confiança: {prediction.confidence:.4f}")
                print(f"Probabilidades: {prediction.probabilities}")
                print("----------------------------------------------------")

    @staticmethod
    def _authentication(env_file: Path) -> httpx.BasicAuth:
        """Leia o primeiro par de BASIC_AUTH no .env, preservando dois-pontos na senha."""
        configuration = dotenv_values(env_file, interpolate=False)
        for credentials in (configuration.get("BASIC_AUTH") or "").split(","):
            if ":" in credentials:
                username, password = credentials.split(":", 1)
                return httpx.BasicAuth(username.strip(), password.strip())
        raise ValueError(f"Configure BASIC_AUTH no formato usuario:senha em {env_file}.")


if __name__ == "__main__":
    ClassificationExample.run()
