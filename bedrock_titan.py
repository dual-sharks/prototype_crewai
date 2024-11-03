import boto3
import json
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from pydantic import PrivateAttr

AWS_ACCESS_KEY = 'AKIAR5LKRLXTQ3ZTRMM3'
AWS_SECRET_KEY = '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'

class BedrockTitanLLM(LLM):
    """
    Bedrock Titan LLM wrapper for LangChain.
    """
    model_name: str = "amazon.titan-tg1-large"
    _client: Any = PrivateAttr() 
    def __init__(self, model_name: str = "amazon.titan-tg1-large", region_name="us-east-1"):
        """
        Initialize the BedrockTitanLLM.
        """
        super().__init__() 
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region_name,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        self.model_name = model_name

    def _call(self, prompt: str, stop=None) -> str:
        print("Debug: Received prompt:", prompt)
        try:
            response = self._client.invoke_model(
                modelId=self.model_name,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": prompt}),
            )
            response_body = json.loads(response['body'].read().decode('utf-8'))
            print("Debug: Response body:", response_body)
            return response_body['results'][0]['outputText']
        except Exception as e:
            print("Debug: Error occurred in _call method:", str(e))
            raise


    def _identifying_params(self) -> Mapping[str, Any]:
        params = {"model_name": self.model_name}
        if params is None or not isinstance(params, dict):
            raise ValueError("Identifying params must be a non-empty dictionary.")
        print("Debug: Identifying params:", params)  # Add a print statement
        return params


    @property
    def _llm_type(self) -> str:
        """
        Get the type of the LLM.
        """
        return "bedrock-titan"

    def dict(self, **kwargs: Any) -> dict:
        """Return a dictionary of the LLM."""
        starter_dict = self._identifying_params()
        starter_dict["_type"] = self._llm_type
        return starter_dict
