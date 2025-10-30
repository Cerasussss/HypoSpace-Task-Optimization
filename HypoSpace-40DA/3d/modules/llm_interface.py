import traceback
import random
import requests
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import re


def _extract_text(resp) -> str:
    # 1) Responses API convenience
    t = getattr(resp, "output_text", None)
    if t:
        return t

    # 2) Responses API: walk output -> message -> content
    try:
        pieces = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    ctype = getattr(c, "type", None) or (isinstance(c, dict) and c.get("type"))
                    if ctype in ("output_text", "text"):
                        # pydantic object or dict
                        text = getattr(c, "text", None) if hasattr(c, "text") else c.get("text")
                        if text:
                            pieces.append(text)
        if pieces:
            return "".join(pieces)
    except Exception:
        pass

    # 3) Chat Completions fallback (if you switch endpoints)
    try:
        return resp.choices[0].message.content
    except Exception:
        pass

    return str(resp)


def _extract_usage(resp):
    u = getattr(resp, "usage", None)
    if not u:
        return 0, 0, 0
    # Responses API names
    input_tokens = getattr(u, "input_tokens", getattr(u, "prompt_tokens", 0))
    output_tokens = getattr(u, "output_tokens", getattr(u, "completion_tokens", 0))
    total_tokens = getattr(u, "total_tokens", input_tokens + output_tokens)
    return input_tokens, output_tokens, total_tokens


class LLMInterface(ABC):
    """Abstract interface for LLM interaction."""

    @abstractmethod
    def query(self, prompt: str) -> str:
        """
        Query the LLM with a prompt and return response.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM's response as a string
        """
        pass

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        """Query the LLM and return response with usage stats."""
        # Default implementation for backward compatibility
        return {
            'response': self.query(prompt),
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'cost': 0.0
        }

    @abstractmethod
    def get_name(self) -> str:
        """Get the name/identifier of the LLM."""
        pass

    def get_model_pricing(self) -> Dict[str, float]:
        """Get pricing per 1M tokens for this model."""
        # Default pricing (can be overridden by subclasses)
        return {'input': 0.0, 'output': 0.0}

    def reset(self):
        """Reset any internal state (optional)."""
        pass


class OpenRouterLLM(LLMInterface):
    """
    OpenRouter API interface for various LLM models.

    OpenRouter provides access to multiple models through a single API.
    """

    def __init__(
            self,
            model: str = "anthropic/claude-3.5-sonnet",
            api_key: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 40960,
            base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize OpenRouter LLM interface.

        Args:
            model: Model identifier (e.g., "anthropic/claude-3.5-sonnet", "openai/gpt-4")
            api_key: OpenRouter API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            base_url: OpenRouter API base URL
        """
        if not api_key:
            raise ValueError("OpenRouter API key is required")

        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def query(self, prompt: str) -> str:
        """Query OpenRouter API."""
        result = self.query_with_usage(prompt)
        return result['response']

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        """Query OpenRouter API with usage tracking."""
        try:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert in 3D structure analysis and reconstruction."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }

            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()

            # Extract usage information
            usage = result.get('usage', {})
            usage_data = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }

            # Calculate cost based on model pricing
            pricing = self.get_model_pricing()
            cost = (usage_data['prompt_tokens'] * pricing['input'] +
                    usage_data['completion_tokens'] * pricing['output']) / 1_000_000

            return {
                'response': result['choices'][0]['message']['content'],
                'usage': usage_data,
                'cost': cost
            }

        except requests.exceptions.RequestException as e:
            return {
                'response': f"Error querying OpenRouter: {str(e)}",
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'cost': 0.0
            }
        except (KeyError, IndexError) as e:
            return {
                'response': f"Error parsing OpenRouter response: {str(e)}",
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'cost': 0.0
            }

    def get_name(self) -> str:
        """Get the model name."""
        return f"OpenRouter({self.model})"

    def get_model_pricing(self) -> Dict[str, float]:
        """Get pricing per 1M tokens for common models."""
        # Pricing in dollars per 1M tokens
        pricing_map = {
            'anthropic/claude-3.5-sonnet': {'input': 3.0, 'output': 15.0},
            'anthropic/claude-3-opus': {'input': 15.0, 'output': 75.0},
            'openai/gpt-4o': {'input': 2.5, 'output': 10.0},
            'meta-llama/llama-3.3-70b-instruct': {'input': 0.038, 'output': 0.12},
            'google/gemini-2.5-pro': {'input': 1.25, 'output': 10.0},
            'deepseek/deepseek-r1': {'input': 0.4, 'output': 2},
        }
        return pricing_map.get(self.model, {'input': 1.0, 'output': 1.0})


class OpenAILLM(LLMInterface):
    """
    OpenAI API interface for GPT models.

    Requires openai package and API key.
    """

    def __init__(
            self,
            model: str = "gpt-4",
            api_key: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 40960
    ):
        """
        Initialize OpenAI LLM interface.

        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (uses environment variable if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")

        self.client = openai.OpenAI(api_key=api_key)

    def query(self, prompt: str) -> str:
        """Query OpenAI API."""
        result = self.query_with_usage(prompt)
        return result['response']

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": "You are an expert in 3D structure analysis and reconstruction."},
                    {"role": "user", "content": prompt},
                ],
                reasoning={"effort": "medium"},
                max_output_tokens=self.max_tokens
            )

            text = _extract_text(resp)
            in_tok, out_tok, tot_tok = _extract_usage(resp)

            pricing = self.get_model_pricing()
            cost = (in_tok * pricing['input'] + out_tok * pricing['output']) / 1_000_000

            return {
                "response": text,
                "usage": {
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                    "total_tokens": tot_tok,
                },
                "cost": cost,
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "response": f"Error querying OpenAI: {str(e)}",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost": 0.0,
            }

    def get_name(self) -> str:
        """Get the model name."""
        return f"OpenAI({self.model})"

    def get_model_pricing(self) -> Dict[str, float]:
        """Get pricing per 1M tokens for OpenAI models."""
        # Pricing in dollars per 1M tokens
        pricing_map = {
            'gpt-4o': {'input': 2.5, 'output': 10.0},
            'gpt-4o-mini': {'input': 0.15, 'output': 0.6},
            'gpt-5': {'input': 1.25, 'output': 10.0}
        }
        return pricing_map.get(self.model, {'input': 10.0, 'output': 30.0})


class AnthropicLLM(LLMInterface):
    """
    Anthropic Claude API interface.

    Requires anthropic package and API key.
    """

    def __init__(
            self,
            model: str = "claude-3-opus-20240229",
            api_key: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 500
    ):
        """
        Initialize Anthropic LLM interface.

        Args:
            model: Anthropic model to use
            api_key: Anthropic API key (uses environment variable if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic package: pip install anthropic")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = anthropic.Anthropic(api_key=api_key)

    def query(self, prompt: str) -> str:
        """Query Anthropic API."""
        result = self.query_with_usage(prompt)
        return result['response']

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        """Query Anthropic API with usage tracking."""
        try:
            response = self.client.messages.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract usage information
            usage = {
                'prompt_tokens': response.usage.input_tokens if hasattr(response, 'usage') else 0,
                'completion_tokens': response.usage.output_tokens if hasattr(response, 'usage') else 0,
                'total_tokens': (response.usage.input_tokens + response.usage.output_tokens) if hasattr(response,
                                                                                                        'usage') else 0
            }

            # Calculate cost
            pricing = self.get_model_pricing()
            cost = (usage['prompt_tokens'] * pricing['input'] +
                    usage['completion_tokens'] * pricing['output']) / 1_000_000

            return {
                'response': response.content[0].text,
                'usage': usage,
                'cost': cost
            }
        except Exception as e:
            return {
                'response': f"Error querying Anthropic: {str(e)}",
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'cost': 0.0
            }

    def get_name(self) -> str:
        """Get the model name."""
        return f"Anthropic({self.model})"

    def get_model_pricing(self) -> Dict[str, float]:
        """Get pricing per 1M tokens for Anthropic models."""
        # Pricing in dollars per 1M tokens
        pricing_map = {
            'claude-3-opus-20240229': {'input': 15.0, 'output': 75.0},
            'claude-3-sonnet-20240229': {'input': 3.0, 'output': 15.0},
            'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
            'claude-3.5-sonnet-20241022': {'input': 3.0, 'output': 15.0},
        }
        return pricing_map.get(self.model, {'input': 3.0, 'output': 15.0})


# Local LLM
class QwenLocalLLM(LLMInterface):
    """
    Local Qwen model interface using Hugging Face Transformers.

    Requires transformers package and local Qwen model.
    """

    def __init__(
            self,
            model_path: str = "Qwen/Qwen2-7B-Instruct",
            temperature: float = 0.7,
            max_tokens: int = 512,
            device: str = "auto"
    ):
        """
        Initialize Local Qwen LLM interface.

        Args:
            model_path: Path to the local Qwen model or Hugging Face model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            device: Device to run the model on ("cpu", "cuda", "auto", etc.)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers package: pip install transformers")

        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device

        print(f"Loading Qwen model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device
        )
        print("Model loaded successfully.")

    def query(self, prompt: str) -> str:
        """Query local Qwen model."""
        result = self.query_with_usage(prompt)
        return result['response']

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        """Query local Qwen model with usage tracking."""
        try:
            # Prepare input for Qwen model
            messages = [
                {"role": "system", "content": "You are an expert in 3D structure analysis and reconstruction."},
                {"role": "user", "content": prompt}
            ]

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # Generate response
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True
            )

            # Extract generated text
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            # Calculate token usage (approximate)
            input_tokens = len(model_inputs.input_ids[0])
            output_tokens = len(output_ids)
            total_tokens = input_tokens + output_tokens

            return {
                'response': response,
                'usage': {
                    'prompt_tokens': input_tokens,
                    'completion_tokens': output_tokens,
                    'total_tokens': total_tokens
                },
                'cost': 0.0  # Local model has no cost
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'response': f"Error querying local Qwen model: {str(e)}",
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'cost': 0.0
            }

    def get_name(self) -> str:
        """Get the model name."""
        return f"QwenLocal({self.model_path})"