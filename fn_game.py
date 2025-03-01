import openai
from openai import AsyncOpenAI
from typing import Optional, List, Any, Tuple, Type, Union, Dict
import os


class ModelInterface:
    """
    A unified, asynchronous interface for calling language models with the
    structured parse method. If the user does not provide an output_type,
    we can parse into a trivial wrapper with a single text field.
    """

    def __init__(
        self, model_name: str, api_key: Optional[str] = None, max_tokens: int = None
    ):
        """
        Args:
            model_name: e.g. "gpt-4o" or huggingface like "DeepSeek/.."
            api_key: optional override of environment API key
            max_tokens: max tokens for completions (OpenAI) or max_new_tokens (HuggingFace)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_KEY"),
            organization=os.environ.get("OPENAI_ORG"),
        )
        self.uses_openai = True


    def _format_messages(
        self, system_prompt: str, user_messages: List[dict]
    ) -> List[dict]:
        """
        Utility to prepend a system message.
        user_messages is typically: [{'role': 'user', 'content': ...}, ...]
        """
        return [{"role": "system", "content": system_prompt}] + user_messages

    async def call(
        self,
        messages: List[dict],
        system_prompt: str = "You are a helpful assistant.",
        output_type: Optional[Type[BaseModel]] = None,
        max_retries: int = 3,
        tools: Optional[Dict[str, Any]] = None,
    ) -> Union[str, BaseModel]:
        """
        Calls the LLM using OpenAI's chat completion or HF, returning either raw text
        or a Pydantic-validated object.
        """
        if not self.uses_openai:
            # For HF models, we simply flatten messages into a single string prompt:
            formatted_prompt = f"{system_prompt}\n"
            for m in messages:
                formatted_prompt += f"{m['role'].upper()}:\n{m['content']}\n\n"
            logging.info("\n=== Model Call (HuggingFace) ===")
            logging.info(formatted_prompt)
            logging.info("=" * 80)
            return self.call_hf(formatted_prompt, max_length=self.max_tokens)

        # Otherwise, if using an OpenAI-based model:
        formatted_messages = self._format_messages(system_prompt, messages)
        # logging.info("\n=== Model Call (OpenAI) ===")
        # logging.info("System:", system_prompt)
        # for msg in messages:
        #     logging.info(f"{msg['role'].upper()}:", msg['content'])
        # logging.info("=" * 80)

        last_exception = None
        for attempt in range(max_retries):
            try:
                if output_type is None:
                    response = await self.client.chat.completions.create(
                        messages=formatted_messages,
                        model=self.model_name,
                        max_completion_tokens=self.max_tokens,
                        tools=tools
                    )
                    print(response)
                    if tools:
                        result = response.choices[0].message
                    else:
                        result = response.choices[0].message.content
                    # logging.info("\nRESPONSE:", result)
                    # logging.info("=" * 80)
                    return result
                else:
                    response = await self.client.beta.chat.completions.parse(
                        messages=formatted_messages,
                        model=self.model_name,
                        max_completion_tokens=self.max_tokens,
                        response_format=output_type,
                    )
                    parsed_obj = response.choices[0].message.parsed
                    result = output_type.model_validate(parsed_obj)
                    # logging.info("\nRESPONSE:", result)
                    # logging.info("=" * 80)
                    return result

            except Exception as e:
                logging.info(
                    f"[DEBUG] Attempt {attempt+1}/{max_retries} failed with exception: {e}"
                )
                logging.info("[DEBUG] System prompt (verbatim):")
                logging.info(system_prompt)
                logging.info("[DEBUG] User messages (verbatim):")
                for idx, msg in enumerate(messages):
                    logging.info(f"  Message {idx+1} - role='{msg['role']}':")
                    logging.info(msg["content"])
                last_exception = e
                await asyncio.sleep(1.0 * (attempt + 1))

        raise last_exception

    def call_hf(
        self, formatted_prompt: str, max_length: int = 1000, temp: float = 0.7
    ) -> str:
        """
        Calls a huggingface model with text-generation pipeline.
        """
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        generation_args = {
            "max_new_tokens": max_length,
            "return_full_text": False,
            "temperature": temp,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        output = pipe(formatted_prompt, **generation_args)
        return output[0]["generated_text"]




class Game:

    def __init__(self, env, model):
        self.env = env
        self.model = model

