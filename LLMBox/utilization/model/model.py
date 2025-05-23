import time
from functools import cached_property, wraps
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Type, Union

import tiktoken
from tiktoken import Encoding
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from utilization.utils.logging import passed_in_commandline

from ..model_enum import API_MODELS, ENDPOINT_ARGS, ERROR_OVERVIEW
from ..utils import ModelArguments, resolve_generation_args
from ..utils.arguments import ModelBackendMixin
from .model_utils.conversation import Conversation
from .model_utils.prefix_caching import Cacher

if TYPE_CHECKING:
    # solve the circular import
    from ..utils import ModelArguments
    from .vllm_model import LLM

logger = getLogger(__name__)


class Model(ModelBackendMixin):
    r"""The base model object for all models.

    Args:
        args (ModelArguments): The global configurations.

    Attributes:
        name (str): The name of this model.
        model_type (str): The type of this model, which can be chosen from `base` and `instruction`.
        tokenizer (Union[transformers.PreTrainedTokenizer, PreTrainedTokenizerFast, tiktoken.Encoding]): The tokenizer of this model.
        max_tokens (int): The maximum token length of this model.
        generation_kwargs (dict): The configurations for open-ended generation.
        ppl_kwargs (dict, *optional*): The configurations for computing PPL score.
    """
    name: str
    model_type: Literal["base", "chat"]
    model_backend: Literal["anthropic", "dashscope", "huggingface", "openai", "qianfan", "vllm"]

    model: Union[PreTrainedModel, "LLM", None] = None
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, Encoding, None] = None
    cacher: Optional["Cacher"] = None
    model_max_input_and_output: int
    support_cache: bool = True
    multi_turn = False

    def __init__(self, args: "ModelArguments"):
        self.args = args
        self.name = args.model_name_or_path
        self.model_type = args.model_type

    @cached_property
    def chat_template(self) -> Dict[str, Any]:
        """Get the chat template for this model. Return a dictionary with the following keys:

        - chat_template (str): The chat template name.
        - special_tokens_map (optional, Dict[str, str]): The special tokens map."""
        use_tokenizer = self.args.chat_template == "tokenizer"
        if self.args.chat_template is not None and not use_tokenizer:
            return {"chat_template": self.args.chat_template}

        if self.is_local_model() and self.model_type == "chat":
            special_tokens_map = getattr(self.tokenizer, "special_tokens_map", None)
            tokenizer_template = getattr(self.tokenizer, "chat_template", None)
            logger.debug(
                f"Tokenizer template: {tokenizer_template}, special tokens map: {special_tokens_map}, use_tokenizer: {use_tokenizer}"
            )
            if tokenizer_template is not None and use_tokenizer:
                logger.info(f"Automatically load chat_template to {tokenizer_template}.")
                template = {"chat_template": tokenizer_template}
                if special_tokens_map is not None:
                    template["special_tokens_map"] = special_tokens_map
                return template

            from ..chat_templates import DEFAULT_CHAT_CONFIGS

            model_name = self.name.lower().replace("-", "").replace("_", "")
            logger.debug(f"model_name: {model_name}, DEFAULT_CHAT_CONFIGS: {DEFAULT_CHAT_CONFIGS.keys()}")
            for config_name in DEFAULT_CHAT_CONFIGS:
                if config_name in model_name:
                    logger.info(f"Automatically set chat_template to {config_name}.")
                    template = {"chat_template": config_name}
                    if special_tokens_map is not None:
                        template["special_tokens_map"] = special_tokens_map
                    return template

        return {"chat_template": None}

    def set_cacher(self, cacher: Any = None):
        r"""Set the cacher for this model. The cacher is used to cache the generated results for the model."""
        if isinstance(cacher, Cacher):
            self.cacher = cacher
        elif cacher is None:
            self.cacher = None

    def _remove_tokenizer(self):
        return

    def _reload_tokenizer(self):
        return

    @property
    def use_cache(self) -> bool:
        """Whether to use the cache for this model. This should be used during iterating the dataset."""
        return self.support_cache and self.cacher is not None

    @use_cache.setter
    def use_cache(self, value: bool) -> bool:
        if value:
            raise ValueError("Please use `set_cacher` to set the cacher.")
        else:
            self.cacher = None
        return False

    def set_ppl_args(self, **extra_model_args):
        r"""Set the configurations for PPL score calculation. This is useful because different datasets may have different requirements for ppl calculation."""
        raise NotImplementedError(f"{self.name} model must implement the `set_ppl_args` function.")

    def get_ppl(self, batched_inputs: List[Tuple[str, str]]) -> List[Tuple[float, int]]:
        r"""Compute the PPL score of the option given the context for this batch.

        Args:
            batched_inputs (List[Tuple[str, str]]): The batch of context and option pairs.

        Returns:
            List[float]: The list of PPL scores.
        """
        raise NotImplementedError(f"{self.name} model does not support `get_ppl`.")

    def set_generation_args(self, **extra_model_args):
        r"""Set the configurations for open-ended generation. This is useful because different datasets may have different requirements for generation."""

        raise NotImplementedError(f"{self.name} model must implement the `set_generation_args` function.")

    def generation(self, batched_inputs: List[str]) -> List[str]:
        r"""Generate the response of given question for this batch.

        Args:
            batched_inputs (List[str]): The batch of questions.

        Returns:
            List[str]: The list of generation results.
        """
        raise NotImplementedError(f"{self.name} model does not support `generation`.")

    def set_prob_args(self, **extra_model_args):
        r"""Set the configurations for generation. This is useful because different datasets may have different requirements for get_prob."""

        raise NotImplementedError(f"{self.name} model must implement the `set_prob_args` function.")

    def get_prob(self, batched_inputs: List[Tuple[str, int]]) -> List[int]:
        r"""Calculates the probability of each option in a multiple-choice question being the correct continuation of a given text prompt.

        Args:
            batched_inputs (List[Tuple[str, int]]): The batch of questions and corresponding option nums.

        Returns:
            List[int]: The option index of greatest probabiltiy.
        """
        raise NotImplementedError(f"{self.name} model does not support `get_prob`.")

    def _aggregate_model_attr(self) -> Dict[str, Any]:
        kwargs = {}
        for key in getattr(self, "_repr", []):
            if getattr(self, key, None):
                kwargs[key] = getattr(self, key)
        return kwargs


class SkipError(Exception):
    """Skip the current instance, most likely due to inappropriate content."""

    pass


class RetryError(Exception):
    """Retry the current instance, most likely due to rate limit or internet issuses."""

    pass


class RaiseError(Exception):
    """Abort the process, most likely due to parameter errors or api key issues."""

    pass


class EnsureTypeError(Exception):
    """Ensure the type of the return value."""

    pass


class SkipResponse(str):
    pass


def ensure_type(type_):
    """Ensure the type of the return value. If the return value is not the expected type, raise an EnsureTypeError."""

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) == 1 and len(kwargs) == 0 and func.__name__ == "_get_assistant" and isinstance(
                args[0], SkipResponse
            ):
                return str(args[0])

            results = func(*args, **kwargs)
            if not isinstance(results, type_):
                raise EnsureTypeError(f"Expected {type_}, but got {type(results)}. {func.__name__}: {args}, {kwargs}")
            return results

        return wrapper

    return decorator


class ApiModel(Model):

    model_backend: Literal["anthropic", "dashscope", "openai", "qianfan"]

    model = None
    tokenizer: Encoding
    cacher = None
    model_max_input_and_output: int
    support_cache: bool = False

    max_retry_times: int = 10

    endpoint: Literal["completions", "chat/completions"]

    _retry_errors: Tuple[Type[Exception], ...]
    """Retry the current request for max_retry_times. Most likely due to rate limit or internet issues."""

    _raise_errors: Tuple[Type[Exception], ...]
    """Retry the current request for once and then skip. If the error persists, raise the error. Due with most complicated issues."""

    _skip_errors: Tuple[Type[Exception], ...]
    """Skip the current request whenever catches these errors. Most likely due to inappropriate content."""

    def __init__(self, args: ModelArguments):
        super().__init__(args)
        if getattr(self, "tokenizer", None) is None:
            try:
                self.tokenizer = tiktoken.get_encoding(args.tokenizer_name_or_path)
                logger.debug(f"Tokenizer: {self.tokenizer}")
            except Exception as e:
                logger.warning(
                    f"Failed to load tokenizer from `{args.tokenizer_name_or_path}`. Please specify a valid tokenizer through `--tokenizer`."
                )
                self.tokenizer = None
        if args.model_name_or_path in API_MODELS:
            self.endpoint = API_MODELS[args.model_name_or_path]["endpoint"]
        else:
            self.endpoint = "chat/completions"
        self._retries_before_raise = 0
        if EnsureTypeError not in self._raise_errors:
            self._raise_errors = self._raise_errors + (EnsureTypeError,)

    def _completions(self, *, prompt, model, **kwargs):
        """completions endpoint adapter."""
        raise NotImplementedError(f"{self.name} model must implement the `_completions` function.")

    def _chat_completions(self, *, messages, model, **kwargs):
        """chat/completions endpoint adapter."""
        raise NotImplementedError(f"{self.name} model must implement the `_chat_completions` function.")

    @staticmethod
    def _get_assistant(msg: Any) -> str:
        """Get the assistant content from the response."""
        raise NotImplementedError(f"Model must implement the `_get_assistant` function.")

    @staticmethod
    def _get_error_type(response: Any) -> Optional[Type[Exception]]:
        """If the api library does not raise an error, we can check the response to determine the error type."""
        return None

    def set_generation_args(self, **extra_model_args):
        r"""Set the configurations for open-ended generation. This is useful because different datasets may have different requirements for generation."""
        if self.name in API_MODELS and "args" in API_MODELS[self.name]:
            endpoint_schema = API_MODELS[self.name]["args"]
        else:
            endpoint_name = self.model_backend + "/" + self.endpoint
            if endpoint_name not in ENDPOINT_ARGS:
                raise ValueError(f"Endpoint {endpoint_name} is not supported.")
            endpoint_schema = ENDPOINT_ARGS[endpoint_name]

        self.multi_turn = extra_model_args.pop("multi_turn", False)
        self.generation_kwargs = resolve_generation_args(self.args, extra_model_args, endpoint_schema)

        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")
        return self.generation_kwargs

    def generation(
        self,
        batched_inputs: Union[List[str], List[Conversation]],
    ) -> Union[List[str], List[Tuple[str, ...]]]:
        multi_turn_results = self.request(
            prompt=batched_inputs,
            multi_turn=self.multi_turn,
            **self.generation_kwargs,
        )
        answers = [self._get_assistant(turn) for turn in multi_turn_results]

        # group the multi-turn results into a tuple
        if self.multi_turn:
            answers = [tuple(answers)]
        return answers

    def request(
        self,
        prompt: Union[List[str], List[Conversation]],
        *,
        multi_turn: bool = False,
        **model_kwargs,
    ) -> List[Any]:
        r"""Call the OpenAI API.

        Args:
            prompt (List[str]): The list of input prompts.
            openai_kwargs (dict): The additional calling configurations.
            multi_turn (bool): Default is False. Set to True if multi-turns needed.

        Returns:
            List[Any]: A batch of results (completions endpoint) or a list of conversation (chat/completions endpoint).
        """
        retry_times = 0
        while retry_times < self.max_retry_times:
            try:
                if self.endpoint == "completions":
                    assert not multi_turn
                    results = self._completions(prompt=prompt, model=self.name, **model_kwargs)
                    self._retries_before_raise = 0
                    return results
                elif self.endpoint == "chat/completions":
                    assert (len(prompt) == 1), "Chat completions does not support batch input."
                    conversation = prompt[0]

                    # legacy support for string prompt
                    if isinstance(conversation, str):
                        messages = [{"role": "user", "content": c} for c in conversation.split("__SEPARATOR__")]
                        conversation = Conversation(messages)

                    results = []
                    for _ in range(conversation.num_turns):
                        response = self._chat_completions(
                            messages=conversation.messages,
                            model=self.name,
                            **model_kwargs,
                        )

                        # dashscope compatibility
                        error_type = self._get_error_type(response)
                        if error_type is not None:
                            raise error_type(response.message)

                        results.append(response)
                        content = self._get_assistant(response)
                        conversation.add_multi_turn(assistant=content)
                    self._retries_before_raise = 0
                    return results

            except self._retry_errors as e:
                logger.warning(f"Receive {e.__class__.__name__}: {str(e)}, retrying...")
                retry_times += 1
                if retry_times < self.max_retry_times:
                    time.sleep(10)

            except self._raise_errors as e:
                error_msg = e.__class__.__name__ + ": " + str(e)
                if e.__class__.__name__ in ERROR_OVERVIEW:
                    error_msg += "\n" + ERROR_OVERVIEW.get(e.__class__.__name__)
                self._retries_before_raise += 1
                if self._retries_before_raise > self.max_retry_times:
                    raise ConnectionError(error_msg)
                # elif self._retries_before_raise == 1:
                #     logger.warning(f"Receive {e.__class__.__name__}: {str(e)}, retrying...")
                #     time.sleep(1)
                # else:
                #     logger.warning(f"Receive {e.__class__.__name__}: {str(e)}, skipping...")
                #     return [SkipResponse()]
                elif self._retries_before_raise == 8:
                    logger.warning(f"Receive {e.__class__.__name__}: {str(e)}, skipping...")
                    return [SkipResponse()]

                else:
                    logger.warning(f"Receive {e.__class__.__name__}: {str(e)},{self._retries_before_raise} times, retrying...")
                    time.sleep(1)

            except self._skip_errors as e:
                logger.warning(f"Receive {e.__class__.__name__}: {str(e)}, skipping...")
                return [SkipResponse()]

            except Exception as e:
                logger.warning(f"Receive {e.__class__.__name__}: {str(e)}, retrying...")
                retry_times += 1
                if retry_times < self.max_retry_times:
                    time.sleep(retry_times)

        raise ConnectionError(f"API endpoint error after {self.max_retry_times} times of retries.")
