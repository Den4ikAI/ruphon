import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import (
    cached_file,
    copy_func,
    download_url,
    extract_commit_hash,
    is_remote_url,
)


class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, characters: Sequence[str], model_max_length: int, vocab_file: str = "vocab.json", **kwargs):
        """Character tokenizer for Hugging Face transformers.

        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                <unk> with id=3. Following are list of all of the special tokens with
                their corresponding ids:
                    "<pad>": 0
                    "<s>": 1
                    "</s>": 2
                    "<unk>": 3
                an id (starting at 7) will be assigned to each character.

            model_max_length (int): Model maximum sequence length.
            vocab_file (str): Path to the vocabulary file.
        """
        self.characters = characters
        self.model_max_length = model_max_length
        self.vocab_file = vocab_file

        self._vocab_str_to_int = {
            **{ch: i for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        pad_token = AddedToken("<pad>", lstrip=False, rstrip=False)
        bos_token = AddedToken("<s>", lstrip=False, rstrip=False)
        eos_token = AddedToken("</s>", lstrip=False, rstrip=False)
        unk_token = AddedToken("<unk>", lstrip=False, rstrip=False)
        sep_token = AddedToken("<sep>", lstrip=False, rstrip=False)
        cls_token = AddedToken("<cls>", lstrip=False, rstrip=False)
        mask_token = AddedToken("<mask>", lstrip=True, rstrip=False)
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        i = 0
        while i < len(text):
            # Проверяем, есть ли составной токен, начинающийся с текущей позиции
            for j in range(len(text), i, -1):
                if text[i:j] in self._vocab_str_to_int:
                    tokens.append(text[i:j])
                    i = j
                    break
            else:
                # Если составной токен не найден, добавляем один символ
                tokens.append(text[i])
                i += 1
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["<unk>"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def get_config(self) -> Dict:
        return {
            "name": "CharacterTokenizer",
            "vocab_file": self.vocab_file,
            "model_max_length": self.model_max_length,
            "size": len(self.characters)
        }

    @classmethod
    def from_config(cls, config: Dict, save_directory: Union[str, os.PathLike]) -> "CharacterTokenizer":
        vocab_file_path = os.path.join(save_directory, config["vocab_file"])
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        characters = [k for k in vocab.keys() if k not in ["<pad>", "<s>", "</s>", "<unk>", "<sep>", "<cls>", "<mask>"]]
        return cls(characters=characters, model_max_length=config["model_max_length"], vocab_file=config["vocab_file"])

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)
        vocab_file_path = Path(save_directory) / self.vocab_file
        
        # Создаем новый словарь, преобразуя AddedToken в строки
        vocab_to_save = {}
        for key, value in self._vocab_str_to_int.items():
            if isinstance(key, AddedToken):
                key = str(key)
            vocab_to_save[key] = value
        
        with open(vocab_file_path, "w", encoding="utf-8") as f:
            json.dump(vocab_to_save, f, indent=4, ensure_ascii=False)
    
    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)
    
        is_local = os.path.isdir(save_directory)
        if os.path.exists(save_directory):
            resolved_config_file = os.path.join(save_directory, "tokenizer_config.json")
            is_local = True
        else:
            configuration_file = "tokenizer_config.json"
    
            try:
                resolved_config_file = cached_file(
                    save_directory,
                    configuration_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _commit_hash=commit_hash,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            except EnvironmentError:
                raise
            except Exception:
                raise EnvironmentError(
                    f"Can't load the configuration of '{save_directory}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{save_directory}' is the correct path to a directory"
                    f" containing a {save_directory} file"
                )
    
        with open(resolved_config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        
        # Загрузка словаря
        vocab_file_path = os.path.join(save_directory, cfg["vocab_file"])
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        
        # Создание экземпляра токенизатора
        characters = [k for k in vocab.keys()]
        instance = cls(characters=characters, model_max_length=cfg["model_max_length"], vocab_file=cfg["vocab_file"])
        
        return instance

    def add_tokens(self, new_tokens: Union[str, List[str]], special_tokens: bool = False):
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
        
        added_count = 0
        for token in new_tokens:
            if token not in self._vocab_str_to_int:
                index = len(self._vocab_str_to_int)
                self._vocab_str_to_int[token] = index
                self._vocab_int_to_str[index] = token
                added_count += 1
        return added_count

    
    def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, AddedToken]]):
        added = super().add_special_tokens(special_tokens_dict)
        for token_name, token in special_tokens_dict.items():
            if isinstance(token, AddedToken):
                token = token.content
            if token not in self._vocab_str_to_int:
                self._vocab_str_to_int[token] = len(self._vocab_str_to_int)
                self._vocab_int_to_str[self._vocab_str_to_int[token]] = token
        return added

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def encode_plus(
        self,
        text,
        add_special_tokens=True,
        padding=False,
        truncation=None,
        max_length=None,
        stride=0,
        is_split_into_words=False,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_token_type_ids=False,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        verbose=True,
        **kwargs,
    ):
        return super().encode_plus(text=text,
                                   add_special_tokens=add_special_tokens,
                                   padding=padding,
                                   truncation=truncation,
                                   max_length=max_length,
                                   stride=stride,
                                   is_split_into_words=False,
                                   pad_to_multiple_of=pad_to_multiple_of,
                                   return_tensors=return_tensors,
                                   return_attention_mask=return_attention_mask,
                                   return_overflowing_tokens=return_overflowing_tokens,
                                   return_special_tokens_mask=return_special_tokens_mask,
                                   return_offsets_mapping=return_offsets_mapping,
                                   return_length=return_length,
                                   verbose=verbose,
                                   **kwargs
                                   )
