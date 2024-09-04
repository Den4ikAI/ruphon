import os
import re
import json
import string
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from razdel import sentenize
from razdel.substring import Substring
from huggingface_hub import hf_hub_download, snapshot_download
from .char_tokenizer import CharacterTokenizer


class TextPreprocessor:
    def split_by_words(string):
        string = string.replace(" - ",' ~ ')
        match = list(re.finditer(r"\w*(?:\+\w+)*|[^\w\s]+", string.lower()))
        remaining_text =  [string[l.end():r.start()] for l,r in zip(match, match[1:])]

        words = [string[x.start():x.end()] for x in match]
        words_mask = [i for i, w  in enumerate(words) if w]
        
        valid_words = [words[i] for i in words_mask]
        if len(words_mask) == 0:
            return valid_words, ["", ""]
        remaining_text_res = ["".join(remaining_text[:words_mask[0]])] + ["".join(remaining_text[l+1:r]) for l, r in zip(words_mask, words_mask[1:])]
        remaining_text_res.append("".join(remaining_text[words_mask[-1]+1:]))
        return valid_words, remaining_text_res

    def split_by_sentences(string):
        sentences = list(sentenize(string))
        if len(sentences) == 0:
            return []
        result = [string[l.stop:r.start] + r.text if l.stop != r.start else r.text for l,r in zip([Substring(0,0, "")] + sentences, sentences)]
        result[-1] = result[-1] + string[sentences[-1].stop:]
        return result
        
    def delete_spaces_before_punc(text):
        punc = "!\"#$%&'()*,./:;<=>?@[\\]^_`{|}-"
        for char in punc:
            if char == '-':
                text = text.replace(" " + char, char).replace(char + " ", char)
            text = text.replace(" " + char, char)
        return text.replace('~', '-')



class RUPhon:
    @classmethod
    def load(cls, model_type: str, workdir: str = None, device: str = "CPU"):
        if model_type not in ["small", "big"]:
            raise ValueError("model_type must be 'small' or 'big'")
    
        if workdir is None:
            workdir = os.path.dirname(os.path.abspath(__file__))
    
        repo_id = f"ruphon/phonemizer-{model_type}"
        local_dir = os.path.join(workdir, repo_id.split('/')[-1])
    
        if not os.path.exists(local_dir) or not os.listdir(local_dir):
            local_dir = snapshot_download(repo_id=repo_id, local_dir_use_symlinks=False, local_dir=local_dir)
            

    
        instance = cls()
        instance._initialize(local_dir, device)
        return instance
    
        
    def __init__(self):
        self.tokenizer = None
        self.ort_session = None
        self.id2label = None
        self.input_names = None
        self.output_names = None

    def _initialize(self, model_dir: str, device: str):

        self.tokenizer = CharacterTokenizer.from_pretrained(model_dir)
        
        model_path = os.path.join(model_dir, "model.onnx")
        providers = ['CPUExecutionProvider'] if device == "CPU" else ['CUDAExecutionProvider']
        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            self.id2label = {int(k): v for k, v in config['id2label'].items()}
            self.id2label = {k: "" if v == "<DELETE>" else v for k, v in self.id2label.items()}

        self.input_names = [input.name for input in self.ort_session.get_inputs()]
        self.output_names = [output.name for output in self.ort_session.get_outputs()]

    def _predict_single(self, word: str) -> List[str]:
        inputs = self.tokenizer([word.lower()], padding=True, return_tensors="np")
        ort_inputs = {name: inputs[name] for name in self.input_names}
        ort_outputs = self.ort_session.run(self.output_names, ort_inputs)
        
        logits = ort_outputs[0]
        predictions = np.argmax(logits, axis=-1)
        
        word_labels = []
        for p in predictions[0]:
            if p != -100:  
                word_labels.append(self.id2label[p])
        
        return word_labels

    def phonemize(self, text: str, put_stress: bool=True, stress_symbol: str="'") -> str:
        sentences = TextPreprocessor.split_by_sentences(text)
        phonemized_sentences = []
        output_sentence = ""
        
        for sentence in sentences:
            words, spaces = TextPreprocessor.split_by_words(sentence)
            phonemized_words = []

            for word in words:
                if word not in string.punctuation:    
                    word_phonemes = self._predict_single(word)
                    phonemized_words.append(word_phonemes)
                else:
                    phonemized_words.append(word + " ")
            
            phonemized_sentence = ""
            for word_phonemes, space in zip(phonemized_words, spaces):
                phonemized_sentence += "".join(word_phonemes) + " "
            phonemized_sentences.append(phonemized_sentence)
            
        output_sentence = TextPreprocessor.delete_spaces_before_punc(" ".join(phonemized_sentences))
        if not put_stress:
            output_sentence = output_sentence.replace("'","")
        else:
            output_sentence = output_sentence.replace("'", stress_symbol)
            
        return output_sentence
