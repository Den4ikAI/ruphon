# RUPhon

RUPhon - это библиотека для фонемизации русского текста, использующая передовые модели RUAccent-encoder.

## Особенности

RUPhon - библиотека, позволяющая фонемизировать текст с учетом ударений. Модель поддерживает два языка - английский и русский. 

Фонемсет следующий:

```
k|l|s:|ʒ|t~ɕ|j|'ɐ|iː|fʲ:|ɹ|'ʉ|v|'ɵ|ə+r|d͡ʒ|ʊ|ɫ|ɪ|oː|jɵ|ɔː|ɚ|j:|dʲ:|ʔ|sʲ|ɛː|u|nʲ:|ʃ|d~zʲ|'je|t~s|ɑː|mʲ|ɵ|ɡʲ|jə|d:|jʊ|ɛ|k:|vʲ:|x|nʲ|n|jɪ|zʲ|ɐ|æː|n:|pʲ:|v:|r|l̩|t~ɕ:|'ju|xʲ|'jæ|'ɪ|ɕ|b|aː|o|kʲ:|'ə|ɕ:|dʲ|rʲ|d~ʐ|'ji|tʲ:|w|bʲ|p:|r:|ɝ|eː|ə|t|'ja|'ʊ|b:|mʲ:|'jʉ|'u|z|ju|t~sʲ|ɣ|z:|jæ|ð|je|ʐ|ʂ:|ɜː|ʐ:|ʑ:|lʲ|ɡ:|ɨ|ji|pʲ|ɒ|ɪː|zʲ:|ɔ|θ|ɫ:|fʲ|p|vʲ|i|t~ʂ|'i|'a|ŋ|ɜ|ʊ̯|ɪ̯|n̩|kʲ|'o|uː|f|jʉ|'ɛ|s+_|ɡ|'æ|ʉ|m|s|a|'e|æ|tʲ|h|ɪ+rʲ|m:|ɑ|e|'jɵ|d|sʲ:|ʌ|d~z|lʲ:|t~s+_|m̩|ʂ|ja|t~s:|ɝː|'ɨ|ʍ|t:|t͡ʃ|'jɪ|rʲ:
```

## Установка

```
pip install ruphon
```

или

```
pip install git+https://github.com/Den4ikAI/ruphon.git
```

## Основные функции

### Phonemizer.load(model_type, workdir, device)

Загружает модель фонемизатора.

- `model_type`: "small" или "big". small - 14 миллионов параметров, big - 28 миллионов
- `workdir`: директория для сохранения моделей (по умолчанию - директория библиотеки)
- `device`: "CPU" или "CUDA" (для использования GPU) (Требуется установить onnxruntime-gpu)

### phonemizer.phonemize(text, put_stress, stress_symbol)

Фонемизирует входной текст.

- `text`: входной текст для фонемизации
- `put_stress`: добавлять ли ударения (по умолчанию True)
- `stress_symbol`: символ для обозначения ударения (по умолчанию "'")



## Использование

```python
from ruphon import RUPhon

phonemizer = RUPhon()

phonemizer = phonemizer.load("small", workdir="./models", device="CPU")

input_text = "+я программ+ирую н+а python."
result = phonemizer.phonemize(input_text, put_stress=True, stress_symbol="'")

print(f"Input: {input_text}")
print(f"Phonemized: {result}")
```

## Использование с автоматической расстановкой ударений

```python
from ruphon import RUPhon
from ruaccent import RUAccent

phonemizer = RUPhon()
phonemizer = phonemizer.load("small", workdir="./models", device="CPU")

accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo3', use_dictionary=True, tiny_mode=False)

input_text = "я программирую на python."

accented_text = accentizer.process_all(input_text)

print(f"Input: {input_text}")
print(f"Accented: {accented_text}")

result = phonemizer.phonemize(accented_text, put_stress=True, stress_symbol="'")

print(f"Phonemized: {result}")
```

## Донат
Вы можете поддержать проект деньгами. Это поможет быстрее разрабатывать более качественные новые версии. 
CloudTips: https://pay.cloudtips.ru/p/b9d86686