from ruphon import RUPhon

# Инициализация фонемизатора
phonemizer = RUPhon()

phonemizer = phonemizer.load("small", workdir="./models", device="CPU")

# Фонемизация текста
input_text = "+я программ+ирую н+а python."
result = phonemizer.phonemize(input_text, put_stress=True, stress_symbol="'")

print(f"Input: {input_text}")
print(f"Phonemized: {result}")
