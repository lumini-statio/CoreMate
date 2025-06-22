# train.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import polars as pl
import pickle

# 1. Configuración
max_vocab_size = 50000
sequence_len = 90
embed_dim = 256
num_heads = 4
ff_dim = 512

# 2. Cargar y preparar datos
df = pl.read_csv('general.csv').select(['question', 'answer'])
df2 = pl.read_csv('reddit.csv').select(['question', 'answer'])
df3 = pl.read_csv('suicidio.csv').select(['question', 'answer'])

df_total = pl.concat([df, df2, df3], how="vertical").drop_nulls().unique()
questions = df_total["question"].cast(str).to_list()
answers = [f"[START] {a.strip()} [END]" for a in df_total["answer"].cast(str).to_list()]

# 3. Vectorizadores
question_vectorizer = layers.TextVectorization(max_tokens=max_vocab_size, output_mode="int", output_sequence_length=sequence_len)
answer_vectorizer = layers.TextVectorization(max_tokens=max_vocab_size, output_mode="int", output_sequence_length=sequence_len)
question_vectorizer.adapt(questions)
answer_vectorizer.adapt(answers)

# 4. Dataset
questions_tensor = question_vectorizer(questions)
answers_tensor = answer_vectorizer(answers)
encoder_input = questions_tensor
decoder_input = answers_tensor[:, :-1]
decoder_target = answers_tensor[:, 1:]

batch_size = 64
dataset = tf.data.Dataset.from_tensor_slices(((encoder_input, decoder_input), decoder_target))
dataset = dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 5. Modelo
def transformer_encoder(embed_dim, num_heads, ff_dim, dropout=0.1):
    inputs = layers.Input(shape=(None,))
    x = layers.Embedding(max_vocab_size, embed_dim)(inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    ffn = keras.Sequential([
        layers.Dense(ff_dim, activation="relu"),
        layers.Dense(embed_dim),
    ])
    x = layers.Add()([x, ffn(x)])
    return keras.Model(inputs, x)


def transformer_decoder(embed_dim, num_heads, ff_dim, dropout=0.1):
    enc_inputs = layers.Input(shape=(None, embed_dim))
    dec_inputs = layers.Input(shape=(None,))
    x = layers.Embedding(max_vocab_size, embed_dim)(dec_inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    attn1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = layers.Add()([x, attn1])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    attn2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, enc_inputs)
    x = layers.Add()([x, attn2])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    ffn = keras.Sequential([
        layers.Dense(ff_dim, activation="relu"),
        layers.Dense(embed_dim),
    ])
    x = layers.Add()([x, ffn(x)])
    outputs = layers.Dense(max_vocab_size, activation="softmax")(x)
    return keras.Model([enc_inputs, dec_inputs], outputs)


encoder = transformer_encoder(embed_dim, num_heads, ff_dim)
decoder = transformer_decoder(embed_dim, num_heads, ff_dim)
enc_inputs = layers.Input(shape=(None,))
dec_inputs = layers.Input(shape=(None,))
enc_outputs = encoder(enc_inputs)
dec_outputs = decoder([enc_outputs, dec_inputs])
model = keras.Model([enc_inputs, dec_inputs], dec_outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(dataset, epochs=10)

# 6. Guardar
model.save("CoreMate_v1")
with open("question_vectorizer.pkl", "wb") as f:
    pickle.dump(question_vectorizer, f)
with open("answer_vectorizer.pkl", "wb") as f:
    pickle.dump(answer_vectorizer, f)
