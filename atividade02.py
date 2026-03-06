import pandas as pd
import numpy as np

mapeamento_palavras = {"gabriel": 0, "fez": 1, "a": 2, "atividade": 3}
vocabulario_df = pd.DataFrame(list(mapeamento_palavras.items()), columns=['Palavra', 'ID'])

tamanho_vocabulario = len(mapeamento_palavras)
d_model = 64
d_ff = 256
N = 6 

frase_exemplo = "gabriel fez a atividade"
tokens_ids = [mapeamento_palavras[palavra] for palavra in frase_exemplo.split()]

np.random.seed(10)
matriz_embeddings = np.random.randn(tamanho_vocabulario, d_model)
entrada_embeddings = matriz_embeddings[tokens_ids] 

X = np.expand_dims(entrada_embeddings, axis=0) 

print("\n--- Tabela de Vocabulário ---")
print(vocabulario_df.to_string(index=False))
print("\n--- Dimensões do Processamento ---")
print(f"Tabela de Embeddings (Vocabulário x Dimensão): {matriz_embeddings.shape}")
print(f"Tensor Final X (Batch, Sequência, Dimensão): {X.shape}")

def aplicar_softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def camada_normalizacao(x, epsilon=1e-6):
    media = np.mean(x, axis=-1, keepdims=True)
    variancia = np.var(x, axis=-1, keepdims=True)
    return (x - media) / np.sqrt(variancia + epsilon)

class EncoderLayer:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        self.Wq = np.random.randn(d_model, d_model)
        self.Wk = np.random.randn(d_model, d_model)
        self.Wv = np.random.randn(d_model, d_model)
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros(d_model)

    def mecanismo_atencao(self, X):
        Q = X @ self.Wq  
        K = X @ self.Wk  
        V = X @ self.Wv  
        K_t = np.transpose(K, axes=(0, 2, 1))
        scores = (Q @ K_t) / np.sqrt(self.d_model)
        pesos = aplicar_softmax(scores)
        return pesos @ V

    def rede_feed_forward(self, x):
        intermediario = np.maximum(0, x @ self.W1 + self.b1)
        return intermediario @ self.W2 + self.b2

    def forward(self, X):
        X_att = self.mecanismo_atencao(X)
        X_norm1 = camada_normalizacao(X + X_att)
        X_ffn = self.rede_feed_forward(X_norm1)
        X_out = camada_normalizacao(X_norm1 + X_ffn)
        return X_out

np.random.seed(10)
camadas_encoder = [EncoderLayer(d_model, d_ff) for _ in range(N)]

print(f"--- Iniciando o Encoder Stack (N={N}) ---")
for i, camada in enumerate(camadas_encoder):
    X = camada.forward(X)
    print(f"Camada {i+1}: Shape mantido em {X.shape}")

Vetor_Z = X
print(f"\nValidação do Shape final do Vetor Z: {Vetor_Z.shape}")
print(f"Amostra dos primeiros 5 valores (features) da primeira palavra:\n{Vetor_Z[0, 0, :5]}")