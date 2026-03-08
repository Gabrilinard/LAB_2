# Implementação Simplificada de um Encoder Transformer em Python

Este projeto apresenta uma **implementação educacional e simplificada do Encoder do Transformer**, utilizando apenas **NumPy e Pandas**.

O objetivo é demonstrar de forma clara como funcionam os principais componentes do Transformer:

- Tokenização simples
- Tabela de embeddings
- Positional Encoding (Seno e Cosseno)
- Self-Attention
- Feed Forward Network
- Layer Normalization
- Empilhamento de múltiplas camadas de encoder

Essa implementação **não utiliza frameworks de deep learning** como TensorFlow ou PyTorch para facilitar o entendimento matemático do funcionamento interno.

---

# Estrutura do Projeto

```
projeto-transformer/
│
├── atividade02.py
└── README.md
```

---

# Requisitos

Para executar o projeto, você precisa ter instalado:

- Python 3.8 ou superior
- NumPy
- Pandas

---

# Instalação

## 1. Instalar Python

Baixe o Python em:

https://www.python.org/downloads/

Durante a instalação marque a opção:

```
Add Python to PATH
```

---

## 2. Instalar as bibliotecas necessárias

Abra o terminal ou prompt de comando e execute:

```bash
pip install numpy pandas
```

---

# Como Executar o Código

1. Certifique-se de que o arquivo se chama `atividade02.py`

2. Execute no terminal:

```bash
python atividade02.py
```

A saída mostrará:

- IDs dos tokens
- tabela de vocabulário
- dimensões dos embeddings
- evolução das camadas do encoder
- vetor final do token **"gabriel"**

---

# Explicação Geral do Código

O código simula o processamento da frase:

```
gabriel fez a atividade
```

Fluxo geral do processamento:

```
Frase - Tokenização - Embeddings - Positional Encoding (Soma de vetores posicionais) - Encoder Transformer (Self-Attention, Residual + LayerNorm, Feed Forward Network, Residual + LayerNorm) - Representação final dos tokens (Vetor Z)

```

---

# Arquivo Principal

O código completo está no arquivo:

`atividade02.py`

---

# Parte 1 — Importação das Bibliotecas

```python
import numpy as np
import pandas as pd
```

- **NumPy** → utilizado para cálculos matemáticos e manipulação de matrizes  
- **Pandas** → utilizado para exibir o vocabulário em formato de tabela  

---

# Parte 2 — Construção do Vocabulário

```python
vocabulario_map = {"gabriel": 0, "fez": 1, "a": 2, "atividade": 3}
```

Criamos um **vocabulário simples**, onde cada palavra recebe um **ID numérico**.

| Palavra | ID |
|------|------|
| gabriel | 0 |
| fez | 1 |
| a | 2 |
| atividade | 3 |

Modelos de IA trabalham com **números**, não diretamente com texto.

---

# Parte 3 — Parâmetros do Modelo

```python
tamanho_vocabulario = len(mapeamento_palavras)
d_model = 64
d_ff = 256
N = 6
```

- `tamanho_vocabulario` - número de palavras no vocabulário  
- `d_model` - dimensão do vetor de embedding 
- `d_model` - dimensão da Feed Forward Network (expandindo 64 - 256 - 64)
- `d_model` - número de camadas do encoder

---

# Parte 4 — Tokenização da Frase

```python
frase_exemplo = "gabriel fez a atividade"
tokens_ids = [mapeamento_palavras[palavra] for palavra in frase_exemplo.split()]
```

Primeiro dividimos a frase:

```
["gabriel", "fez", "a", "atividade"]
```

Depois convertemos para IDs:

```
[0, 1, 2, 3]
```

---

# Parte 5 — Tabela de Embeddings

```python
np.random.seed(10)
matriz_identidade_visual = np.random.randn(tamanho_vocabulario, d_model)
```

Criamos uma **matriz de embeddings aleatória**.

Formato:

```
(tamanho_vocabulario , d_model)

(4 , 64)
```

Cada linha representa uma palavra.

---

# Parte 6 — Positional Encoding 
Como o Transformer processa todas as palavras ao mesmo tempo, ele não sabe a ordem natural da frase. Para resolver isso, implementamos o Positional Encoding usando funções trigonométricas:

```
def gerar_positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            div_term = np.exp(i * -(np.log(10000.0) / d_model))
            pe[pos, i] = np.sin(pos * div_term)
            pe[pos, i + 1] = np.cos(pos * div_term)
    return pe
```
Essas funções permitem que o modelo aprenda relações relativas entre as posições. O resultado é somado diretamente aos embeddings:
X = entrada_embeddings + pos_encoding

---

# Parte 7 — Seleção dos Embeddings da Frase

```python
entrada_embeddings = matriz_embeddings[tokens_ids] 
```

Formato da matriz:

```
(4 , 64)
```

4 palavras  
64 dimensões por palavra.

---

# Parte 8— Dimensão de Batch

```python
X = np.expand_dims(entrada_embeddings, axis=0) 
```

Formato final:

```
(batch , tokens , dimensão)

(1 , 4 , 64)
```

---

# Parte 9 — Softmax

```python
def aplicar_softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)
```

A função **Softmax** transforma valores em **probabilidades**.

Fórmula:

```
softmax(x_i) = e^x_i / Σ e^x_j
```

---

# Parte 10 — Layer Normalization

```python
def camada_normalizacao(x, epsilon=1e-6):
    media = np.mean(x, axis=-1, keepdims=True)
    variancia = np.var(x, axis=-1, keepdims=True)
    return (x - media) / np.sqrt(variancia + epsilon)
```

Normaliza os valores para estabilizar o modelo.

---

# Parte 11 — Classe EncoderLayer

```python
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
```

Cada camada possui:

Pesos fixos para Self-Attention

FFN (64 → 256 → 64)

LayerNorm e conexões residuais

Mecanismo de Atenção cria três representações:

- **Query (Q)**
- **Key (K)**
- **Value (V)**

---

## Cálculo de Q, K e V

```python
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

---

## Cálculo das Similaridades

```python
K_t = np.transpose(K, axes=(0, 2, 1))
scores = (Q @ K_t) / np.sqrt(d_model)
```

Resultado:

```
(1 , 4 , 4)
```

Cada palavra calcula atenção com todas as outras.

---

## Aplicação do Softmax

```python
pesos = aplicar_softmax(scores)
```

Transforma os valores em probabilidades de atenção.

---

## Combinação com Values

```python
pesos @ V
```

Produz vetores contextualizados.

---

# Parte 12 — Feed Forward Network

```python
def rede_feed_forward(x, d_model, d_ff=256):
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.zeros(d_ff)
    
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.zeros(d_model)
    
    intermediario = np.maximum(0, x @ W1 + b1)
    return intermediario @ W2 + b2
```

Rede neural aplicada **independentemente a cada token**.

Arquitetura:

```
Linear
 ↓
ReLU
 ↓
Linear
```

Dimensões:

```
64 → 256 → 64
```

---

# Parte 13 — Encoder Transformer

```python
N = 6
```

O encoder possui **6 camadas**, como no artigo original do Transformer.

Cada camada executa:

```
Self Attention
↓
Residual + LayerNorm
↓
Feed Forward
↓
Residual + LayerNorm
```

---

# Parte 14 — Loop das Camadas

```python
for i, camada in enumerate(camadas_encoder):
    X = camada.forward(X)
    print(f"Camada {i+1}: Shape mantido em {X.shape}")
```

Cada camada refina a representação da frase mantendo os pesos fixos definidos na inicialização da classe EncoderLayer.

---

# Parte 15 — Representação Final

```python
Vetor_Z = X
```

Após passar pelas 6 camadas, obtemos os **embeddings finais contextualizados**.

Formato:

```
(1 , 4 , 64)
```

---

# Parte 16 — Saída Final

```python
print(Z[0,0,:5])
```

Isso imprime **os primeiros 5 valores do vetor da palavra "gabriel"** após todo o processamento.

Este vetor contém informação contextual de toda a frase, pois cada camada do encoder aplica Self-Attention e Feed Forward Network.

---

# Limitações desta Implementação

Esta implementação é **didática**.

Não inclui:

- treinamento do modelo
- multi-head attention
- decoder
- backpropagation

---

# Possíveis Melhorias

O projeto pode evoluir adicionando:

- **Multi Head Attention**
- **Decoder Transformer**
- **Treinamento com PyTorch ou TensorFlow**

---

# Objetivo Educacional

Este código tem como objetivo ajudar a entender:

- como Transformers funcionam internamente
- como Self-Attention é calculado
- como vetores são transformados ao longo das camadas

## . Nota de Integridade e Créditos
Este trabalho seguiu rigorosamente as diretrizes de integridade do laboratório:
- **Uso de IA:** Ferramentas de IA Generativa (Gemini/ChatGPT) foram consultadas exclusivamente para suporte em sintaxe da biblioteca `numpy` (como manipulação de eixos no `np.mean` e lógica de transposição de tensores 3D) e para auxílio na estruturação deste documento explicativo. Elas auxiliaram na resolução de dúvidas de sintaxe da biblioteca numpy e no brainstorming para a estruturação lógica em pequenas partes do código, dúvidas mesmo. Mas toda a implementação foi eu tentando fazer a mão e entender cada parte do código e fazer de acordo com o que foi pedido no laboratório. A documentação foI revisada e validada por mim, garantindo que o trabalho reflita meu entendimento sobre o funcionamento do Encoder Transformer, conforme as regras do contrato pedagógico.
