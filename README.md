# Projeto IA - Classificacao e Regressao

Disciplina: Inteligencia Artificial Computacional - UNIFOR

Este projeto compara algoritmos supervisionados para duas tarefas:

- Classificacao: detectar se um site e phishing ou legitimo.
- Regressao: prever o preco de venda de casas.

O codigo principal esta em `projeto_IA.py` e foi mantido sem `sklearn` e sem `pandas`. A biblioteca `numpy` e usada para vetores, matrizes e operacoes matematicas.

## Avaliacao de aderencia aos enunciados

### AV2

O projeto atende aos pontos centrais do primeiro enunciado:

- Usa dois datasets, um de classificacao e um de regressao.
- Os datasets possuem mais de 1000 instancias e mais de 10 atributos.
- Implementa leitura e processamento de arquivos ARFF.
- Usa normalizacao Min-Max calculada apenas no treino de cada fold.
- Usa validacao cruzada 5-fold manual.
- Implementa KNN com distancia Euclidiana e Manhattan.
- Implementa Naive Bayes Gaussiano Univariado.
- Implementa Naive Bayes Gaussiano Multivariado com matriz de covariancia.
- Implementa Regressao Linear Multipla.
- Calcula medias, desvios padrao e tempos de treino/teste.
- Mostra tabelas comparativas.

### Extensao com redes neurais

Foram feitos ajustes para cobrir a etapa nova do trabalho:

- Perceptron Simples de classificacao implementado manualmente.
- Teste de diferentes taxas de aprendizado e numeros de epocas no Perceptron.
- Matriz de confusao acumulada nos folds para cada classificador.
- Metricas de classificacao: acuracia, sensibilidade/recall, especificidade, precisao e F1-score.
- MLP de regressao implementado manualmente com forward pass e backpropagation.
- Comparacao de topologias diferentes de MLP.
- Teste de diferentes quantidades de camadas ocultas, neuronios, learning rate, epocas e funcoes de ativacao.
- Metricas de regressao: MSE, RMSE, MAE, R2 e R2 ajustado.
- Tabelas com tempo de treinamento e tempo de teste.
- Resumos finais para apoiar a discussao sobre desempenho, generalizacao e custo computacional.

## Arquivos

| Arquivo | Funcao |
|---|---|
| `projeto_IA.py` | Codigo principal com leitura, modelos, metricas, validacao cruzada e tabelas. |
| `phpV5QYya.arff` | Dataset de classificacao. O alvo e a coluna `Result`. |
| `dataset.arff` | Dataset de regressao. O alvo e a coluna `price`. |
| `Projeto_IA___AV2 (2).pdf` | PDF do trabalho da AV2. |
| `SlidesIA.pdf` | Slides da apresentacao anterior. |

## Como rodar

No terminal, dentro da pasta do projeto:

```bash
python projeto_IA.py
```

ou:

```bash
python.exe projeto_IA.py
```

O script principal esta configurado com:

```python
CLASSIFICATION_ARFF = "phpV5QYya.arff"
REGRESSION_DATA = "dataset.arff"
REGRESSION_IS_ARFF = True
REGRESSION_TARGET = "price"
REGRESSION_DROP = ["id"]
K_FOLDS = 5
K_NEIGHBORS = 5
```

Observacao: a execucao completa pode demorar porque o KNN compara muitas amostras durante o teste, especialmente no dataset de regressao.

## Datasets usados

### Classificacao

Arquivo: `phpV5QYya.arff`

- Problema: deteccao de phishing.
- Instancias: 11055.
- Atributos preditores: 30.
- Alvo: `Result`.
- Classes: `-1` e `1`.

### Regressao

Arquivo: `dataset.arff`

- Problema: previsao de preco de casas.
- Instancias: 21613.
- Atributos no ARFF: 23.
- Atributos usados como entrada: 21.
- Alvo: `price`.
- Coluna removida das entradas: `id`.

O alvo `price` esta na segunda coluna do ARFF, por isso o codigo usa `REGRESSION_TARGET = "price"`.

## Modelos de classificacao

O script avalia:

- KNN com distancia Euclidiana.
- KNN com distancia Manhattan.
- Naive Bayes Gaussiano Univariado.
- Naive Bayes Gaussiano Multivariado.
- Perceptron Simples com `learning_rate=0.01` e `epochs=50`.
- Perceptron Simples com `learning_rate=0.01` e `epochs=100`.
- Perceptron Simples com `learning_rate=0.05` e `epochs=100`.
- Perceptron Simples com `learning_rate=0.10` e `epochs=200`.

Metricas exibidas:

- Matriz de confusao.
- Acuracia.
- Sensibilidade, tambem chamada de recall.
- Especificidade.
- Precisao.
- F1-score.
- Tempo medio de treino.
- Tempo medio de teste.

## Modelos de regressao

O script avalia:

- KNN com distancia Euclidiana.
- KNN com distancia Manhattan.
- Regressao Linear Multipla.
- MLP com topologia `(8)`, ReLU, `learning_rate=0.010`, `epochs=250`.
- MLP com topologia `(16)`, tanh, `learning_rate=0.010`, `epochs=300`.
- MLP com topologia `(16, 8)`, ReLU, `learning_rate=0.005`, `epochs=350`.
- MLP com topologia `(32, 16)`, ReLU, `learning_rate=0.003`, `epochs=400`.

Metricas exibidas:

- MSE.
- RMSE.
- MAE.
- R2.
- R2 ajustado.
- Tempo medio de treino.
- Tempo medio de teste.

## Funcionamento resumido

1. Le os arquivos `.arff`.
2. Separa atributos `X` e alvo `y`.
3. Codifica rotulos de classificacao.
4. Cria os folds da validacao cruzada.
5. Em cada fold, normaliza os atributos usando apenas o treino.
6. Treina cada modelo manualmente.
7. Testa cada modelo.
8. Calcula metricas e tempos.
9. Imprime tabelas comparativas e resumos para discussao.

## Pontos para defender na apresentacao

- O Perceptron e um classificador linear; por isso seu desempenho depende da separabilidade dos dados.
- Learning rate alto pode acelerar ajustes, mas tambem pode gerar oscilacao.
- Mais epocas podem melhorar convergencia, mas aumentam o custo de treino.
- O MLP permite modelar relacoes nao lineares, mas pode custar mais tempo de treinamento.
- Arquiteturas maiores nem sempre generalizam melhor; elas podem aumentar custo computacional e risco de overfitting.
- As metricas de erro da regressao devem ser analisadas junto com R2/R2 ajustado e tempo de execucao.
- A melhor configuracao deve equilibrar desempenho, generalizacao e eficiencia, nao apenas maximizar uma unica metrica.

## Referencias

- Cover, T. and Hart, P. Nearest Neighbor Pattern Classification.
- Bishop, C. M. Pattern Recognition and Machine Learning.
- Murphy, K. P. Machine Learning: A Probabilistic Perspective.
- OpenML: https://www.openml.org/
