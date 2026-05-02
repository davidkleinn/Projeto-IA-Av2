# README - Projeto IA: Classificacao e Regressao

Disciplina: Inteligencia Artificial Computacional - UNIFOR

## O que este projeto faz

Este projeto compara algoritmos de aprendizado de maquina supervisionado em dois problemas:

- Classificacao: prever se um site e phishing ou legitimo.
- Regressao: prever o preco de venda de uma casa.

O codigo foi implementado manualmente, sem `sklearn` e sem `pandas`, como pedido no PDF da atividade. A biblioteca `numpy` e usada para operacoes matematicas.

## Arquivos

| Arquivo | Funcao |
|---|---|
| `projeto_IA.py` | Codigo principal do trabalho. Roda todos os algoritmos, metricas e tabelas. |
| `phpV5QYya.arff` | Dataset de classificacao. O alvo e a coluna `Result`. |
| `dataset.arff` | Dataset de regressao. O alvo e a coluna `price`. |
| `knn_algoritmo.py` | Versao separada/simplificada do KNN para estudo. |
| `Naive_Bayes.py` | Versao separada/simplificada do Naive Bayes univariado para estudo. |
| `aplication_KNN (1).py` | Exemplo pequeno usando KNN no dataset de phishing. |

## Como rodar

No terminal, dentro da pasta do projeto:

```bash
python.exe projeto_IA.py
```

Se o seu terminal aceitar `python`, tambem funciona:

```bash
python projeto_IA.py
```

O script principal ja esta configurado para usar:

```python
CLASSIFICATION_ARFF = "phpV5QYya.arff"
REGRESSION_DATA = "dataset.arff"
REGRESSION_IS_ARFF = True
REGRESSION_TARGET = "price"
REGRESSION_DROP = ["id"]
K_FOLDS = 5
K_NEIGHBORS = 5
```

## Datasets usados

### Dataset de classificacao

Arquivo: `phpV5QYya.arff`

- Problema: detectar phishing.
- Instancias: 11055.
- Atributos preditores: 30.
- Alvo: `Result`.
- Classes: `-1` e `1`.

### Dataset de regressao

Arquivo: `dataset.arff`

- Problema: prever preco de casas.
- Instancias: 21613.
- Atributos no ARFF: 23.
- Atributos usados como entrada: 21.
- Alvo: `price`.
- Coluna removida das entradas: `id`.

Observacao importante: neste dataset, o alvo `price` nao esta na ultima coluna. Ele esta na segunda coluna. Por isso o codigo usa `REGRESSION_TARGET = "price"`.

## O que o codigo principal faz

O arquivo `projeto_IA.py` segue estes passos:

1. Le os arquivos `.arff`.
2. Separa os dados em `X` e `y`.
3. Normaliza os atributos com Min-Max.
4. Divide os dados com validacao cruzada 5-fold.
5. Treina os algoritmos.
6. Testa os algoritmos.
7. Calcula metricas de desempenho.
8. Mede tempo de treino e tempo de teste.
9. Mostra tabelas comparativas com media e desvio padrao.

`X` representa os atributos de entrada.

`y` representa a resposta que o modelo tenta prever.

## Algoritmos implementados

### Classificacao

O codigo compara:

- KNN com distancia Euclidiana.
- KNN com distancia Manhattan.
- Naive Bayes Gaussiano Univariado.
- Naive Bayes Gaussiano Multivariado.

### Regressao

O codigo compara:

- KNN com distancia Euclidiana.
- KNN com distancia Manhattan.
- Regressao Linear Multipla.

## KNN

KNN significa K-Nearest Neighbors.

A ideia e olhar os `k` exemplos mais proximos de uma nova amostra.

No projeto, `k = 5`.

Para classificacao:

- O modelo olha os 5 vizinhos mais proximos.
- A classe mais frequente entre eles vence.

Para regressao:

- O modelo olha os 5 vizinhos mais proximos.
- A previsao e a media dos valores desses vizinhos.

O codigo usa duas distancias:

### Distancia Euclidiana

E a distancia em linha reta:

```text
d = sqrt((x1-y1)^2 + (x2-y2)^2 + ... + (xn-yn)^2)
```

### Distancia Manhattan

Soma as diferencas absolutas:

```text
d = |x1-y1| + |x2-y2| + ... + |xn-yn|
```

## Naive Bayes

Naive Bayes e um classificador probabilistico.

Ele calcula qual classe e mais provavel para uma amostra.

### Naive Bayes Univariado

Calcula media e variancia de cada atributo separadamente para cada classe.

### Naive Bayes Multivariado

Usa vetor de medias e matriz de covariancia. A matriz de covariancia permite considerar relacoes entre atributos.

O codigo adiciona um valor pequeno na diagonal da matriz para evitar erro de matriz singular.

## Regressao Linear Multipla

A regressao linear multipla tenta prever um valor numerico usando uma combinacao linear dos atributos:

```text
y = b0 + b1*x1 + b2*x2 + ... + bn*xn
```

No codigo, a solucao e calculada com pseudo-inversa:

```python
theta = np.linalg.pinv(X_b) @ y_train
```

Isso e equivalente a resolver o problema linear de forma manual, mas e mais estavel do que inverter diretamente `X.T @ X`, principalmente quando existem atributos correlacionados.

## Normalizacao Min-Max

KNN depende de distancia. Se um atributo tem valores muito grandes, ele pode dominar a distancia.

Por isso, o codigo normaliza os atributos:

```text
x_normalizado = (x - minimo) / (maximo - minimo)
```

O minimo e o maximo sao calculados apenas no treino e aplicados no teste, evitando vazamento de dados.

## Validacao Cruzada 5-Fold

O codigo usa 5-fold cross-validation:

1. Divide o dataset em 5 partes.
2. Usa 4 partes para treino.
3. Usa 1 parte para teste.
4. Repete o processo 5 vezes.
5. Calcula media e desvio padrao das metricas.

Isso deixa a avaliacao mais confiavel do que uma divisao unica treino/teste.

## Metricas

### Classificacao

O codigo calcula:

- Acuracia.
- Precisao macro.
- Recall macro.
- F1-score macro.
- Tempo medio de treino.
- Tempo medio de teste.

### Regressao

O codigo calcula:

- R2.
- R2 ajustado.
- Tempo medio de treino.
- Tempo medio de teste.

## Resultado esperado

Ao rodar `projeto_IA.py`, o programa imprime duas tabelas:

- Tabela comparativa de classificadores.
- Tabela comparativa de regressores.

Nos testes feitos, o melhor desempenho foi:

- Classificacao: KNN com distancia Manhattan.
- Regressao: KNN com distancia Manhattan.

Porem, KNN Manhattan tambem foi mais lento no tempo de teste. Essa e uma boa conclusao para apresentar: melhor desempenho nem sempre significa menor tempo de execucao.

## Conclusao para apresentar

Uma conclusao simples e correta:

```text
O objetivo do trabalho foi comparar algoritmos supervisionados em classificacao e regressao.
Para classificacao, foi usado um dataset de phishing. Para regressao, foi usado um dataset de preco de casas.
Todos os algoritmos foram implementados manualmente, sem sklearn e sem pandas.
Foi usada validacao cruzada 5-fold, calculando media, desvio padrao e tempos de treino e teste.
O KNN com distancia Manhattan obteve o melhor desempenho nos dois problemas, mas teve maior tempo de teste.
Modelos como Naive Bayes e Regressao Linear foram mais rapidos, porem menos precisos.
```

## Problemas comuns

| Problema | Causa | Solucao |
|---|---|---|
| `FileNotFoundError` | Arquivo de dataset nao esta na pasta | Verifique se `phpV5QYya.arff` e `dataset.arff` estao junto de `projeto_IA.py`. |
| Execucao lenta | KNN calcula distancias para muitas amostras | Aguarde a execucao ou reduza `K_FOLDS` apenas para teste. |
| Resultado estranho na regressao | Alvo errado | Confirme que `REGRESSION_TARGET = "price"`. |
| Erro de matriz singular | Covariancia ou regressao com colunas correlacionadas | O codigo ja usa regularizacao no Bayes multivariado e pseudo-inversa na regressao linear. |

## Referencias

- Cover, T. and Hart, P. Nearest Neighbor Pattern Classification.
- Bishop, C. M. Pattern Recognition and Machine Learning.
- Murphy, K. P. Machine Learning: A Probabilistic Perspective.
- OpenML: https://www.openml.org/
