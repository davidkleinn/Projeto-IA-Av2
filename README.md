# Machine Learning do zero: classificação e regressão só com NumPy

Projeto da disciplina **Inteligência Artificial Computacional** (Ciência da Computação, UNIFOR).

Implementei do zero, **sem scikit-learn e sem pandas**, os principais algoritmos supervisionados e os comparei em dois problemas reais com validação cruzada:

- **Classificação:** detectar se um site é *phishing* ou legítimo (11.055 instâncias, 30 atributos).
- **Regressão:** prever o preço de venda de casas (21.613 instâncias, 21 atributos de entrada).

Tudo está em um único script (`projeto_IA.py`), incluindo o leitor de arquivos ARFF, a normalização, a validação cruzada, as métricas e os modelos.

## Resultados

Validação cruzada 5-fold. Valores são média ± desvio padrão entre os folds. A execução completa está em [`results/execucao_completa.txt`](results/execucao_completa.txt).

### Classificação (phishing)

| Modelo | Acurácia | F1 (macro) | Tempo de treino | Tempo de teste |
|---|---|---|---|---|
| **KNN k=5, Manhattan** | **0,9565 ± 0,0022** | **0,9558 ± 0,0022** | ~0 s | 3,03 s |
| KNN k=5, Euclidiana | 0,9455 ± 0,0017 | 0,9447 ± 0,0018 | ~0 s | 0,22 s |
| Perceptron (lr=0,10; 200 épocas) | 0,9061 ± 0,0112 | 0,9033 ± 0,0129 | 2,12 s | 0,0002 s |
| Perceptron (lr=0,05; 100 épocas) | 0,9036 ± 0,0138 | 0,9012 ± 0,0156 | 1,01 s | 0,0003 s |
| Perceptron (lr=0,01; 50 épocas) | 0,9000 ± 0,0301 | 0,8963 ± 0,0334 | 0,50 s | 0,0002 s |
| Perceptron (lr=0,01; 100 épocas) | 0,8763 ± 0,0385 | 0,8746 ± 0,0376 | 1,02 s | 0,0004 s |
| Naive Bayes multivariado | 0,7908 ± 0,0093 | 0,7890 ± 0,0093 | 0,004 s | 0,16 s |
| Naive Bayes univariado | 0,6185 ± 0,0063 | 0,5895 ± 0,0056 | 0,003 s | 0,06 s |

Matriz de confusão acumulada do melhor modelo (KNN Manhattan; linhas = classe real, colunas = prevista):

|  | Previsto −1 | Previsto 1 |
|---|---|---|
| **Real −1** | 4589 | 309 |
| **Real 1** | 172 | 5985 |

### Regressão (preço de casas)

| Modelo | R² | RMSE | MAE | Tempo de treino | Tempo de teste |
|---|---|---|---|---|---|
| **KNN k=5, Manhattan** | **0,7626 ± 0,0147** | **178.724** | **92.970** | ~0 s | 9,04 s |
| KNN k=5, Euclidiana | 0,7197 ± 0,0132 | 194.200 | 103.600 | ~0 s | 0,72 s |
| Regressão linear múltipla | 0,6996 ± 0,0103 | 201.100 | 125.800 | 0,012 s | 0,0002 s |
| MLP (16), tanh | 0,4326 ± 0,0157 | 276.400 | 165.700 | 1,30 s | 0,0007 s |
| MLP (16, 8), ReLU | 0,2356 ± 0,0159 | 320.800 | 204.800 | 1,69 s | 0,0005 s |
| MLP (32, 16), ReLU | 0,1402 ± 0,0155 | 340.100 | 212.500 | 3,89 s | 0,0010 s |
| MLP (8), ReLU | 0,0048 ± 0,0018 | 366.000 | 233.800 | 0,50 s | 0,0005 s |

As tabelas completas (MSE, R² ajustado, desvios) estão no arquivo de resultados.

### O que os números mostram

- **Na classificação, o KNN com distância de Manhattan foi o melhor** (95,65% de acurácia). O custo é no teste: ele leva cerca de 14 vezes mais tempo que o KNN Euclidiano, porque a Euclidiana usa a identidade com produto matricial (`‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b`), enquanto a Manhattan não tem esse atalho e precisa montar um tensor 3D de diferenças em lotes menores.
- **O Perceptron fica em torno de 90%** com tempo de predição quase nulo. Isso indica que os dados são razoavelmente (mas não totalmente) separáveis por um hiperplano. Ele ganha em custo de inferência e perde em acurácia para o KNN.
- **O Naive Bayes foi o pior.** O univariado assume atributos independentes e erra a maior parte da classe 1 (4.209 de 6.157 exemplos classificados como −1). O multivariado, ao usar a matriz de covariância, melhora cerca de 17 pontos percentuais.
- **Na regressão, o KNN também venceu** (R² 0,76), à frente da regressão linear (0,70). Uma explicação plausível é que o preço dependa de relações locais e não lineares entre os atributos, que o KNN captura melhor que um modelo linear (não testei isso diretamente).
- **Os MLPs ficaram abaixo da regressão linear**, e isso é um resultado negativo que registrei de propósito. A implementação usa gradiente descendente em lote completo, com poucas épocas (250 a 400) e *clipping* de gradiente, e a configuração mais simples (tanh, 1 camada) foi a melhor. A hipótese mais provável é subtreinamento, não falta de capacidade do modelo. Não testei essa hipótese; os próximos passos abaixo seriam o primeiro experimento.
- Nas médias macro, **sensibilidade e especificidade saem iguais** em problema binário (a especificidade de uma classe é a sensibilidade da outra), por isso as duas colunas coincidem no log.

## O que está implementado

| Área | Implementação manual |
|---|---|
| Dados | Leitor de arquivos ARFF (com remoção de colunas e escolha do alvo), leitor de CSV |
| Pré-processamento | Normalização Min-Max calculada **apenas no treino de cada fold**, para evitar vazamento de dados |
| Validação | K-Fold com 5 partições, com semente fixa para reprodutibilidade |
| Classificadores | KNN (Euclidiana e Manhattan), Naive Bayes Gaussiano univariado e multivariado, Perceptron Simples |
| Regressores | KNN, Regressão Linear Múltipla (pseudo-inversa), MLP com backpropagation (ReLU, tanh e sigmoide) |
| Métricas de classificação | Acurácia, precisão, sensibilidade, especificidade, F1 macro, matriz de confusão |
| Métricas de regressão | MSE, RMSE, MAE, R², R² ajustado |
| Análise | Tempo de treino e de teste por modelo e resumos comparativos |

## Como rodar

Requisitos: Python 3 e NumPy.

```bash
git clone https://github.com/davidkleinn/Projeto-IA-Av2.git
cd Projeto-IA-Av2
pip install -r requirements.txt
python projeto_IA.py
```

A execução completa leva alguns minutos (a maior parte é o KNN Manhattan na regressão). As configurações (arquivos, atributo alvo, número de folds e de vizinhos) ficam no bloco final de `projeto_IA.py`.

## Estrutura

```text
.
├── projeto_IA.py            # código completo: dados, modelos, métricas e execução
├── requirements.txt
├── data/
│   ├── phpV5QYya.arff       # classificação (phishing), alvo: Result
│   └── dataset.arff         # regressão (preço de casas), alvo: price
├── docs/
│   └── SlidesIA.pdf         # slides da apresentação
└── results/
    └── execucao_completa.txt  # saída completa da execução
```

## Datasets

Ambos vêm do [OpenML](https://www.openml.org/).

- **Phishing** (`phpV5QYya.arff`): 11.055 instâncias, 30 atributos preditores, classes `-1` e `1`.
- **Preço de casas** (`dataset.arff`): 21.613 instâncias, 23 colunas no ARFF e 21 usadas como entrada (a coluna `id` é removida), alvo `price` (de 75 mil a 7,7 milhões, média de 540 mil).

## Próximos passos

- Treinar o MLP com mini-batches e o otimizador Adam, e comparar com a versão atual para confirmar (ou refutar) a hipótese de subtreinamento.
- Aplicar log no preço da regressão, já que a distribuição é assimétrica.
- Reduzir o custo da distância de Manhattan no KNN (por exemplo, com uma estrutura de busca como KD-Tree) para diminuir o tempo de teste.
- Adicionar testes automatizados para as métricas, comparando com valores calculados à mão.

## Referências

- Cover, T. e Hart, P. *Nearest Neighbor Pattern Classification.*
- Bishop, C. M. *Pattern Recognition and Machine Learning.*
- Murphy, K. P. *Machine Learning: A Probabilistic Perspective.*
