# neural-network-experiment

# Experimento com Rede Neural

Este repositório contém um experimento desenvolvido para a disciplina de **Deep Learning**, com o objetivo de analisar os efeitos de diferentes funções de ativação e funções de perda em uma rede neural implementada manualmente utilizando NumPy.

O experimento utiliza o dataset **Two Moons**, disponibilizado pelo Scikit-learn, para um problema de classificação binária.

## Objetivo

O objetivo do experimento é realizar modificações na implementação original da rede neural e comparar como diferentes funções de ativação e perda afetam seu treinamento e sua acurácia.

Foram avaliadas três configurações:

1. **Modelo Original:** Sigmoid + Mean Squared Error (MSE)
2. **Experimento 1:** Tanh + Mean Squared Error (MSE)
3. **Experimento 2:** Tanh + Binary Cross-Entropy (BCE)

## Configuração do Experimento

A mesma arquitetura básica foi mantida nos três experimentos:

- 2 variáveis de entrada;
- 2 neurônios na camada oculta;
- 1 neurônio na camada de saída;
- taxa de aprendizado de `0.1`;
- 10.000 iterações de treinamento;
- 100 amostras geradas pelo dataset Two Moons.

Para permitir que os experimentos fossem reproduzidos sob as mesmas condições, foram utilizadas sementes fixas para a geração do dataset e para a inicialização dos parâmetros:

```python id="5wgy30"
random_state=42
np.random.seed(42)
```

Dessa forma, os experimentos utilizam o mesmo conjunto de dados e os mesmos valores iniciais para pesos e biases.

## Modificação 1 — Tanh 

Na primeira modificação, a função de ativação da camada oculta foi alterada de **Sigmoid para Tanh**.

Na implementação original, a derivada da Sigmoid é calculada como:

`y * (1 - y)`

Com a utilização da Tanh, o cálculo passa a ser:

`1 - y²`

A camada de saída continua utilizando a função Sigmoid.

## Modificação 2 — Binary Cross-Entropy

Na segunda modificação, a função Tanh foi mantida na camada oculta e a função de perda **Mean Squared Error (MSE)** foi substituída pela **Binary Cross-Entropy (BCE)**.

A Binary Cross-Entropy é definida como:

`L = -[d log(y) + (1-d) log(1-y)]`

Como a camada de saída utiliza Sigmoid em conjunto com Binary Cross-Entropy, o cálculo do gradiente da saída pode ser simplificado para:

`y - d`

## Resultados

| Modelo | Ativação da camada oculta | Função de perda | Acurácia |
|---|---|---|---:|
| Original | Sigmoid | MSE | 89% |
| Experimento 1 | Tanh | MSE | 85% |
| Experimento 2 | Tanh | Binary Cross-Entropy | 90% |

Nas condições controladas do experimento, a substituição da função Sigmoid pela Tanh, mantendo MSE como função de perda, não resultou em aumento da acurácia. O modelo original apresentou 89%, enquanto a configuração Tanh + MSE apresentou 85%.

Ao substituir também a função de perda por Binary Cross-Entropy, a rede atingiu 90% de acurácia, que corresponde ao maior resultado observado entre as três configurações.

Os resultados obtidos são específicos para o dataset, arquitetura, inicialização e parâmetros utilizados neste experimento. Portanto, não indicam superioridade de uma determinada função de ativação ou função de perda.

## Notebook

A implementação completa do experimento, incluindo o treinamento das três configurações, comparação das acurácias, curvas de perda e resultados obtidos, está disponível no notebook:

`neural_network_experiment.ipynb`

## Tecnologias Utilizadas

- Python
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab
