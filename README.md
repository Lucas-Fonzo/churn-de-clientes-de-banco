# Regressão Logística

**Objetivo:**

O principal objetivo do algoritmo de regressão logística é modelar a probabilidade de ocorrência de um determinado evento binário (ou multiclasse), onde a variável dependente é categórica. No caso binário, o objetivo é prever a probabilidade de uma amostra pertencer a uma das duas classes.

**Exemplo:**

Um exemplo clássico de aplicação da regressão logística é a previsão do **churn** de um cliente. A regressão logística pode ser utilizada para calcular uma medida que interpretamos como a probabilidade de um cliente deixar de utilizar o serviço, baseada em suas características, como tempo de uso, saldo de conta, entre outras.

**Funcionamento:**

O funcionamento da regressão logística envolve a modelagem da relação entre as variáveis independentes (features) e a probabilidade de uma amostra pertencer a uma das classes da variável dependente. Diferente da regressão linear, que prevê valores contínuos, a regressão logística prevê a probabilidade de uma classe, usando a função logística (ou sigmoide):

\[
P(y = 1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_n X_n)}}
\]

Onde:

- \( P(y = 1|X) \) é a probabilidade de a variável dependente \( y \) ser 1 dado os valores das variáveis independentes \( X_1, X_2, \dots, X_n \).
- \( \beta_0 \) é o intercepto.
- \( \beta_1, \beta_2, \dots, \beta_n \) são os coeficientes que quantificam a contribuição de cada variável independente \( X_1, X_2, \dots, X_n \).

Essa função transforma a combinação linear das variáveis independentes em um valor entre 0 e 1, que pode ser interpretado como uma probabilidade. A decisão de classificação final é feita aplicando um limiar (geralmente 0,5) para determinar a classe prevista.

Para ajustar os coeficientes \( \beta \) de maneira que as previsões sejam o mais precisas possível, utilizamos uma função de custo específica para classificação binária, chamada de **Log-Loss** (ou entropia cruzada). A **Log-Loss** mede o erro entre as probabilidades previstas pelo modelo e os rótulos reais. A função de custo **Log-Loss** é calculada da seguinte forma:

\[
\text{Log-Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
\]

Onde:

- \( y_i \) é o rótulo real (0 ou 1) da amostra \( i \),
- \( p_i \) é a probabilidade prevista pelo modelo para a classe 1,
- \( N \) é o número de amostras.

A **Log-Loss** penaliza fortemente as previsões incorretas com alta confiança, ou seja, quando o modelo prevê uma probabilidade muito alta para a classe errada. Quanto menor o valor da **Log-Loss**, melhor o desempenho do modelo.

A otimização dos coeficientes \( \beta \) para minimizar a **Log-Loss** é feita usando métodos como o gradiente descendente. Nesse processo, o gradiente da função de custo é calculado em relação aos coeficientes, e esses coeficientes são ajustados iterativamente na direção que reduz o erro. O modelo atualiza os coeficientes até que a **Log-Loss** atinja um mínimo, o que indica que o modelo encontrou a combinação de coeficientes que melhor se ajusta aos dados de treino.

### Escoragem:
Com o modelo treinado, a previsão de novas amostras envolve calcular a probabilidade de cada amostra pertencer a uma classe. A partir dessa probabilidade, podemos aplicar um limiar para classificar a amostra em uma das classes. Por exemplo, se a probabilidade for maior que 0,5, a amostra é classificada como pertencente à classe 1; caso contrário, é classificada como classe 0.

### Vantagens:
- **Simplicidade e Interpretabilidade**: A regressão logística é fácil de interpretar, especialmente em problemas binários, onde os coeficientes podem ser relacionados diretamente à variação na probabilidade de um evento.
- **Probabilidades como Saída**: A regressão logística não apenas fornece uma classificação, mas também a probabilidade associada a essa classificação, o que pode ser útil em diversas aplicações.

### Desvantagens:
- **Assunção de Linearidade**: A regressão logística assume que existe uma relação linear entre as variáveis independentes e o logit da variável dependente. Se essa suposição não for verdadeira, o modelo pode não ser adequado.
- **Limitada a Problemas Lineares**: Como a regressão linear, a regressão logística pode ter dificuldades para modelar relações complexas e não lineares entre as variáveis.

### Particularidades:
- **Regularização**: Assim como na regressão linear, técnicas de regularização como L1 (Lasso) e L2 (Ridge) podem ser utilizadas para evitar overfitting e melhorar a generalização do modelo.
- **Multicolinearidade**: A presença de multicolinearidade (correlações altas entre variáveis independentes) pode prejudicar o desempenho do modelo. Técnicas como remoção de variáveis correlacionadas ou a aplicação de regularização podem ajudar a mitigar esse problema.
