# cnn-ia-2025

Projeto de Convolutional Neural Network (CNN) para a disciplina de IA

## Objetivo

Implementar uma rede neural convolucional (CNN) com uso permitido de bibliotecas especializadas, como TensorFlow ou PyTorch.

## Passos

1. **Implementar uma CNN com:**

   - Várias camadas convolucionais e de pooling. [ ]

   - Camadas densas ao final. [ ]

   - Critério de parada definido. [ ]

   - Lógica completa de treinamento, teste e avaliação. [ ]

2. **Realizar dois tipos de tarefa:**

   - Multiclasse: reconhecer todas as classes do conjunto. [ ]

   - Binária: escolher duas classes ou dois conjuntos de classes para classificação. [ ]

3. **Conjunto de dados:**

   Recomendado: MNIST (imagens de dígitos manuscritos de 0 a 9)
   Link: https://www.tensorflow.org/datasets/catalog/mnist?hl=pt-br

4. **Formas de entrada da CNN:**

   Direto com imagem bruta
   OU

   Com descritores extraídos (como HOG, LBP, Haar, etc) — usar bibliotecas especializadas para isso.

5. **Gerar arquivos de saída com:**

   - Hiperparâmetros da arquitetura e da inicialização. [ ]

   - Pesos iniciais e finais. [ ]

   - Erro por iteração. [ ]

   - Saídas da rede para os dados de teste. [ ]

6. **Gravar o vídeo 3 (CNN):**

   Tempo: mínimo 5 minutos, máximo 15 minutos.

   Mostrar no vídeo:

   - Como foi modelada a entrada da CNN.

   - Como foram aplicadas as camadas de kernel e pooling.

   - Como foram modeladas as camadas densas.

   - A lógica do treinamento e critério de parada.

   - O cálculo de erro.

   - A resposta da rede (acertos/erros).

   - Testes com MNIST ou conjunto CARACTERES.

   - Matriz de confusão para avaliar o resultado.

## Gerenciamento de bibliotecas

Utilize `pip install -r requirements.txt` para instalar todas as bibliotecas necessárias.

Caso haja a necessidade de introduzir alguma outra biblioteca, utilize
`pip freeze > requirements.txt` (após seu pip install) para adicionar a dependência à lista de requisitos do programa.

## Ambiente Virtual

1. **Crie o Ambiente Virtual**

```bash
python3 -m venv .venv
```

2. **Ative o Ambiente Virtual**

```bash
source .venv/bin/activate
```

Para desativar: `deactivate`
