# PTB-XL · Classificação de ECGs e estimativa de idade

Projeto da Unidade 2 da disciplina **DCA3606 — Inteligência Artificial**, da Universidade Federal do Rio Grande do Norte (UFRN). [Enunciado do trabalho](ProjetoUnidade2.pdf)

## Propósito

O trabalho investiga quanto o sinal de um eletrocardiograma (ECG) permite distinguir exames **normais de alterados** e **estimar a idade cronológica** do paciente, em anos. O objetivo é comparar a qualidade das previsões, o custo computacional e o efeito de substituir o sinal completo por uma representação compacta aprendida por um autoencoder.

## Dados e metodologia

O dataset reúne 21.799 ECGs de 12 derivações de 18.869 pacientes. O projeto usa a versão dos sinais de 100 Hz, com 10 segundos e 1.000 amostras por derivação. [PTB-XL v1.0.3 no PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)

- **Classificação binária.** A classe 0 (Normal) contém exames marcados como `NORM` sem outras superclasses diagnósticas. A classe 1 (Alterado) contém exames com pelo menos uma das superclasses `MI`, `STTC`, `CD` ou `HYP`, sem `NORM`. Exames ambíguos ou sem diagnóstico elegível ficam fora da tarefa.
- **Regressão.** O alvo é a idade cronológica em anos. O preparo exclui idades inválidas e o valor anonimizado `300`.

Os modelos recebem apenas os sinais do ECG. O pré-processamento verifica a qualidade dos registros e calcula a média e o desvio padrão de cada derivação usando apenas o treino. A mesma transformação é aplicada à validação e ao teste. Os folds oficiais mantêm cada paciente em uma única partição.

| Partição | Folds | ECGs para classificação | ECGs para regressão |
| --- | --- | ---: | ---: |
| Treino | 1–8 | 16.730 | 17.200 |
| Validação | 9 | 2.105 | 2.141 |
| Teste final | 10 | 2.107 | 2.164 |
| **Total após o preparo** | | **20.942** | **21.505** |

### Modelos comparados

| Abordagem | Como usa o ECG |
| --- | --- |
| MLP | Recebe o sinal normalizado como vetor de 12.000 valores. |
| CNN 1D | Aplica convoluções no tempo às 12 derivações. |
| ResNet 1D | Usa convoluções temporais e conexões residuais. |
| Random Forest com entrada original | Recebe o mesmo vetor de 12.000 valores usado pela MLP. |
| Random Forest com embeddings | Recebe 64 valores produzidos pelo encoder congelado. |

O autoencoder aprende a reconstruir os sinais do treino sem usar idade ou diagnóstico. O fold 9 orienta a seleção dos checkpoints das redes e do autoencoder. Para medir o efeito da representação, os dois Random Forests de cada tarefa usam os mesmos exames e hiperparâmetros.

No fold 10, a avaliação mede acurácia, precisão, recall e F1 macro na classificação, além de MAE, MSE, RMSE e R² na regressão. As referências são a classe majoritária e a idade média, calculadas no treino. O notebook final gera tabelas, matrizes de confusão, gráficos de resíduos e uma comparação de tempos de treino e inferência que inclui o custo do encoder.

## Notebooks em ordem de execução

| Etapa | Notebook | O que faz |
| --- | --- | --- |
| 1 | [Análise exploratória](01_analise_exploratoria.ipynb) | Examina os sinais, os diagnósticos, as idades e a distribuição por fold. |
| 2 | [Pré-processamento](02_preprocessamento.ipynb) | Audita os sinais, constrói os alvos e salva a matriz normalizada e seus índices. |
| 3 | [Treinamento das redes](03_treinamento_deep_learning.ipynb) | Treina MLP, CNN 1D e ResNet 1D nas duas tarefas e salva checkpoints e históricos. |
| 4 | [Autoencoder e Random Forest](04_autoencoder_modelos_classicos.ipynb) | Treina o autoencoder, extrai embeddings e ajusta os Random Forests com as duas entradas. |
| 5 | [Avaliação final](05_avaliacao_final.ipynb) | Carrega os modelos salvos e gera resultados para o fold 10. |

## Como executar

Dataset, sinais preparados, checkpoints e resultados são mantidos fora do Git. Para reproduzir o trabalho, use Python com Jupyter para as etapas 1 e 2 e Google Colab com Drive para as etapas 3 a 5. Use GPU nas etapas de redes quando disponível.

### Gerar os dados preparados e treinar

1. Baixe o PTB-XL v1.0.3 e extraia os dados na estrutura `ptb-xl-1.0.3/<pasta-do-dataset>/` dentro do projeto. Essa pasta deve conter `ptbxl_database.csv`, `scp_statements.csv` e `records100/`. [Download no PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
2. Em um ambiente Python, instale as dependências e abra os notebooks 01 e 02 a partir da raiz do projeto.

   ```bash
   python -m pip install jupyterlab ipykernel numpy pandas matplotlib wfdb
   python -m jupyterlab
   ```

3. Copie os notebooks, a pasta `src/` e `resultados/preprocessamento/` para `MyDrive/IA-2026`. No Colab, as etapas 3 a 5 usam NumPy, pandas, Matplotlib, PyTorch, scikit-learn, joblib e IPython.
4. Para treinar do zero, abra os notebooks 03 e 04 no Colab com `MODO_TESTE=False` e `EXECUTAR_TREINO_COMPLETO=True`. No 03, ajuste `RAIZ_PROJETO_COLAB` para `/content/drive/MyDrive/IA-2026`. As duas etapas geram pastas `treino_completo_v1` em seus respectivos diretórios de resultados.

### Avaliar os modelos já treinados

Com os artefatos abaixo e os módulos de `src/` em `MyDrive/IA-2026`, abra o notebook 05 no Colab, autorize a montagem do Drive e execute todas as células. Use a mesma versão do scikit-learn que salvou os modelos `.joblib`.

| Diretório dentro de `MyDrive/IA-2026` | Artefatos |
| --- | --- |
| `resultados/preprocessamento/` | `normalizacao.json`, `manifesto.csv`, `classificacao.csv`, `regressao.csv` e `sinais_normalizados.float32.bin`. |
| `resultados/deep_learning/treino_completo_v1/` | Checkpoints selecionados, configurações, históricos e curvas das seis combinações de arquitetura e tarefa. |
| `resultados/autoencoder_classicos/treino_completo_v1/` | Checkpoint do autoencoder, quatro Random Forests, embeddings de treino e validação, configurações, métricas e gráficos. |

O notebook 05 confere a integridade dos arquivos antes de avaliar o fold 10. Cada execução gera uma pasta em `resultados/avaliacao_final/` com tabelas CSV, gráficos PNG, previsões por ECG e uma discussão baseada nas métricas calculadas.
