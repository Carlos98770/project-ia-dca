"""Pré-processamento reproduzível do PTB-XL, sem treinamento de modelos.

O arquivo de sinais gerado contém uma única cópia float32 normalizada de cada
ECG elegível e aprovado. Os CSVs das tarefas referenciam linhas desse arquivo.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


DERIVACOES = ('I', 'II', 'III', 'AVR', 'AVL', 'AVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
SUPERCLASSES = frozenset(('NORM', 'MI', 'STTC', 'CD', 'HYP'))
ALTERACOES = frozenset(('MI', 'STTC', 'CD', 'HYP'))
FORMA_SINAL = (1000, 12)
FREQUENCIA_HZ = 100


def localizar_dataset(raiz_projeto: Path) -> Path:
    """Localiza a extração local sem depender de um caminho absoluto fixo."""
    candidatos = [raiz_projeto, *raiz_projeto.glob('ptb-xl-1.0.3/*')]
    for pasta in candidatos:
        if all((pasta / nome).exists() for nome in
               ('ptbxl_database.csv', 'scp_statements.csv', 'records100')):
            return pasta
    raise FileNotFoundError('PTB-XL extraído não encontrado a partir da raiz do projeto.')


def ler_codigos_scp(valor: object) -> dict:
    """Converte o dicionário SCP da EDA sem executar texto como código."""
    if pd.isna(valor):
        return {}
    try:
        codigos = ast.literal_eval(valor)
    except (ValueError, SyntaxError, TypeError):
        return {}
    return codigos if isinstance(codigos, dict) else {}


def _particao(fold: object) -> str:
    if pd.isna(fold):
        raise ValueError('Fold ausente no CSV original.')
    if int(fold) != fold or not 1 <= int(fold) <= 10:
        raise ValueError(f'Fold oficial inválido: {fold!r}')
    return 'treino' if fold <= 8 else ('validacao' if fold == 9 else 'teste')


def construir_metadados(raiz_dados: Path) -> pd.DataFrame:
    """Reproduz exatamente a regra de alvo e de idade definida na EDA."""
    df = pd.read_csv(raiz_dados / 'ptbxl_database.csv')
    scp = pd.read_csv(raiz_dados / 'scp_statements.csv', index_col=0)
    obrigatorias = {'ecg_id', 'patient_id', 'scp_codes', 'age', 'strat_fold', 'filename_lr'}
    if not obrigatorias.issubset(df.columns):
        raise ValueError(f'Colunas ausentes: {sorted(obrigatorias - set(df.columns))}')
    if df.ecg_id.isna().any() or df.ecg_id.duplicated().any():
        raise ValueError('ecg_id ausente ou repetido no CSV original.')
    if df.patient_id.isna().any():
        raise ValueError('patient_id ausente; não é possível verificar vazamento.')
    if df.filename_lr.isna().any() or df.filename_lr.duplicated().any():
        raise ValueError('filename_lr ausente ou repetido no CSV original.')

    diagnosticos = scp.loc[pd.to_numeric(scp.diagnostic, errors='coerce').eq(1)]
    mapa = diagnosticos.diagnostic_class.to_dict()
    codigos = df.scp_codes.map(ler_codigos_scp)
    superclasses = codigos.map(lambda d: {mapa[c] for c in d
                                          if c in mapa and mapa[c] in SUPERCLASSES})
    normal = superclasses.map(lambda s: 'NORM' in s)
    alterado = superclasses.map(lambda s: bool(s & ALTERACOES))
    df['motivo_classificacao'] = np.select(
        [normal & ~alterado, ~normal & alterado, normal & alterado],
        ['normal', 'alterado', 'ambiguo_norm_e_alteracao'],
        default='sem_superclasse_diagnostica')
    df['alvo_classificacao'] = pd.Series(
        np.select([normal & ~alterado, ~normal & alterado], [0, 1], default=-1),
        index=df.index).replace(-1, pd.NA).astype('Int64')

    idade = pd.to_numeric(df.age, errors='coerce')
    idade_valida = idade.notna() & np.isfinite(idade) & idade.gt(0) & idade.lt(300)
    df['motivo_regressao'] = np.select(
        [idade_valida, idade.eq(300), idade.isna()],
        ['idade_valida', 'idade_300_anonimizada', 'idade_ausente'],
        default='idade_fora_do_intervalo')
    df['alvo_idade'] = idade.where(idade_valida)
    df['particao'] = df.strat_fold.map(_particao)
    df['elegivel_algum_alvo'] = df.alvo_classificacao.notna() | df.alvo_idade.notna()
    df['linha_sinal'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    df['status_sinal'] = 'nao_elegivel_para_as_tarefas'

    cruzados = df.groupby('patient_id').particao.nunique()
    if cruzados.gt(1).any():
        raise ValueError(f'{int(cruzados.gt(1).sum())} pacientes cruzam treino/validação/teste.')
    return df


def _validar_registro(registro: wfdb.Record) -> tuple[list[str], dict]:
    """Valida geometria, unidades e valores; amplitude extrema é só alerta."""
    sinal = registro.p_signal
    motivos: list[str] = []
    if sinal is None or sinal.shape != FORMA_SINAL:
        motivos.append('dimensoes_invalidas')
    if registro.fs != FREQUENCIA_HZ:
        motivos.append('frequencia_invalida')
    if tuple(registro.sig_name) != DERIVACOES:
        motivos.append('ordem_ou_nomes_das_derivacoes_invalidos')
    if tuple(registro.units) != ('mV',) * 12:
        motivos.append('unidade_diferente_de_mV')

    detalhes = {'nan': 0, 'inf': 0, 'derivacoes_constantes': '',
                'amplitude_min_mv': np.nan, 'amplitude_max_mv': np.nan,
                'amplitude_absoluta_max_mv': np.nan}
    if sinal is not None:
        detalhes['nan'] = int(np.isnan(sinal).sum())
        detalhes['inf'] = int(np.isinf(sinal).sum())
        if detalhes['nan']:
            motivos.append('nan_no_sinal')
        if detalhes['inf']:
            motivos.append('infinito_no_sinal')
        finitos = sinal[np.isfinite(sinal)]
        if finitos.size:
            detalhes['amplitude_min_mv'] = float(finitos.min())
            detalhes['amplitude_max_mv'] = float(finitos.max())
            detalhes['amplitude_absoluta_max_mv'] = float(np.max(np.abs(finitos)))
        if sinal.ndim == 2 and sinal.shape[1] == 12 and np.isfinite(sinal).all():
            constantes = [registro.sig_name[i] for i in np.flatnonzero(np.ptp(sinal, axis=0) == 0)]
            detalhes['derivacoes_constantes'] = ';'.join(constantes)
            if constantes:
                motivos.append('derivacao_constante')
    return motivos, detalhes


def _salvar_json(caminho: Path, conteudo: dict) -> None:
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2), encoding='utf-8')


def preparar_ptbxl(raiz_projeto: Path, pasta_saida: Path,
                  limiar_alerta_mv: float = 5.0, lote_normalizacao: int = 128) -> dict:
    """Inspeciona sinais em fluxo, ajusta escala no treino e persiste uma matriz.

    O limiar de amplitude é apenas alerta de auditoria, nunca critério de
    exclusão ou parâmetro aprendido. O valor é definido antes de ler os sinais.
    """
    if limiar_alerta_mv <= 0 or lote_normalizacao < 1:
        raise ValueError('Limiar e lote devem ser positivos.')
    raiz_projeto = Path(raiz_projeto)
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    raiz_dados = localizar_dataset(raiz_projeto)
    df = construir_metadados(raiz_dados)
    candidatos = df[df.elegivel_algum_alvo]
    if candidatos.empty:
        raise ValueError('Nenhum ECG elegível para as tarefas.')

    temporario = pasta_saida / 'sinais_normalizados.float32.partial'
    destino = pasta_saida / 'sinais_normalizados.float32.bin'
    matriz = np.memmap(temporario, dtype='float32', mode='w+',
                       shape=(len(candidatos), *FORMA_SINAL))
    soma = np.zeros(12, dtype=np.float64)
    soma_quadrados = np.zeros(12, dtype=np.float64)
    n_ajuste = 0
    n_validos = 0
    qualidade = []
    for indice, linha in candidatos.iterrows():
        nome = linha.filename_lr
        info = {'ecg_id': int(linha.ecg_id), 'particao': linha.particao,
                'filename_lr': nome, 'status': 'rejeitado', 'motivos': '',
                'nan': 0, 'inf': 0, 'derivacoes_constantes': '',
                'amplitude_min_mv': np.nan, 'amplitude_max_mv': np.nan,
                'amplitude_absoluta_max_mv': np.nan, 'alerta_amplitude_extrema': False}
        base = raiz_dados / nome if isinstance(nome, str) and nome.startswith('records100/') else None
        if base is None or not base.with_suffix('.hea').is_file() or not base.with_suffix('.dat').is_file():
            motivos = ['arquivo_ausente_ou_caminho_invalido']
        else:
            try:
                registro = wfdb.rdrecord(str(base))
                motivos, detalhes = _validar_registro(registro)
                info.update(detalhes)
                info['alerta_amplitude_extrema'] = bool(
                    info['amplitude_absoluta_max_mv'] >= limiar_alerta_mv)
            except Exception as exc:
                motivos = [f'arquivo_ilegivel:{type(exc).__name__}']

        if motivos:
            info['motivos'] = ';'.join(motivos)
            df.at[indice, 'status_sinal'] = 'rejeitado'
        else:
            sinal = registro.p_signal
            if linha.particao == 'treino':
                # Somente folds 1–8, uma vez por ECG da união das tarefas.
                soma += sinal.sum(axis=0, dtype=np.float64)
                soma_quadrados += np.square(sinal, dtype=np.float64).sum(axis=0)
                n_ajuste += 1
            matriz[n_validos] = sinal.astype(np.float32)
            df.at[indice, 'linha_sinal'] = n_validos
            df.at[indice, 'status_sinal'] = 'aprovado'
            info['status'] = 'aprovado'
            n_validos += 1
        qualidade.append(info)

    matriz.flush()
    del matriz
    if not n_ajuste or not n_validos:
        raise ValueError('Nenhum ECG de treino aprovado para ajustar a normalização.')
    # Trunca apenas a cópia temporária criada pelo processamento.
    os.truncate(temporario, n_validos * np.dtype('float32').itemsize * np.prod(FORMA_SINAL))
    media = soma / (n_ajuste * FORMA_SINAL[0])
    variancia = np.maximum(soma_quadrados / (n_ajuste * FORMA_SINAL[0]) - media ** 2, 0)
    desvio = np.sqrt(variancia)
    if (desvio < 1e-8).any():
        raise ValueError('Desvio padrão quase nulo em uma derivação do treino.')

    matriz = np.memmap(temporario, dtype='float32', mode='r+', shape=(n_validos, *FORMA_SINAL))
    for inicio in range(0, n_validos, lote_normalizacao):
        fim = min(inicio + lote_normalizacao, n_validos)
        lote = (np.asarray(matriz[inicio:fim], dtype=np.float64) - media) / desvio
        if not np.isfinite(lote).all():
            raise ValueError(f'Normalização produziu valor não finito no lote {inicio}:{fim}.')
        matriz[inicio:fim] = lote.astype(np.float32)
    matriz.flush()
    del matriz
    os.replace(temporario, destino)

    qualidade_df = pd.DataFrame(qualidade)
    colunas = ['ecg_id', 'patient_id', 'strat_fold', 'particao', 'filename_lr',
               'linha_sinal', 'alvo_classificacao', 'alvo_idade',
               'motivo_classificacao', 'motivo_regressao', 'status_sinal']
    manifesto = df[colunas].copy()
    classificacao = manifesto[manifesto.alvo_classificacao.notna() &
                              manifesto.linha_sinal.notna()].copy()
    regressao = manifesto[manifesto.alvo_idade.notna() &
                          manifesto.linha_sinal.notna()].copy()
    normalizacao = {
        'metodo': 'z_score_global_por_derivacao',
        'formula': 'z[t,c] = (x[t,c] - media_treino[c]) / desvio_treino[c]',
        'derivacoes': list(DERIVACOES), 'unidade_entrada': 'mV',
        'media_treino': media.tolist(), 'desvio_treino': desvio.tolist(),
        'folds_ajuste': list(range(1, 9)), 'particoes_ajuste': ['treino'],
        'registros_ajuste': n_ajuste, 'amostras_por_derivacao_ajuste': n_ajuste * 1000,
        'populacao_ajuste': 'uniao dos alvos elegiveis com sinal aprovado, somente treino',
        'shape_sinais': [n_validos, *FORMA_SINAL], 'dtype_sinais': 'float32',
        'limiar_alerta_amplitude_mv': limiar_alerta_mv,
        'observacao_alerta': 'Amplitude extrema é sinalizada, não excluída.'}
    resumo = {
        'ecgs_originais': len(df), 'candidatos_a_inspecao': len(candidatos),
        'sinais_aprovados': n_validos, 'sinais_rejeitados': int((qualidade_df.status == 'rejeitado').sum()),
        'alertas_amplitude_extrema': int(qualidade_df.alerta_amplitude_extrema.sum()),
        'classificacao': {'total': len(classificacao),
                          'por_particao': {p: int((classificacao.particao == p).sum())
                                           for p in ('treino', 'validacao', 'teste')},
                          'por_classe': {str(k): int(v) for k, v in
                                         classificacao.alvo_classificacao.value_counts().items()}},
        'regressao': {'total': len(regressao),
                     'por_particao': {p: int((regressao.particao == p).sum())
                                      for p in ('treino', 'validacao', 'teste')},
                     'idade': {k: float(v) for k, v in regressao.alvo_idade.agg(
                         ['min', 'max', 'mean', 'median', 'std']).items()}},
        'ambas': int((manifesto.alvo_classificacao.notna() & manifesto.alvo_idade.notna() &
                      manifesto.linha_sinal.notna()).sum()),
        'exclusoes_alvo_classificacao': {str(k): int(v) for k, v in
                                        df.motivo_classificacao.value_counts().items()},
        'exclusoes_alvo_regressao': {str(k): int(v) for k, v in
                                    df.motivo_regressao.value_counts().items()}}

    validar_preparacao(df, classificacao, regressao, destino, normalizacao)
    manifesto.to_csv(pasta_saida / 'manifesto.csv', index=False)
    classificacao.to_csv(pasta_saida / 'classificacao.csv', index=False)
    regressao.to_csv(pasta_saida / 'regressao.csv', index=False)
    qualidade_df.to_csv(pasta_saida / 'qualidade_sinais.csv', index=False)
    _salvar_json(pasta_saida / 'normalizacao.json', normalizacao)
    _salvar_json(pasta_saida / 'resumo.json', resumo)
    return resumo


def validar_preparacao(df: pd.DataFrame, classificacao: pd.DataFrame,
                       regressao: pd.DataFrame, arquivo_sinais: Path,
                       normalizacao: dict) -> None:
    """Verificações automáticas antes de publicar os artefatos gerados."""
    if df.groupby('patient_id').particao.nunique().gt(1).any():
        raise AssertionError('Paciente compartilhado entre partições.')
    if classificacao.alvo_classificacao.isna().any() or regressao.alvo_idade.isna().any():
        raise AssertionError('Alvo ausente em conjunto elegível.')
    if not set(classificacao.alvo_classificacao.unique()) <= {0, 1}:
        raise AssertionError('Classe fora de {0, 1}.')
    if not ((regressao.alvo_idade > 0) & (regressao.alvo_idade < 300)).all():
        raise AssertionError('Idade inválida na regressão.')
    if normalizacao['particoes_ajuste'] != ['treino'] or normalizacao['folds_ajuste'] != list(range(1, 9)):
        raise AssertionError('Normalização ajustada fora do treino.')
    linhas = df.linha_sinal.dropna().astype(int)
    if len(linhas) != len(set(linhas)) or set(linhas) != set(range(len(linhas))):
        raise AssertionError('Índices dos sinais inconsistentes.')
    n = len(linhas)
    matriz = np.memmap(arquivo_sinais, dtype='float32', mode='r', shape=(n, *FORMA_SINAL))
    if matriz.shape != tuple(normalizacao['shape_sinais']):
        raise AssertionError('Dimensões finais incorretas.')
    for inicio in range(0, n, 128):
        if not np.isfinite(matriz[inicio:inicio + 128]).all():
            raise AssertionError(f'NaN ou infinito nas entradas finais a partir da linha {inicio}.')
    del matriz


def abrir_sinais(pasta_saida: Path, modo: str = 'r') -> np.memmap:
    """Abre a matriz canônica; MLP/RF podem usar reshape, CNN/ResNet transpose."""
    pasta_saida = Path(pasta_saida)
    parametros = json.loads((pasta_saida / 'normalizacao.json').read_text(encoding='utf-8'))
    return np.memmap(pasta_saida / 'sinais_normalizados.float32.bin', dtype='float32',
                     mode=modo, shape=tuple(parametros['shape_sinais']))
