"""Autoencoder 1D e Random Forest sobre os dados preparados do PTB-XL.

Os sinais são lidos por memória mapeada; alvos aparecem apenas na etapa
supervisionada dos Random Forests. Nenhuma função acessa sinais do fold 10.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.deep_learning import selecionar_particoes, verificar_artefatos


ARQUIVOS_PREPARADOS = ('normalizacao.json', 'manifesto.csv', 'classificacao.csv',
                       'regressao.csv', 'sinais_normalizados.float32.bin')
PARTICOES_AJUSTE = ('treino', 'validacao')


def preparar_pastas(pasta_saida: Path) -> dict[str, Path]:
    nomes = ('checkpoints', 'embeddings', 'metricas', 'graficos', 'configuracoes', 'modelos')
    pastas = {nome: Path(pasta_saida) / nome for nome in nomes}
    for pasta in pastas.values():
        pasta.mkdir(parents=True, exist_ok=True)
    return pastas


def conferir_arquivos_preparados(pasta: Path) -> dict[str, int]:
    """Confere presença/tamanho antes de copiar ou abrir os artefatos."""
    pasta = Path(pasta)
    tamanhos = {}
    for nome in ARQUIVOS_PREPARADOS:
        caminho = pasta / nome
        if not caminho.is_file():
            raise FileNotFoundError(f'Arquivo preparado ausente: {caminho}')
        tamanho = caminho.stat().st_size
        if tamanho <= 0:
            raise ValueError(f'Arquivo preparado vazio: {caminho}')
        tamanhos[nome] = tamanho
    parametros = json.loads((pasta / 'normalizacao.json').read_text(encoding='utf-8'))
    esperado = int(np.prod(parametros['shape_sinais'])) * np.dtype('float32').itemsize
    if tamanhos['sinais_normalizados.float32.bin'] != esperado:
        raise ValueError(f'Tamanho do binário incorreto: {tamanhos["sinais_normalizados.float32.bin"]} != {esperado}')
    return tamanhos


def conferir_correspondencia(pasta: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Reusa a verificação da etapa 3 e confirma índices/alvos contra o manifesto."""
    conferir_arquivos_preparados(pasta)
    parametros, tabelas = verificar_artefatos(pasta)
    manifesto = tabelas['manifesto'].set_index('ecg_id')
    for tarefa in ('classificacao', 'regressao'):
        tabela = tabelas[tarefa]
        referencia = manifesto.loc[tabela.ecg_id]
        colunas = ['patient_id', 'strat_fold', 'particao', 'linha_sinal',
                   'alvo_classificacao' if tarefa == 'classificacao' else 'alvo_idade']
        for coluna in colunas:
            if coluna == 'particao':
                iguais = np.array_equal(tabela[coluna].to_numpy(), referencia[coluna].to_numpy())
            else:
                a = pd.to_numeric(tabela[coluna]).to_numpy(dtype=np.float64)
                b = pd.to_numeric(referencia[coluna]).to_numpy(dtype=np.float64)
                iguais = np.array_equal(a, b, equal_nan=True)
            if not iguais:
                raise ValueError(f'{tarefa}: coluna {coluna} não corresponde ao manifesto.')
    return parametros, tabelas


def selecionar_amostras(tabelas: dict[str, pd.DataFrame], modo_teste: bool,
                       semente: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Seleciona a união sem rótulos para o AE e os mesmos exemplos para os RFs.

    No teste rápido, a amostra do AE vem somente do manifesto, sem consultar
    alvos. As amostras supervisionadas são extraídas dessa mesma união.
    """
    manifesto = tabelas['manifesto']
    unicos = manifesto.loc[manifesto.status_sinal.eq('aprovado') &
                            manifesto.particao.isin(PARTICOES_AJUSTE),
                            ['ecg_id', 'patient_id', 'strat_fold', 'particao', 'linha_sinal']].copy()
    unicos['linha_sinal'] = unicos.linha_sinal.astype(np.int64)
    if unicos.ecg_id.duplicated().any() or unicos.linha_sinal.duplicated().any():
        raise ValueError('ECGs duplicados na união do autoencoder.')
    if modo_teste:
        unicos = pd.concat([
            parte.sample(n=min(n, len(parte)), random_state=semente)
            for particao, n in (('treino', 128), ('validacao', 64))
            for parte in [unicos.loc[unicos.particao.eq(particao)]]
        ], ignore_index=True)
    unicos = unicos.sort_values('linha_sinal').reset_index(drop=True)
    selecionados = {}
    ids = set(unicos.ecg_id)
    for tarefa in ('classificacao', 'regressao'):
        tabela = tabelas[tarefa].loc[tabelas[tarefa].ecg_id.isin(ids)].copy()
        treino, validacao = selecionar_particoes(tabela, tarefa, modo_teste, semente)
        selecionados[tarefa] = pd.concat([treino, validacao], ignore_index=True)
        if set(selecionados[tarefa].particao) != set(PARTICOES_AJUSTE):
            raise ValueError(f'Amostra insuficiente em {tarefa}.')
    if set(unicos.particao) != set(PARTICOES_AJUSTE):
        raise ValueError('A união do autoencoder precisa de treino e validação.')
    return unicos, selecionados


class SinaisSemRotulos(Dataset):
    """Retorna apenas sinal, ecg_id e linha_sinal; nunca carrega idade/SCP."""

    def __init__(self, pasta_dados: Path, tabela: pd.DataFrame,
                 shape: tuple[int, int, int]):
        if not set(tabela.particao) <= set(PARTICOES_AJUSTE):
            raise ValueError('Fold 10 não pode entrar no autoencoder.')
        self.arquivo = Path(pasta_dados) / 'sinais_normalizados.float32.bin'
        self.shape = shape
        self.linhas = tabela.linha_sinal.to_numpy(dtype=np.int64)
        self.ecg_ids = tabela.ecg_id.to_numpy(dtype=np.int64)
        self._matriz = None

    def __len__(self) -> int:
        return len(self.linhas)

    def __getstate__(self) -> dict:
        estado = self.__dict__.copy()
        estado['_matriz'] = None
        return estado

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        if self._matriz is None:
            self._matriz = np.memmap(self.arquivo, dtype='float32', mode='r', shape=self.shape)
        # Copia somente um ECG; PyTorch recebe canais antes do tempo.
        x = np.ascontiguousarray(self._matriz[self.linhas[i]].T)
        return torch.from_numpy(x), int(self.ecg_ids[i]), int(self.linhas[i])


class AutoencoderECG1D(nn.Module):
    def __init__(self, dimensao_latente: int = 64):
        super().__init__()
        if not 1 <= dimensao_latente < 12000:
            raise ValueError('Dimensão latente deve estar entre 1 e 11.999.')
        self.dimensao_latente = dimensao_latente
        self.encoder_convolucional = nn.Sequential(
            nn.Conv1d(12, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.projecao_latente = nn.Linear(128 * 125, dimensao_latente)
        self.expansao_latente = nn.Linear(dimensao_latente, 128 * 125)
        self.decoder_convolucional = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, 12, 4, stride=2, padding=1))

    def codificar(self, x: torch.Tensor) -> torch.Tensor:
        return self.projecao_latente(self.encoder_convolucional(x).flatten(1))

    def decodificar(self, z: torch.Tensor) -> torch.Tensor:
        x = self.expansao_latente(z).reshape(-1, 128, 125)
        return self.decoder_convolucional(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decodificar(self.codificar(x))


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


@dataclass(frozen=True)
class ConfigAE:
    dimensao_latente: int
    epocas: int
    tamanho_lote: int
    taxa_aprendizado: float
    decaimento_pesos: float
    paciencia: int
    delta_minimo: float
    semente: int
    trabalhadores: int
    modo_teste: bool


def _salvar_json(caminho: Path, objeto: dict) -> None:
    temporario = caminho.with_suffix(caminho.suffix + '.tmp')
    temporario.write_text(json.dumps(objeto, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporario, caminho)


def _salvar_torch(caminho: Path, objeto: dict) -> None:
    temporario = caminho.with_suffix(caminho.suffix + '.tmp')
    torch.save(objeto, temporario)
    os.replace(temporario, caminho)


def _salvar_csv(caminho: Path, tabela: pd.DataFrame) -> None:
    temporario = caminho.with_suffix(caminho.suffix + '.tmp')
    tabela.to_csv(temporario, index=False)
    os.replace(temporario, caminho)


def _epoca_ae(modelo: AutoencoderECG1D, loader: DataLoader,
              criterio: nn.Module, dispositivo: torch.device,
              otimizador: torch.optim.Optimizer | None = None) -> float:
    treino = otimizador is not None
    modelo.train(treino)
    soma = 0.0
    with torch.set_grad_enabled(treino):
        for x, _, _ in loader:  # IDs servem só para auditoria; nenhum rótulo é fornecido.
            x = x.to(dispositivo, non_blocking=True)
            if treino:
                otimizador.zero_grad(set_to_none=True)
            reconstruido = modelo(x)
            if reconstruido.shape != x.shape:
                raise ValueError(f'Reconstrução {reconstruido.shape} difere da entrada {x.shape}.')
            perda = criterio(reconstruido, x)
            if not torch.isfinite(perda):
                raise FloatingPointError('Perda de reconstrução não finita.')
            if treino:
                perda.backward()
                otimizador.step()
            soma += float(perda.detach().item()) * len(x)
    return soma / len(loader.dataset)


def _grafico_perdas(historico: list[dict], caminho: Path) -> None:
    tabela = pd.DataFrame(historico)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tabela.epoca, tabela.perda_treino, label='Treino')
    ax.plot(tabela.epoca, tabela.perda_validacao, label='Validação')
    ax.set(title='Perda de reconstrução do autoencoder', xlabel='Época',
           ylabel='Smooth L1 (sinal normalizado)')
    ax.legend()
    fig.tight_layout()
    temporario = caminho.with_suffix('.png.tmp')
    fig.savefig(temporario, format='png', dpi=140)
    plt.close(fig)
    os.replace(temporario, caminho)


def treinar_autoencoder(pasta_dados: Path, pastas: dict[str, Path],
                       unicos: pd.DataFrame, shape: tuple[int, int, int],
                       config: ConfigAE, dispositivo: torch.device) -> dict:
    """Treina sem rótulos nos ECGs únicos de treino; valida no fold 9."""
    if not set(unicos.particao) <= set(PARTICOES_AJUSTE):
        raise ValueError('Fold 10 encontrado na entrada do autoencoder.')
    if unicos.ecg_id.duplicated().any() or unicos.linha_sinal.duplicated().any():
        raise ValueError('ECG duplicado na entrada do autoencoder.')
    treino = unicos.loc[unicos.particao.eq('treino')]
    validacao = unicos.loc[unicos.particao.eq('validacao')]
    if treino.empty or validacao.empty:
        raise ValueError('Treino ou validação do autoencoder vazio.')
    config_dict = asdict(config)
    caminho_config = pastas['configuracoes'] / 'autoencoder.json'
    caminho_ultimo = pastas['checkpoints'] / 'autoencoder_ultimo.pt'
    caminho_melhor = pastas['checkpoints'] / 'autoencoder_melhor.pt'
    caminho_historico = pastas['metricas'] / 'historico_autoencoder.csv'
    caminho_grafico = pastas['graficos'] / 'perda_reconstrucao.png'
    if not caminho_ultimo.exists():
        torch.manual_seed(config.semente)
        if dispositivo.type == 'cuda':
            torch.cuda.manual_seed_all(config.semente)
    modelo = AutoencoderECG1D(config.dimensao_latente).to(dispositivo)
    n_parametros = contar_parametros(modelo)
    criterio = nn.SmoothL1Loss(beta=1.0)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=config.taxa_aprendizado,
                                   weight_decay=config.decaimento_pesos)
    gerador = torch.Generator().manual_seed(config.semente)
    kwargs = {'batch_size': config.tamanho_lote, 'num_workers': config.trabalhadores,
              'pin_memory': dispositivo.type == 'cuda'}
    loader_treino = DataLoader(SinaisSemRotulos(pasta_dados, treino, shape),
                                shuffle=True, generator=gerador, **kwargs)
    loader_validacao = DataLoader(SinaisSemRotulos(pasta_dados, validacao, shape),
                                   shuffle=False, **kwargs)
    identificacao = {'hiperparametros': config_dict, 'parametros_treinaveis': n_parametros,
                    'perda': 'SmoothL1Loss(beta=1.0)', 'entrada': [12, 1000],
                    'dimensao_original': 12000, 'dimensao_latente': config.dimensao_latente,
                    'ecgs_treino': len(treino), 'ecgs_validacao': len(validacao),
                    'sem_rotulos': True, 'particoes_usadas': ['treino', 'validacao']}
    if caminho_config.exists():
        anterior = json.loads(caminho_config.read_text(encoding='utf-8'))
        if anterior != identificacao:
            raise ValueError('Configuração ou amostra do autoencoder mudou; use outra pasta de saída.')
    else:
        _salvar_json(caminho_config, identificacao)

    historico: list[dict] = []
    melhor_perda = float('inf')
    sem_melhora = 0
    epoca_inicial = 1
    tempo_total = 0.0
    finalizado = False
    if caminho_ultimo.exists():
        estado = torch.load(caminho_ultimo, map_location=dispositivo, weights_only=False)
        modelo.load_state_dict(estado['modelo'])
        otimizador.load_state_dict(estado['otimizador'])
        historico = estado['historico']
        melhor_perda = estado['melhor_perda']
        sem_melhora = estado['sem_melhora']
        epoca_inicial = estado['epoca'] + 1
        tempo_total = estado['tempo_total_segundos']
        finalizado = estado['finalizado']
        torch.set_rng_state(estado['rng_cpu'])
        if dispositivo.type == 'cuda' and estado.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(estado['rng_cuda'])
        gerador.set_state(estado['rng_dataloader'])
    if finalizado or epoca_inicial > config.epocas:
        if not caminho_melhor.is_file():
            raise FileNotFoundError('Checkpoint do melhor autoencoder não encontrado.')
        return {'epocas_executadas': len(historico), 'melhor_perda_validacao': melhor_perda,
                'tempo_total_segundos': tempo_total, 'parametros_treinaveis': n_parametros,
                'retomado_sem_treinar': True}

    for epoca in range(epoca_inicial, config.epocas + 1):
        inicio = time.perf_counter()
        perda_treino = _epoca_ae(modelo, loader_treino, criterio, dispositivo, otimizador)
        perda_val = _epoca_ae(modelo, loader_validacao, criterio, dispositivo)
        segundos = time.perf_counter() - inicio
        tempo_total += segundos
        if perda_val < melhor_perda - config.delta_minimo:
            melhor_perda = perda_val
            sem_melhora = 0
            _salvar_torch(caminho_melhor, {'modelo': modelo.state_dict(),
                          'epoca': epoca, 'perda_validacao': perda_val,
                          'dimensao_latente': config.dimensao_latente})
        else:
            sem_melhora += 1
        historico.append({'epoca': epoca, 'perda_treino': perda_treino,
                          'perda_validacao': perda_val, 'segundos_epoca': segundos,
                          'segundos_acumulados': tempo_total})
        finalizado = epoca == config.epocas or sem_melhora >= config.paciencia
        _salvar_torch(caminho_ultimo, {'modelo': modelo.state_dict(),
                      'otimizador': otimizador.state_dict(), 'historico': historico,
                      'epoca': epoca, 'melhor_perda': melhor_perda,
                      'sem_melhora': sem_melhora, 'tempo_total_segundos': tempo_total,
                      'finalizado': finalizado, 'rng_cpu': torch.get_rng_state(),
                      'rng_cuda': torch.cuda.get_rng_state_all() if dispositivo.type == 'cuda' else None,
                      'rng_dataloader': gerador.get_state()})
        _salvar_csv(caminho_historico, pd.DataFrame(historico))
        _grafico_perdas(historico, caminho_grafico)
        print(f'AE época {epoca}/{config.epocas}: treino {perda_treino:.5f}, '
              f'validação {perda_val:.5f}, {segundos:.1f} s', flush=True)
        if sem_melhora >= config.paciencia:
            print(f'AE: early stopping após {epoca} épocas.', flush=True)
            break
    del loader_treino, loader_validacao, modelo, otimizador
    if dispositivo.type == 'cuda':
        torch.cuda.empty_cache()
    return {'epocas_executadas': len(historico), 'melhor_perda_validacao': melhor_perda,
            'tempo_total_segundos': tempo_total, 'parametros_treinaveis': n_parametros,
            'retomado_sem_treinar': False}


def carregar_melhor_autoencoder(pasta_checkpoints: Path, dimensao_latente: int,
                               dispositivo: torch.device) -> AutoencoderECG1D:
    caminho = Path(pasta_checkpoints) / 'autoencoder_melhor.pt'
    estado = torch.load(caminho, map_location=dispositivo, weights_only=False)
    if estado['dimensao_latente'] != dimensao_latente:
        raise ValueError('Dimensão latente não corresponde ao checkpoint.')
    modelo = AutoencoderECG1D(dimensao_latente).to(dispositivo)
    modelo.load_state_dict(estado['modelo'])
    modelo.eval()
    for parametro in modelo.parameters():
        parametro.requires_grad_(False)
    return modelo


def _sha256_arquivo(caminho: Path) -> str:
    hash_obj = hashlib.sha256()
    with caminho.open('rb') as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b''):
            hash_obj.update(bloco)
    return hash_obj.hexdigest()


def extrair_embeddings(pasta_dados: Path, pastas: dict[str, Path],
                       unicos: pd.DataFrame, shape: tuple[int, int, int],
                       dimensao_latente: int, dispositivo: torch.device,
                       tamanho_lote: int = 128, trabalhadores: int = 0
                       ) -> tuple[Path, pd.DataFrame]:
    """Extrai Z de treino/validação com o melhor encoder congelado.

    A ordem de `indice_embeddings.csv` é idêntica à das linhas do `.npy`.
    Uma extração existente só é reutilizada se checkpoint e índice coincidirem.
    """
    if unicos.ecg_id.duplicated().any() or not set(unicos.particao) <= set(PARTICOES_AJUSTE):
        raise ValueError('Índice de extração com duplicatas ou fold 10.')
    ordem = unicos.sort_values('linha_sinal').reset_index(drop=True)
    caminho_checkpoint = pastas['checkpoints'] / 'autoencoder_melhor.pt'
    hash_checkpoint = _sha256_arquivo(caminho_checkpoint)
    pasta = pastas['embeddings']
    caminho_matriz = pasta / 'representacoes.npy'
    caminho_indice = pasta / 'indice_embeddings.csv'
    caminho_meta = pasta / 'metadados_embeddings.json'
    esperado = ordem[['ecg_id', 'patient_id', 'strat_fold', 'particao', 'linha_sinal']].copy()
    esperado['linha_embedding'] = np.arange(len(esperado), dtype=np.int64)
    metadados = {'checkpoint_sha256': hash_checkpoint, 'shape': [len(ordem), dimensao_latente],
                 'dimensao_original': 12000, 'dimensao_latente': dimensao_latente,
                 'particoes': ['treino', 'validacao'], 'sem_rotulos_na_entrada': True}
    if caminho_matriz.is_file() and caminho_indice.is_file() and caminho_meta.is_file():
        anterior = json.loads(caminho_meta.read_text(encoding='utf-8'))
        indice_existente = pd.read_csv(caminho_indice)
        if anterior == metadados and indice_existente.equals(esperado):
            matriz_existente = np.load(caminho_matriz, mmap_mode='r')
            if matriz_existente.shape == tuple(metadados['shape']) and np.isfinite(matriz_existente).all():
                print('Embeddings existentes correspondem ao melhor checkpoint; extração reutilizada.')
                return caminho_matriz, indice_existente
        raise ValueError('Embeddings existentes não correspondem ao checkpoint/índice atual; use outra pasta de saída.')

    modelo = carregar_melhor_autoencoder(pastas['checkpoints'], dimensao_latente, dispositivo)
    loader = DataLoader(SinaisSemRotulos(pasta_dados, ordem, shape), batch_size=tamanho_lote,
                        shuffle=False, num_workers=trabalhadores,
                        pin_memory=dispositivo.type == 'cuda')
    temporario = pasta / 'representacoes.partial.npy'
    matriz = np.lib.format.open_memmap(temporario, mode='w+', dtype='float32',
                                        shape=(len(ordem), dimensao_latente))
    cursor = 0
    with torch.inference_mode():
        for x, ids, linhas in loader:
            fim = cursor + len(x)
            if not np.array_equal(ids.numpy(), ordem.ecg_id.iloc[cursor:fim].to_numpy()) or \
               not np.array_equal(linhas.numpy(), ordem.linha_sinal.iloc[cursor:fim].to_numpy()):
                raise AssertionError('Ordem dos embeddings divergiu do manifesto.')
            z = modelo.codificar(x.to(dispositivo, non_blocking=True)).cpu().numpy()
            if z.shape != (len(x), dimensao_latente) or not np.isfinite(z).all():
                raise ValueError('Embedding com dimensão incorreta ou valor não finito.')
            matriz[cursor:fim] = z.astype(np.float32)
            cursor = fim
    if cursor != len(ordem):
        raise AssertionError('Extração incompleta de embeddings.')
    matriz.flush()
    del matriz, loader, modelo
    if dispositivo.type == 'cuda':
        torch.cuda.empty_cache()
    os.replace(temporario, caminho_matriz)
    _salvar_csv(caminho_indice, esperado)
    _salvar_json(caminho_meta, metadados)
    return caminho_matriz, esperado


def grafico_reconstrucoes(pasta_dados: Path, pastas: dict[str, Path],
                         unicos: pd.DataFrame, shape: tuple[int, int, int],
                         dimensao_latente: int, dispositivo: torch.device,
                         semente: int) -> list[int]:
    """Compara ECGs reais e reconstruídos, usando somente exames de treino."""
    treino = unicos.loc[unicos.particao.eq('treino')]
    exemplos = treino.sample(n=min(2, len(treino)), random_state=semente)
    modelo = carregar_melhor_autoencoder(pastas['checkpoints'], dimensao_latente, dispositivo)
    matriz = np.memmap(Path(pasta_dados) / 'sinais_normalizados.float32.bin',
                       dtype='float32', mode='r', shape=shape)
    derivacoes = (1, 7, 10)  # II, V2, V5; seleção de visualização, sem interpretação clínica.
    nomes = ('II', 'V2', 'V5')
    fig, axes = plt.subplots(len(exemplos), len(derivacoes),
                             figsize=(14, 3.2 * len(exemplos)), squeeze=False)
    ids = []
    with torch.inference_mode():
        for linha_figura, linha in enumerate(exemplos.itertuples()):
            original = np.array(matriz[int(linha.linha_sinal)].T, copy=True)
            x = torch.from_numpy(original[None]).to(dispositivo)
            reconstruido = modelo(x).cpu().numpy()[0]
            ids.append(int(linha.ecg_id))
            tempo = np.arange(1000) / 100
            for j, (canal, nome) in enumerate(zip(derivacoes, nomes)):
                ax = axes[linha_figura, j]
                ax.plot(tempo, original[canal], lw=0.8, label='Sinal real')
                ax.plot(tempo, reconstruido[canal], lw=0.8, label='Reconstrução')
                ax.set(title=f'ECG {int(linha.ecg_id)} — {nome}', xlabel='Tempo (s)',
                       ylabel='Amplitude normalizada')
                if linha_figura == 0 and j == 0:
                    ax.legend()
    fig.suptitle('ECGs de treino: sinais reais e reconstruídos')
    fig.tight_layout()
    caminho = pastas['graficos'] / 'exemplos_reconstrucao.png'
    temporario = caminho.with_suffix('.png.tmp')
    fig.savefig(temporario, format='png', dpi=140)
    plt.close(fig)
    os.replace(temporario, caminho)
    del matriz, modelo
    if dispositivo.type == 'cuda':
        torch.cuda.empty_cache()
    return ids


@dataclass(frozen=True)
class ConfigRF:
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    max_features: str
    n_jobs: int
    random_state: int
    bootstrap: bool
    modo_teste: bool


def _assinatura_amostra(tabela: pd.DataFrame, alvo: str) -> str:
    colunas = ['ecg_id', 'linha_sinal', 'particao', alvo]
    valores = pd.util.hash_pandas_object(tabela[colunas], index=False).to_numpy()
    return hashlib.sha256(valores.tobytes()).hexdigest()


def _matriz_rf_original(pasta_dados: Path, shape: tuple[int, int, int],
                       linhas: np.ndarray) -> np.ndarray:
    matriz = np.memmap(Path(pasta_dados) / 'sinais_normalizados.float32.bin',
                       dtype='float32', mode='r', shape=shape)
    # Indexação avançada cria apenas a matriz da tarefa/partição em uso.
    resultado = matriz.reshape(shape[0], 12000)[linhas].astype(np.float32, copy=False)
    del matriz
    return resultado


def _matriz_rf_embeddings(caminho_embeddings: Path, indice_embeddings: pd.DataFrame,
                          tabela: pd.DataFrame) -> np.ndarray:
    mapa = indice_embeddings.set_index('ecg_id')
    if not tabela.ecg_id.isin(mapa.index).all():
        raise ValueError('ECG do Random Forest sem embedding correspondente.')
    ref = mapa.loc[tabela.ecg_id]
    if not np.array_equal(ref.linha_sinal.to_numpy(), tabela.linha_sinal.to_numpy()) or \
       not np.array_equal(ref.particao.to_numpy(), tabela.particao.to_numpy()):
        raise ValueError('Embedding associado ao ECG ou à partição errada.')
    matriz = np.load(caminho_embeddings, mmap_mode='r')
    resultado = np.asarray(matriz[ref.linha_embedding.to_numpy(dtype=np.int64)], dtype=np.float32)
    del matriz
    return resultado


def _matriz_finita_em_lotes(matriz: np.ndarray, tamanho_lote: int = 128) -> bool:
    return all(np.isfinite(matriz[i:i + tamanho_lote]).all()
               for i in range(0, len(matriz), tamanho_lote))


def _grafico_confusao(matriz: np.ndarray, caminho: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    imagem = ax.imshow(matriz, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(matriz[i, j])), ha='center', va='center')
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['Normal', 'Alterado'], yticklabels=['Normal', 'Alterado'],
           xlabel='Previsto', ylabel='Real', title='Matriz de confusão — validação')
    fig.colorbar(imagem, ax=ax, fraction=0.046)
    fig.tight_layout()
    temporario = caminho.with_suffix('.png.tmp')
    fig.savefig(temporario, format='png', dpi=140)
    plt.close(fig)
    os.replace(temporario, caminho)


def treinar_random_forest(pasta_dados: Path, pastas: dict[str, Path],
                         tarefa: str, representacao: str, tabela: pd.DataFrame,
                         shape: tuple[int, int, int], config: ConfigRF,
                         caminho_embeddings: Path, indice_embeddings: pd.DataFrame
                         ) -> dict:
    """Treina um RF na CPU; as duas representações recebem a mesma tabela/config."""
    if tarefa not in ('classificacao', 'regressao') or representacao not in ('original', 'embedding'):
        raise ValueError('Tarefa ou representação inválida.')
    if not set(tabela.particao) <= set(PARTICOES_AJUSTE):
        raise ValueError('Fold 10 não pode entrar no Random Forest.')
    alvo = 'alvo_classificacao' if tarefa == 'classificacao' else 'alvo_idade'
    if tabela[alvo].isna().any() or tabela.ecg_id.duplicated().any():
        raise ValueError('Alvos ausentes ou ECGs duplicados no Random Forest.')
    treino = tabela.loc[tabela.particao.eq('treino')]
    validacao = tabela.loc[tabela.particao.eq('validacao')]
    if treino.empty or validacao.empty:
        raise ValueError('Treino ou validação do Random Forest vazio.')
    assinatura = _assinatura_amostra(tabela, alvo)
    prefixo = f'{tarefa}_{representacao}'
    caminho_modelo = pastas['modelos'] / f'random_forest_{prefixo}.joblib'
    caminho_metricas = pastas['metricas'] / f'random_forest_{prefixo}.json'
    caminho_config = pastas['configuracoes'] / f'random_forest_{prefixo}.json'
    config_atual = {'tarefa': tarefa, 'representacao': representacao,
                    'hiperparametros': asdict(config), 'assinatura_amostra': assinatura,
                    'ecgs_treino': len(treino), 'ecgs_validacao': len(validacao),
                    'particoes_usadas': ['treino', 'validacao']}
    if caminho_config.exists():
        anterior = json.loads(caminho_config.read_text(encoding='utf-8'))
        if anterior != config_atual:
            raise ValueError(f'Configuração ou amostra mudou em {prefixo}; use outra pasta de saída.')
    else:
        _salvar_json(caminho_config, config_atual)
    if caminho_modelo.is_file() and caminho_metricas.is_file():
        resultado = json.loads(caminho_metricas.read_text(encoding='utf-8'))
        print(f'RF {prefixo}: modelo e métricas existentes reutilizados.')
        return resultado

    x_treino = x_val = modelo = None
    try:
        if representacao == 'original':
            x_treino = _matriz_rf_original(pasta_dados, shape,
                                          treino.linha_sinal.to_numpy(dtype=np.int64))
            x_val = _matriz_rf_original(pasta_dados, shape,
                                       validacao.linha_sinal.to_numpy(dtype=np.int64))
        else:
            x_treino = _matriz_rf_embeddings(caminho_embeddings, indice_embeddings, treino)
            x_val = _matriz_rf_embeddings(caminho_embeddings, indice_embeddings, validacao)
        if x_treino.shape[0] != len(treino) or x_val.shape[0] != len(validacao):
            raise AssertionError('Número de exemplos divergiu entre representações.')
        if x_treino.shape[1] != (12000 if representacao == 'original' else
                                 np.load(caminho_embeddings, mmap_mode='r').shape[1]):
            raise AssertionError('Dimensão da representação incorreta.')
        if not _matriz_finita_em_lotes(x_treino) or not _matriz_finita_em_lotes(x_val):
            raise ValueError('Entrada do Random Forest contém NaN ou infinito.')
        y_treino = treino[alvo].to_numpy()
        y_val = validacao[alvo].to_numpy()
        argumentos = {k: v for k, v in asdict(config).items() if k != 'modo_teste'}
        modelo = (RandomForestClassifier(**argumentos) if tarefa == 'classificacao'
                  else RandomForestRegressor(**argumentos))
        inicio = time.perf_counter()
        modelo.fit(x_treino, y_treino)
        segundos_treino = time.perf_counter() - inicio
        inicio = time.perf_counter()
        predito = modelo.predict(x_val)
        segundos_inferencia = time.perf_counter() - inicio
        if not np.isfinite(predito).all():
            raise ValueError('Predição não finita do Random Forest.')
        metricas = {'tarefa': tarefa, 'representacao': representacao,
                    'dimensao_entrada': int(x_treino.shape[1]),
                    'ecgs_treino': len(treino), 'ecgs_validacao': len(validacao),
                    'assinatura_amostra': assinatura,
                    'segundos_treinamento': segundos_treino,
                    'segundos_inferencia_validacao': segundos_inferencia}
        if tarefa == 'classificacao':
            metricas.update({
                'acuracia': float(accuracy_score(y_val, predito)),
                'precisao': float(precision_score(y_val, predito, pos_label=1, zero_division=0)),
                'recall': float(recall_score(y_val, predito, pos_label=1, zero_division=0)),
                'precisao_macro': float(precision_score(y_val, predito, average='macro', zero_division=0)),
                'recall_macro': float(recall_score(y_val, predito, average='macro', zero_division=0)),
                'f1_macro': float(f1_score(y_val, predito, average='macro', zero_division=0))})
            matriz_confusao = confusion_matrix(y_val, predito, labels=[0, 1])
            _salvar_csv(pastas['metricas'] / f'matriz_confusao_{representacao}.csv',
                        pd.DataFrame({'classe_real': ['Normal', 'Alterado'],
                                      'previsto_normal': matriz_confusao[:, 0],
                                      'previsto_alterado': matriz_confusao[:, 1]}))
            _grafico_confusao(matriz_confusao,
                              pastas['graficos'] / f'matriz_confusao_{representacao}.png')
        else:
            mse = float(mean_squared_error(y_val, predito))
            metricas.update({'mae': float(mean_absolute_error(y_val, predito)),
                             'mse': mse, 'rmse': float(np.sqrt(mse)),
                             'r2': float(r2_score(y_val, predito))})
        temporario = caminho_modelo.with_suffix('.joblib.tmp')
        joblib.dump(modelo, temporario, compress=0)
        os.replace(temporario, caminho_modelo)
        _salvar_json(caminho_metricas, metricas)
        print(f'RF {prefixo}: {len(treino)} treino, {len(validacao)} validação, '
              f'{x_treino.shape[1]} atributos, {segundos_treino:.1f} s de treino.', flush=True)
        return metricas
    except MemoryError as exc:
        raise RuntimeError(
            'RAM insuficiente para Random Forest. Reduza n_estimators, max_depth ou n_jobs '
            'para AMBAS as representações e use uma nova pasta de rodada; preserve todos os ECGs/atributos.'
        ) from exc
    finally:
        del x_treino, x_val, modelo
        gc.collect()


def salvar_comparacoes(pastas: dict[str, Path], resultados: list[dict]) -> None:
    tabela = pd.DataFrame(resultados)
    for tarefa in ('classificacao', 'regressao'):
        parte = tabela.loc[tabela.tarefa.eq(tarefa)].copy()
        if len(parte) != 2 or set(parte.representacao) != {'original', 'embedding'}:
            raise ValueError(f'Comparação incompleta em {tarefa}.')
        if parte.assinatura_amostra.nunique() != 1 or parte.ecgs_treino.nunique() != 1 or \
           parte.ecgs_validacao.nunique() != 1:
            raise AssertionError('Representações não usaram exatamente os mesmos exemplos.')
        _salvar_csv(pastas['metricas'] / f'comparacao_{tarefa}.csv', parte)
