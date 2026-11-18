"""Redes e treinamento supervisionado sobre os artefatos do pré-processamento.

Este módulo não relê nem transforma o PTB-XL original. A matriz normalizada
é aberta por memória mapeada; os CSVs da etapa 2 definem alvos e partições.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


TAREFAS = ('classificacao', 'regressao')
MODELOS = ('mlp', 'cnn1d', 'resnet1d')
PARTICOES = ('treino', 'validacao', 'teste')


def verificar_artefatos(pasta_dados: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Verifica matriz, alvos, folds e isolamento de pacientes antes do treino."""
    pasta_dados = Path(pasta_dados)
    parametros = json.loads((pasta_dados / 'normalizacao.json').read_text(encoding='utf-8'))
    shape = tuple(parametros['shape_sinais'])
    if len(shape) != 3 or shape[1:] != (1000, 12) or parametros['dtype_sinais'] != 'float32':
        raise ValueError('Formato dos sinais preparados diferente de (N, 1000, 12) float32.')
    if parametros['folds_ajuste'] != list(range(1, 9)) or parametros['particoes_ajuste'] != ['treino']:
        raise ValueError('Normalização não foi ajustada apenas no treino.')
    binario = pasta_dados / 'sinais_normalizados.float32.bin'
    if binario.stat().st_size != int(np.prod(shape)) * np.dtype('float32').itemsize:
        raise ValueError('Tamanho do arquivo de sinais não corresponde à forma declarada.')

    tabelas = {nome: pd.read_csv(pasta_dados / f'{nome}.csv')
              for nome in ('manifesto', 'classificacao', 'regressao')}
    manifesto = tabelas['manifesto']
    if manifesto.ecg_id.duplicated().any() or manifesto.patient_id.isna().any():
        raise ValueError('IDs de ECG duplicados ou pacientes ausentes no manifesto.')
    esperado = np.where(manifesto.strat_fold.between(1, 8), 'treino',
               np.where(manifesto.strat_fold.eq(9), 'validacao',
               np.where(manifesto.strat_fold.eq(10), 'teste', 'fold_invalido')))
    if not np.array_equal(manifesto.particao.to_numpy(), esperado):
        raise ValueError('Partições não correspondem aos folds oficiais.')
    if manifesto.groupby('patient_id').particao.nunique().gt(1).any():
        raise ValueError('Há paciente compartilhado entre as partições.')

    for tarefa in TAREFAS:
        df = tabelas[tarefa]
        alvo = 'alvo_classificacao' if tarefa == 'classificacao' else 'alvo_idade'
        if df.empty or df.ecg_id.duplicated().any() or df[alvo].isna().any():
            raise ValueError(f'IDs ou alvos inválidos em {tarefa}.')
        if set(df.particao) != set(PARTICOES):
            raise ValueError(f'Partições incompletas em {tarefa}.')
        if df.linha_sinal.isna().any() or not df.linha_sinal.between(0, shape[0] - 1).all():
            raise ValueError(f'Índices de sinais inválidos em {tarefa}.')
        if not df.ecg_id.isin(manifesto.ecg_id).all():
            raise ValueError(f'ECG fora do manifesto em {tarefa}.')
        if tarefa == 'classificacao' and not set(df[alvo].unique()) <= {0, 1}:
            raise ValueError('Alvo de classificação fora de {0, 1}.')
        if tarefa == 'regressao' and not df[alvo].between(0, 300, inclusive='neither').all():
            raise ValueError('Idades inválidas no conjunto de regressão.')
        cruzados = df.groupby('patient_id').particao.nunique()
        if cruzados.gt(1).any():
            raise ValueError(f'Paciente compartilhado em {tarefa}.')
    return parametros, tabelas


def selecionar_particoes(df: pd.DataFrame, tarefa: str, modo_teste: bool,
                        semente: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna apenas treino e validação; teste nunca é entregue ao treinador."""
    if tarefa not in TAREFAS:
        raise ValueError(f'Tarefa inválida: {tarefa}')
    treino = df.loc[df.particao.eq('treino')].copy()
    validacao = df.loc[df.particao.eq('validacao')].copy()
    if modo_teste:
        if tarefa == 'classificacao':
            treino = pd.concat([g.sample(n=min(16, len(g)), random_state=semente)
                                for _, g in treino.groupby('alvo_classificacao')])
            validacao = pd.concat([g.sample(n=min(8, len(g)), random_state=semente)
                                   for _, g in validacao.groupby('alvo_classificacao')])
        else:
            treino = treino.sample(n=min(32, len(treino)), random_state=semente)
            validacao = validacao.sample(n=min(16, len(validacao)), random_state=semente)
    return treino.reset_index(drop=True), validacao.reset_index(drop=True)


class ECGDataset(Dataset):
    """Lê somente o ECG solicitado da matriz normalizada da etapa 2."""

    def __init__(self, pasta_dados: Path, tabela: pd.DataFrame, tarefa: str,
                 arquitetura: str, shape: tuple[int, int, int]):
        if tarefa not in TAREFAS or arquitetura not in MODELOS:
            raise ValueError('Tarefa ou arquitetura inválida.')
        if not set(tabela.particao) <= {'treino', 'validacao'}:
            raise ValueError('O Dataset de treinamento não pode conter o teste.')
        self.arquivo = Path(pasta_dados) / 'sinais_normalizados.float32.bin'
        self.shape = shape
        self.indices = tabela.linha_sinal.to_numpy(dtype=np.int64)
        coluna = 'alvo_classificacao' if tarefa == 'classificacao' else 'alvo_idade'
        self.alvos = tabela[coluna].to_numpy(dtype=np.float32)
        self.arquitetura = arquitetura
        self._matriz = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self) -> dict:
        estado = self.__dict__.copy()
        estado['_matriz'] = None  # cada worker reabre seu próprio mapeamento
        return estado

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._matriz is None:
            self._matriz = np.memmap(self.arquivo, dtype='float32', mode='r', shape=self.shape)
        sinal = np.array(self._matriz[self.indices[indice]], copy=True)
        if self.arquitetura == 'mlp':
            sinal = sinal.reshape(12000)
        else:
            sinal = np.ascontiguousarray(sinal.T)
        return torch.from_numpy(sinal), torch.tensor(self.alvos[indice], dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self, dropout: float = 0.30):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Flatten(), nn.Linear(12000, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rede(x).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, dropout: float = 0.20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.cabeca = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeca(self.features(x)).squeeze(-1)


class BlocoResidual1D(nn.Module):
    def __init__(self, entrada: int, saida: int, stride: int = 1):
        super().__init__()
        self.caminho = nn.Sequential(
            nn.Conv1d(entrada, saida, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(saida), nn.ReLU(),
            nn.Conv1d(saida, saida, 3, padding=1, bias=False),
            nn.BatchNorm1d(saida))
        self.atalho = (nn.Identity() if entrada == saida and stride == 1 else
                       nn.Sequential(nn.Conv1d(entrada, saida, 1, stride=stride, bias=False),
                                     nn.BatchNorm1d(saida)))
        self.ativacao = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ativacao(self.caminho(x) + self.atalho(x))


class ResNet1D(nn.Module):
    def __init__(self, dropout: float = 0.20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU())
        self.blocos = nn.Sequential(
            BlocoResidual1D(32, 32), BlocoResidual1D(32, 32),
            BlocoResidual1D(32, 64, stride=2), BlocoResidual1D(64, 64),
            BlocoResidual1D(64, 128, stride=2), BlocoResidual1D(128, 128))
        self.final = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                   nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.blocos(self.stem(x))).squeeze(-1)


def criar_modelo(nome: str, dropout: float) -> nn.Module:
    modelos = {'mlp': MLP, 'cnn1d': CNN1D, 'resnet1d': ResNet1D}
    if nome not in modelos:
        raise ValueError(f'Modelo desconhecido: {nome}')
    return modelos[nome](dropout=dropout)


def parametros_treinaveis(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


@dataclass(frozen=True)
class ConfigTreino:
    tarefa: str
    arquitetura: str
    epocas: int
    tamanho_lote: int
    taxa_aprendizado: float
    decaimento_pesos: float
    dropout: float
    paciencia: int
    delta_minimo: float
    semente: int
    trabalhadores: int
    modo_teste: bool


def _metricas_validacao(tarefa: str, alvos: np.ndarray,
                       saidas: np.ndarray) -> dict[str, float]:
    if tarefa == 'regressao':
        erro = saidas - alvos
        mse = float(np.mean(erro ** 2))
        denominador = float(np.sum((alvos - alvos.mean()) ** 2))
        return {'mae': float(np.mean(np.abs(erro))), 'rmse': float(np.sqrt(mse)),
                'r2': float(1 - np.sum(erro ** 2) / denominador) if denominador else float('nan')}
    preditos = (saidas >= 0).astype(int)  # logit >= 0 equivale a probabilidade >= 0,5
    reais = alvos.astype(int)
    precisao = []
    recall = []
    f1 = []
    for classe in (0, 1):
        tp = int(((preditos == classe) & (reais == classe)).sum())
        fp = int(((preditos == classe) & (reais != classe)).sum())
        fn = int(((preditos != classe) & (reais == classe)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precisao.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {'acuracia': float((preditos == reais).mean()),
            'precisao_macro': float(np.mean(precisao)),
            'recall_macro': float(np.mean(recall)),
            'f1_macro': float(np.mean(f1))}


def _executar_epoca(modelo: nn.Module, carregador: DataLoader,
                   criterio: nn.Module, dispositivo: torch.device,
                   tarefa: str, otimizador: torch.optim.Optimizer | None = None
                   ) -> tuple[float, dict[str, float]]:
    treino = otimizador is not None
    modelo.train(treino)
    perda_total = 0.0
    alvos = []
    saidas = []
    with torch.set_grad_enabled(treino):
        for x, y in carregador:
            x = x.to(dispositivo, non_blocking=True)
            y = y.to(dispositivo, non_blocking=True)
            if treino:
                otimizador.zero_grad(set_to_none=True)
            pred = modelo(x)
            perda = criterio(pred, y)
            if not torch.isfinite(perda):
                raise FloatingPointError('Perda não finita durante o treinamento.')
            if treino:
                perda.backward()
                otimizador.step()
            perda_total += float(perda.detach().item()) * len(y)
            if not treino:
                alvos.append(y.detach().cpu().numpy())
                saidas.append(pred.detach().cpu().numpy())
    perda_media = perda_total / len(carregador.dataset)
    metricas = (_metricas_validacao(tarefa, np.concatenate(alvos), np.concatenate(saidas))
                if not treino else {})
    return perda_media, metricas


def _salvar_torch_atomico(objeto: dict, caminho: Path) -> None:
    temporario = caminho.with_suffix(caminho.suffix + '.tmp')
    torch.save(objeto, temporario)
    os.replace(temporario, caminho)


def _salvar_historico(historico: list[dict], caminho: Path) -> None:
    temporario = caminho.with_suffix('.csv.tmp')
    pd.DataFrame(historico).to_csv(temporario, index=False)
    os.replace(temporario, caminho)


def _salvar_curvas(historico: list[dict], tarefa: str, caminho: Path) -> None:
    df = pd.DataFrame(historico)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(df.epoca, df.perda_treino, label='Treino')
    axes[0].plot(df.epoca, df.perda_validacao, label='Validação')
    axes[0].set(title='Curvas de perda', xlabel='Época', ylabel='Perda')
    axes[0].legend()
    metrica = 'f1_macro' if tarefa == 'classificacao' else 'mae'
    axes[1].plot(df.epoca, df[metrica], color='#9b4e65')
    axes[1].set(title=f'Validação: {metrica}', xlabel='Época',
                ylabel='F1 macro' if tarefa == 'classificacao' else 'MAE (anos)')
    fig.tight_layout()
    temporario = caminho.with_suffix('.png.tmp')
    fig.savefig(temporario, format='png', dpi=130)
    plt.close(fig)
    os.replace(temporario, caminho)


def treinar_experimento(pasta_dados: Path, pasta_saida: Path,
                       tabela: pd.DataFrame, parametros: dict,
                       configuracao: ConfigTreino, dispositivo: torch.device
                       ) -> dict:
    """Treina uma combinação tarefa/modelo com retomada por checkpoint.

    Recebe somente linhas de treino e validação. Nunca abre a partição teste.
    """
    c = configuracao
    if c.tarefa not in TAREFAS or c.arquitetura not in MODELOS:
        raise ValueError('Configuração de tarefa ou arquitetura inválida.')
    if c.epocas < 1 or c.tamanho_lote < 1 or c.paciencia < 1:
        raise ValueError('Épocas, lote e paciência devem ser positivos.')
    if not set(tabela.particao) <= {'treino', 'validacao'}:
        raise ValueError('A tabela entregue ao treinador contém a partição teste.')
    pasta_saida = Path(pasta_saida) / c.tarefa / c.arquitetura
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_config = pasta_saida / 'configuracao.json'
    caminho_ultimo = pasta_saida / 'ultimo.pt'
    caminho_melhor = pasta_saida / 'melhor.pt'
    caminho_historico = pasta_saida / 'historico.csv'
    caminho_curvas = pasta_saida / 'curvas.png'

    config_dict = asdict(c)
    if caminho_config.exists():
        anterior = json.loads(caminho_config.read_text(encoding='utf-8'))
        if anterior['hiperparametros'] != config_dict:
            raise ValueError(f'Configuração mudou em {pasta_saida}; use outra pasta para nova rodada.')
    else:
        torch.manual_seed(c.semente)
        np.random.seed(c.semente)
    treino = tabela.loc[tabela.particao.eq('treino')]
    validacao = tabela.loc[tabela.particao.eq('validacao')]
    shape = tuple(parametros['shape_sinais'])
    dados_treino = ECGDataset(pasta_dados, treino, c.tarefa, c.arquitetura, shape)
    dados_val = ECGDataset(pasta_dados, validacao, c.tarefa, c.arquitetura, shape)
    gerador = torch.Generator().manual_seed(c.semente)
    argumentos = {'batch_size': c.tamanho_lote, 'num_workers': c.trabalhadores,
                  'pin_memory': dispositivo.type == 'cuda'}
    loader_treino = DataLoader(dados_treino, shuffle=True, generator=gerador, **argumentos)
    loader_val = DataLoader(dados_val, shuffle=False, **argumentos)
    modelo = criar_modelo(c.arquitetura, c.dropout).to(dispositivo)
    n_parametros = parametros_treinaveis(modelo)
    criterio = nn.BCEWithLogitsLoss() if c.tarefa == 'classificacao' else nn.MSELoss()
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=c.taxa_aprendizado,
                                   weight_decay=c.decaimento_pesos)
    if not caminho_config.exists():
        caminho_config.write_text(json.dumps({
            'hiperparametros': config_dict, 'parametros_treinaveis': n_parametros,
            'dispositivo_inicial': str(dispositivo), 'criterio': type(criterio).__name__,
            'forma_entrada': [12000] if c.arquitetura == 'mlp' else [12, 1000],
            'amostras_treino': len(treino), 'amostras_validacao': len(validacao),
            'particoes_usadas': ['treino', 'validacao'],
            'normalizacao': parametros['metodo'],
        }, ensure_ascii=False, indent=2), encoding='utf-8')

    historico: list[dict] = []
    melhor_perda = float('inf')
    epoca_inicial = 1
    sem_melhora = 0
    tempo_acumulado = 0.0
    finalizado = False
    if caminho_ultimo.exists():
        ckpt = torch.load(caminho_ultimo, map_location=dispositivo, weights_only=False)
        modelo.load_state_dict(ckpt['modelo'])
        otimizador.load_state_dict(ckpt['otimizador'])
        historico = ckpt['historico']
        melhor_perda = ckpt['melhor_perda']
        sem_melhora = ckpt['sem_melhora']
        tempo_acumulado = ckpt['tempo_acumulado_segundos']
        epoca_inicial = ckpt['epoca'] + 1
        finalizado = ckpt['finalizado']
        torch.set_rng_state(ckpt['rng_torch_cpu'])
        if dispositivo.type == 'cuda' and ckpt.get('rng_torch_cuda') is not None:
            torch.cuda.set_rng_state_all(ckpt['rng_torch_cuda'])
        gerador.set_state(ckpt['rng_dataloader'])
    if finalizado or epoca_inicial > c.epocas:
        return {'tarefa': c.tarefa, 'arquitetura': c.arquitetura,
                'epocas_executadas': len(historico), 'melhor_perda_validacao': melhor_perda,
                'parametros_treinaveis': n_parametros, 'tempo_segundos': tempo_acumulado,
                'retomado_sem_treinar': True}

    for epoca in range(epoca_inicial, c.epocas + 1):
        inicio = time.perf_counter()
        perda_treino, _ = _executar_epoca(modelo, loader_treino, criterio, dispositivo,
                                          c.tarefa, otimizador)
        perda_val, metricas = _executar_epoca(modelo, loader_val, criterio, dispositivo, c.tarefa)
        duracao = time.perf_counter() - inicio
        tempo_acumulado += duracao
        melhorou = perda_val < melhor_perda - c.delta_minimo
        if melhorou:
            melhor_perda = perda_val
            sem_melhora = 0
            _salvar_torch_atomico({'modelo': modelo.state_dict(), 'epoca': epoca,
                                   'perda_validacao': perda_val, 'metricas_validacao': metricas,
                                   'configuracao': config_dict}, caminho_melhor)
        else:
            sem_melhora += 1
        linha = {'epoca': epoca, 'perda_treino': perda_treino,
                 'perda_validacao': perda_val, **metricas,
                 'segundos_epoca': duracao, 'segundos_acumulados': tempo_acumulado}
        historico.append(linha)
        finalizado = epoca == c.epocas or sem_melhora >= c.paciencia
        _salvar_torch_atomico({
            'modelo': modelo.state_dict(), 'otimizador': otimizador.state_dict(),
            'epoca': epoca, 'historico': historico, 'melhor_perda': melhor_perda,
            'sem_melhora': sem_melhora, 'tempo_acumulado_segundos': tempo_acumulado,
            'finalizado': finalizado, 'rng_torch_cpu': torch.get_rng_state(),
            'rng_torch_cuda': torch.cuda.get_rng_state_all() if dispositivo.type == 'cuda' else None,
            'rng_dataloader': gerador.get_state(),
        }, caminho_ultimo)
        _salvar_historico(historico, caminho_historico)
        _salvar_curvas(historico, c.tarefa, caminho_curvas)
        print(f'{c.tarefa}/{c.arquitetura} | época {epoca}/{c.epocas} | '
              f'treino {perda_treino:.4f} | validação {perda_val:.4f} | {duracao:.1f} s',
              flush=True)
        if sem_melhora >= c.paciencia:
            print(f'Early stopping após {epoca} épocas.', flush=True)
            break
    return {'tarefa': c.tarefa, 'arquitetura': c.arquitetura,
            'epocas_executadas': len(historico), 'melhor_perda_validacao': melhor_perda,
            'parametros_treinaveis': n_parametros, 'tempo_segundos': tempo_acumulado,
            'retomado_sem_treinar': False}
