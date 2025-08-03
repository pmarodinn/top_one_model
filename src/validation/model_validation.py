"""
Sistema de Validação e Teste de Modelos
=======================================

Este módulo implementa estratégias robustas de validação temporal,
backtesting e monitoramento de performance conforme especificado
no plano técnico.

Funcionalidades:
- Validação cruzada temporal (sliding/expanding window)
- Backtesting out-of-time
- Análise de estabilidade em diferentes regimes econômicos
- Monitoramento de drift e performance contínua
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Métricas e validação
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

# Detecção de drift
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.test_suite import TestSuite
    from evidently.tests import TestColumnDrift
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("⚠️ Evidently não disponível. Usando métodos alternativos para detecção de drift.")

# Testes estatísticos
from scipy import stats


class TemporalValidator:
    """
    Classe para validação temporal robusta de modelos
    """
    
    def __init__(self, min_train_size: int = 12, test_size: int = 3):
        """
        Inicializa validador temporal
        
        Args:
            min_train_size: Tamanho mínimo da janela de treino (em meses)
            test_size: Tamanho da janela de teste (em meses)
        """
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_temporal_splits(self, df: pd.DataFrame, 
                              date_column: str) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Cria splits temporais para validação
        
        Args:
            df: DataFrame com dados
            date_column: Nome da coluna de data
            
        Returns:
            Lista de tuplas (train_idx, test_idx)
        """
        self.logger.info("Criando splits temporais para validação...")
        
        # Verificar se coluna de data existe
        if date_column not in df.columns:
            self.logger.warning(f"Coluna {date_column} não encontrada. Usando índice sequencial.")
            return self._create_sequential_splits(df)
        
        # Converter para datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df_sorted = df.sort_values(date_column).reset_index(drop=True)
        
        # Definir datas de início e fim
        start_date = df_sorted[date_column].min()
        end_date = df_sorted[date_column].max()
        
        splits = []
        current_date = start_date + pd.DateOffset(months=self.min_train_size)
        
        while current_date + pd.DateOffset(months=self.test_size) <= end_date:
            # Definir janelas
            train_end = current_date
            test_start = current_date
            test_end = current_date + pd.DateOffset(months=self.test_size)
            
            # Índices de treino e teste
            train_mask = df_sorted[date_column] < train_end
            test_mask = (df_sorted[date_column] >= test_start) & (df_sorted[date_column] < test_end)
            
            train_idx = df_sorted[train_mask].index
            test_idx = df_sorted[test_mask].index
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
                self.logger.info(f"Split criado: Train até {train_end.strftime('%Y-%m')}, "
                               f"Test {test_start.strftime('%Y-%m')} - {test_end.strftime('%Y-%m')}")
            
            # Avançar janela
            current_date += pd.DateOffset(months=1)
        
        self.logger.info(f"Total de {len(splits)} splits temporais criados")
        return splits
    
    def _create_sequential_splits(self, df: pd.DataFrame) -> List[Tuple[pd.Index, pd.Index]]:
        """Cria splits sequenciais quando data não está disponível"""
        self.logger.info("Criando splits sequenciais...")
        
        tscv = TimeSeriesSplit(n_splits=5, test_size=len(df)//10)
        splits = []
        
        for train_idx, test_idx in tscv.split(df):
            splits.append((pd.Index(train_idx), pd.Index(test_idx)))
        
        return splits
    
    def validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                      date_column: str = None) -> Dict[str, Any]:
        """
        Executa validação temporal completa
        
        Args:
            model: Modelo para validação
            X: Features
            y: Target
            date_column: Coluna de data (opcional)
            
        Returns:
            Resultados da validação
        """
        self.logger.info("=== INICIANDO VALIDAÇÃO TEMPORAL ===")
        
        # Criar splits temporais
        splits = self.create_temporal_splits(
            pd.concat([X, y], axis=1), 
            date_column
        ) if date_column else self._create_sequential_splits(X)
        
        # Métricas por split
        split_results = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            self.logger.info(f"Processando split {i+1}/{len(splits)}...")
            
            try:
                # Dividir dados
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Treinar modelo
                model.fit(X_train, y_train)
                
                # Fazer predições
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calcular métricas
                metrics = self._calculate_metrics(y_test, y_pred, y_proba)
                metrics['split'] = i + 1
                metrics['train_size'] = len(train_idx)
                metrics['test_size'] = len(test_idx)
                
                split_results.append(metrics)
                
            except Exception as e:
                self.logger.error(f"Erro no split {i+1}: {str(e)}")
                continue
        
        # Consolidar resultados
        validation_results = self._consolidate_results(split_results)
        
        self.logger.info("=== VALIDAÇÃO TEMPORAL CONCLUÍDA ===")
        return validation_results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                          y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calcula métricas de avaliação"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # AUC para classificação multiclasse
        if y_proba is not None:
            try:
                auc_scores = []
                for class_idx in range(y_proba.shape[1]):
                    y_binary = (y_true == class_idx).astype(int)
                    if len(np.unique(y_binary)) > 1:
                        auc = roc_auc_score(y_binary, y_proba[:, class_idx])
                        auc_scores.append(auc)
                
                metrics['auc_mean'] = np.mean(auc_scores) if auc_scores else np.nan
            except:
                metrics['auc_mean'] = np.nan
        
        return metrics
    
    def _consolidate_results(self, split_results: List[Dict]) -> Dict[str, Any]:
        """Consolida resultados de todos os splits"""
        
        if not split_results:
            return {'error': 'Nenhum split válido processado'}
        
        # Converter para DataFrame
        results_df = pd.DataFrame(split_results)
        
        # Estatísticas agregadas
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        aggregated = {}
        
        for col in numeric_cols:
            if col != 'split':
                aggregated[f'{col}_mean'] = results_df[col].mean()
                aggregated[f'{col}_std'] = results_df[col].std()
                aggregated[f'{col}_min'] = results_df[col].min()
                aggregated[f'{col}_max'] = results_df[col].max()
        
        return {
            'aggregated_metrics': aggregated,
            'split_results': split_results,
            'n_splits': len(split_results),
            'results_df': results_df
        }


class BacktestingEngine:
    """
    Engine para backtesting out-of-time
    """
    
    def __init__(self, holdout_months: int = 6):
        """
        Inicializa engine de backtesting
        
        Args:
            holdout_months: Meses para holdout final
        """
        self.holdout_months = holdout_months
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_holdout_split(self, df: pd.DataFrame, 
                           date_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cria split de holdout temporal
        
        Args:
            df: DataFrame completo
            date_column: Coluna de data
            
        Returns:
            Tupla (train_data, holdout_data)
        """
        self.logger.info(f"Criando holdout de {self.holdout_months} meses...")
        
        if date_column not in df.columns:
            self.logger.warning("Coluna de data não encontrada. Usando split sequencial.")
            split_idx = int(len(df) * 0.8)
            return df.iloc[:split_idx], df.iloc[split_idx:]
        
        # Converter para datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Definir data de corte
        max_date = df[date_column].max()
        cutoff_date = max_date - pd.DateOffset(months=self.holdout_months)
        
        # Dividir dados
        train_data = df[df[date_column] <= cutoff_date].copy()
        holdout_data = df[df[date_column] > cutoff_date].copy()
        
        self.logger.info(f"Split criado:")
        self.logger.info(f"  - Dados de treino: {len(train_data):,} registros até {cutoff_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"  - Dados de holdout: {len(holdout_data):,} registros após {cutoff_date.strftime('%Y-%m-%d')}")
        
        return train_data, holdout_data
    
    def run_backtest(self, model, train_data: pd.DataFrame, 
                    holdout_data: pd.DataFrame, feature_cols: List[str],
                    target_col: str) -> Dict[str, Any]:
        """
        Executa backtesting completo
        
        Args:
            model: Modelo para teste
            train_data: Dados de treino
            holdout_data: Dados de holdout
            feature_cols: Colunas de features
            target_col: Coluna de target
            
        Returns:
            Resultados do backtest
        """
        self.logger.info("=== EXECUTANDO BACKTESTING ===")
        
        # Preparar dados
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_holdout = holdout_data[feature_cols]
        y_holdout = holdout_data[target_col]
        
        # Treinar modelo
        self.logger.info("Treinando modelo com dados históricos...")
        model.fit(X_train, y_train)
        
        # Fazer predições no holdout
        self.logger.info("Fazendo predições no período de holdout...")
        y_pred = model.predict(X_holdout)
        y_proba = model.predict_proba(X_holdout) if hasattr(model, 'predict_proba') else None
        
        # Calcular métricas
        backtest_metrics = self._calculate_backtest_metrics(y_holdout, y_pred, y_proba)
        
        # Análise temporal das predições
        temporal_analysis = self._analyze_temporal_performance(
            holdout_data, y_holdout, y_pred, y_proba
        )
        
        results = {
            'backtest_metrics': backtest_metrics,
            'temporal_analysis': temporal_analysis,
            'predictions': {
                'y_true': y_holdout.values,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
        }
        
        self.logger.info("=== BACKTESTING CONCLUÍDO ===")
        return results
    
    def _calculate_backtest_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                   y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calcula métricas específicas do backtest"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def _analyze_temporal_performance(self, holdout_data: pd.DataFrame,
                                    y_true: pd.Series, y_pred: np.ndarray,
                                    y_proba: np.ndarray = None) -> Dict[str, Any]:
        """Analisa performance ao longo do tempo no holdout"""
        
        # Criar DataFrame com resultados
        results_df = holdout_data.copy()
        results_df['y_true'] = y_true.values
        results_df['y_pred'] = y_pred
        
        # Análise por mês (se data disponível)
        temporal_metrics = {}
        
        date_cols = [col for col in results_df.columns if 'data' in col.lower() or 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            results_df[date_col] = pd.to_datetime(results_df[date_col], errors='coerce')
            results_df['month'] = results_df[date_col].dt.to_period('M')
            
            # Métricas por mês
            monthly_metrics = []
            for month in results_df['month'].unique():
                if pd.isna(month):
                    continue
                    
                month_data = results_df[results_df['month'] == month]
                if len(month_data) > 0:
                    monthly_acc = accuracy_score(month_data['y_true'], month_data['y_pred'])
                    monthly_metrics.append({
                        'month': str(month),
                        'accuracy': monthly_acc,
                        'n_samples': len(month_data)
                    })
            
            temporal_metrics['monthly_performance'] = monthly_metrics
        
        return temporal_metrics


class DriftDetector:
    """
    Detector de concept drift e data drift
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Inicializa detector de drift
        
        Args:
            significance_level: Nível de significância para testes
        """
        self.significance_level = significance_level
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_data_drift(self, reference_data: pd.DataFrame,
                         current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecta drift nos dados de entrada
        
        Args:
            reference_data: Dados de referência (treino)
            current_data: Dados atuais (produção)
            
        Returns:
            Resultados da detecção de drift
        """
        self.logger.info("Detectando data drift...")
        
        drift_results = {}
        
        # Verificar colunas comuns
        common_cols = list(set(reference_data.columns) & set(current_data.columns))
        numeric_cols = reference_data[common_cols].select_dtypes(include=[np.number]).columns
        
        # Testes estatísticos para cada coluna
        for col in numeric_cols:
            try:
                ref_values = reference_data[col].dropna()
                curr_values = current_data[col].dropna()
                
                if len(ref_values) > 0 and len(curr_values) > 0:
                    # Teste Kolmogorov-Smirnov
                    ks_stat, ks_pvalue = stats.ks_2samp(ref_values, curr_values)
                    
                    # Teste Mann-Whitney U
                    mw_stat, mw_pvalue = stats.mannwhitneyu(ref_values, curr_values, alternative='two-sided')
                    
                    # Population Stability Index (PSI)
                    psi_value = self._calculate_psi(ref_values, curr_values)
                    
                    drift_results[col] = {
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pvalue,
                        'ks_drift_detected': ks_pvalue < self.significance_level,
                        'mw_statistic': mw_stat,
                        'mw_pvalue': mw_pvalue,
                        'mw_drift_detected': mw_pvalue < self.significance_level,
                        'psi': psi_value,
                        'psi_drift_detected': psi_value > 0.2  # Threshold comum para PSI
                    }
                    
            except Exception as e:
                self.logger.error(f"Erro ao analisar drift em {col}: {str(e)}")
                continue
        
        # Resumo geral
        total_vars = len(drift_results)
        drift_vars_ks = sum(1 for r in drift_results.values() if r['ks_drift_detected'])
        drift_vars_psi = sum(1 for r in drift_results.values() if r['psi_drift_detected'])
        
        summary = {
            'total_variables_analyzed': total_vars,
            'drift_detected_ks': drift_vars_ks,
            'drift_detected_psi': drift_vars_psi,
            'drift_percentage_ks': (drift_vars_ks / total_vars * 100) if total_vars > 0 else 0,
            'drift_percentage_psi': (drift_vars_psi / total_vars * 100) if total_vars > 0 else 0
        }
        
        self.logger.info(f"Data drift detectado em {drift_vars_ks}/{total_vars} variáveis (KS test)")
        self.logger.info(f"Data drift detectado em {drift_vars_psi}/{total_vars} variáveis (PSI)")
        
        return {
            'summary': summary,
            'detailed_results': drift_results
        }
    
    def _calculate_psi(self, expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """
        Calcula Population Stability Index (PSI)
        
        Args:
            expected: Distribuição esperada (referência)
            actual: Distribuição atual
            bins: Número de bins para discretização
            
        Returns:
            Valor do PSI
        """
        try:
            # Criar bins baseados na distribuição esperada
            _, bin_edges = np.histogram(expected, bins=bins)
            
            # Calcular distribuições
            expected_counts, _ = np.histogram(expected, bins=bin_edges)
            actual_counts, _ = np.histogram(actual, bins=bin_edges)
            
            # Converter para proporções
            expected_props = expected_counts / len(expected)
            actual_props = actual_counts / len(actual)
            
            # Evitar divisão por zero
            expected_props = np.where(expected_props == 0, 0.0001, expected_props)
            actual_props = np.where(actual_props == 0, 0.0001, actual_props)
            
            # Calcular PSI
            psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
            
            return psi
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular PSI: {str(e)}")
            return np.nan
    
    def detect_target_drift(self, reference_target: pd.Series,
                          current_target: pd.Series) -> Dict[str, Any]:
        """
        Detecta drift na variável target
        
        Args:
            reference_target: Target de referência
            current_target: Target atual
            
        Returns:
            Resultados da detecção de target drift
        """
        self.logger.info("Detectando target drift...")
        
        # Distribuições das classes
        ref_dist = reference_target.value_counts(normalize=True).sort_index()
        curr_dist = current_target.value_counts(normalize=True).sort_index()
        
        # Teste chi-quadrado para mudança na distribuição
        try:
            # Alinhar distribuições
            all_classes = list(set(ref_dist.index) | set(curr_dist.index))
            ref_aligned = [ref_dist.get(cls, 0) for cls in all_classes]
            curr_aligned = [curr_dist.get(cls, 0) for cls in all_classes]
            
            # Converter para contagens
            ref_counts = np.array(ref_aligned) * len(reference_target)
            curr_counts = np.array(curr_aligned) * len(current_target)
            
            # Teste chi-quadrado
            chi2_stat, chi2_pvalue = stats.chisquare(curr_counts, ref_counts)
            
            target_drift_results = {
                'chi2_statistic': chi2_stat,
                'chi2_pvalue': chi2_pvalue,
                'drift_detected': chi2_pvalue < self.significance_level,
                'reference_distribution': ref_dist.to_dict(),
                'current_distribution': curr_dist.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao detectar target drift: {str(e)}")
            target_drift_results = {'error': str(e)}
        
        return target_drift_results


if __name__ == "__main__":
    print("✅ Módulos de Validação carregados com sucesso!")
    print("📋 Classes disponíveis:")
    print("  - TemporalValidator: Validação cruzada temporal")
    print("  - BacktestingEngine: Backtesting out-of-time")
    print("  - DriftDetector: Detecção de concept/data drift")
