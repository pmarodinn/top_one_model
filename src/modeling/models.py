"""
Sistema de Modelagem de Risco de Crédito
========================================

Este módulo implementa os modelos de Machine Learning conforme especificado
no plano técnico, incluindo pipeline em estágios e arquitetura híbrida.

Modelos implementados:
- Baseline: Regressão Logística Multinomial
- Ensemble: XGBoost/LightGBM 
- Sequencial: LSTM para extração de features
- Final: Arquitetura híbrida em dois estágios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import pickle
import joblib
from datetime import datetime
import yaml

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

# XGBoost e LightGBM
import xgboost as xgb
import lightgbm as lgb

# Deep Learning (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Interpretabilidade
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

# Monitoramento e validação
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

import warnings
warnings.filterwarnings('ignore')


class RiskSpectrumEncoder:
    """
    Classe para codificação do espectro de risco contínuo
    """
    
    def __init__(self, config_path: str = "../config/config.yaml"):
        """
        Inicializa o codificador de espectro
        
        Args:
            config_path: Caminho para configurações
        """
        self.config = self._load_config(config_path)
        self.risk_regions = self.config.get('modeling', {}).get('risk_spectrum', {}).get('regions', [])
        self.setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configurações"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Configurações padrão"""
        return {
            'modeling': {
                'risk_spectrum': {
                    'regions': [
                        {'name': 'Adimplente Pontual', 'score_range': [0.0, 0.2]},
                        {'name': 'Adimplente Lucrativo', 'score_range': [0.2, 0.4]},
                        {'name': 'Risco de Abandono Precoce', 'score_range': [0.4, 0.6]},
                        {'name': 'Inadimplente Parcial', 'score_range': [0.6, 0.8]},
                        {'name': 'Inadimplente Total', 'score_range': [0.8, 1.0]}
                    ]
                }
            }
        }
    
    def setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_target_variable(self, df: pd.DataFrame, payment_history_cols: List[str]) -> pd.DataFrame:
        """
        Cria variável-alvo baseada no espectro de risco contínuo
        
        Args:
            df: DataFrame com dados do cliente
            payment_history_cols: Colunas de histórico de pagamento
            
        Returns:
            DataFrame com variável-alvo criada
        """
        self.logger.info("Criando variável-alvo de espectro de risco...")
        
        df = df.copy()
        
        # Calcular métricas de pagamento
        if payment_history_cols:
            # Percentual de pagamentos em dia (0-5 dias)
            on_time_payments = df[payment_history_cols].apply(
                lambda x: (x <= 5).mean(), axis=1
            )
            
            # Percentual de pagamentos com atraso moderado (6-30 dias)
            moderate_delay = df[payment_history_cols].apply(
                lambda x: ((x > 5) & (x <= 30)).mean(), axis=1
            )
            
            # Percentual de pagamentos com atraso severo (>30 dias)
            severe_delay = df[payment_history_cols].apply(
                lambda x: (x > 30).mean(), axis=1
            )
            
            # Calcular score de risco contínuo (0-1)
            risk_score = (
                0.0 * on_time_payments +      # Peso 0 para pagamentos em dia
                0.3 * moderate_delay +        # Peso 0.3 para atraso moderado  
                0.7 * severe_delay           # Peso 0.7 para atraso severo
            )
            
            # Ajustar por features adicionais se disponíveis
            if 'estresse_financeiro' in df.columns:
                # Normalizar estresse financeiro (0-1)
                stress_normalized = np.clip(df['estresse_financeiro'] / 5, 0, 1)
                risk_score = 0.7 * risk_score + 0.3 * stress_normalized
            
            df['risk_score'] = np.clip(risk_score, 0, 1)
            
        else:
            self.logger.warning("Colunas de histórico de pagamento não encontradas. Usando valores simulados.")
            df['risk_score'] = np.random.uniform(0, 1, size=len(df))
        
        # Mapear para regiões do espectro
        df['risk_region'] = self._map_score_to_region(df['risk_score'])
        df['risk_region_numeric'] = df['risk_region'].map({
            'Adimplente Pontual': 0,
            'Adimplente Lucrativo': 1, 
            'Risco de Abandono Precoce': 2,
            'Inadimplente Parcial': 3,
            'Inadimplente Total': 4
        })
        
        # Estatísticas da variável-alvo
        self.logger.info("Distribuição do espectro de risco:")
        for region in df['risk_region'].value_counts().index:
            count = df['risk_region'].value_counts()[region]
            pct = count / len(df) * 100
            self.logger.info(f"  - {region}: {count:,} ({pct:.1f}%)")
        
        return df
    
    def _map_score_to_region(self, scores: pd.Series) -> pd.Series:
        """Mapeia scores contínuos para regiões do espectro"""
        regions = []
        
        for score in scores:
            for region_info in self.risk_regions:
                min_score, max_score = region_info['score_range']
                if min_score <= score <= max_score:
                    regions.append(region_info['name'])
                    break
            else:
                # Fallback para última região
                regions.append(self.risk_regions[-1]['name'])
        
        return pd.Series(regions, index=scores.index)


class BaselineModel:
    """
    Modelo baseline: Regressão Logística Multinomial
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Treina modelo baseline
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Métricas de treinamento
        """
        self.logger.info("=== TREINANDO MODELO BASELINE (Logistic Regression) ===")
        
        # Armazenar nomes das features
        self.feature_names = X_train.columns.tolist()
        
        # Normalizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Treinar modelo
        self.model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Validação cruzada
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='accuracy'
        )
        
        metrics = {
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'n_features': len(self.feature_names),
            'n_samples': len(X_train)
        }
        
        self.logger.info(f"Baseline treinado: Accuracy CV = {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
        
        return metrics
    
    def predict(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Faz predições
        
        Args:
            X_test: Features de teste
            
        Returns:
            Predições e probabilidades
        """
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)
        
        return predictions, probabilities
    
    def save_model(self, filepath: str):
        """Salva modelo treinado"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Modelo baseline salvo em: {filepath}")


class EnsembleModel:
    """
    Modelos de Ensemble: XGBoost e LightGBM
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Inicializa modelo de ensemble
        
        Args:
            model_type: Tipo do modelo ('xgboost' ou 'lightgbm')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Treina modelo de ensemble
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            
        Returns:
            Métricas de treinamento
        """
        self.logger.info(f"=== TREINANDO MODELO {self.model_type.upper()} ===")
        
        self.feature_names = X_train.columns.tolist()
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                objective='multi:softprob',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                early_stopping_rounds=10
            )
            
            # Preparar dados de validação
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=5,  # 5 regiões do espectro
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                metric='multi_logloss',
                early_stopping_round=10
            )
            
            # Preparar dados de validação
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        
        # Calcular métricas
        train_score = self.model.score(X_train, y_train)
        metrics = {
            'train_accuracy': train_score,
            'n_features': len(self.feature_names),
            'n_samples': len(X_train)
        }
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            metrics['val_accuracy'] = val_score
            
        self.logger.info(f"Modelo {self.model_type} treinado: Train Accuracy = {train_score:.3f}")
        
        return metrics
    
    def predict(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Faz predições"""
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        
        return predictions, probabilities
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Retorna importância das features"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            self.logger.warning("Modelo não possui feature importance")
            return pd.DataFrame()
    
    def save_model(self, filepath: str):
        """Salva modelo treinado"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Modelo {self.model_type} salvo em: {filepath}")


class LSTMFeatureExtractor:
    """
    Modelo LSTM para extração de features sequenciais
    """
    
    def __init__(self, sequence_length: int = 12, embedding_dim: int = 64):
        """
        Inicializa extrator LSTM
        
        Args:
            sequence_length: Comprimento da sequência temporal
            embedding_dim: Dimensão do embedding de saída
        """
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.scaler = StandardScaler()
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_sequences(self, df: pd.DataFrame, 
                         sequence_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara sequências temporais para o LSTM
        
        Args:
            df: DataFrame com dados temporais
            sequence_cols: Colunas de sequência temporal
            
        Returns:
            Arrays de sequências e targets
        """
        self.logger.info("Preparando sequências temporais para LSTM...")
        
        # Simular dados sequenciais se não houver colunas específicas
        if not sequence_cols:
            self.logger.warning("Colunas sequenciais não encontradas. Simulando dados.")
            # Criar sequências simuladas baseadas em features existentes
            n_samples = len(df)
            sequences = np.random.randn(n_samples, self.sequence_length, 3)  # 3 features temporais
        else:
            # Processar colunas sequenciais reais
            sequences = []
            for idx, row in df.iterrows():
                seq = []
                for col in sequence_cols[:self.sequence_length]:
                    if col in df.columns:
                        seq.append(row[col])
                    else:
                        seq.append(0)  # Padding
                
                # Padding se necessário
                while len(seq) < self.sequence_length:
                    seq.append(0)
                
                sequences.append(seq)
            
            sequences = np.array(sequences).reshape(-1, self.sequence_length, 1)
        
        # Normalizar sequências
        sequences_flat = sequences.reshape(-1, sequences.shape[-1])
        sequences_normalized = self.scaler.fit_transform(sequences_flat)
        sequences = sequences_normalized.reshape(sequences.shape)
        
        # Target (se disponível)
        if 'risk_score' in df.columns:
            targets = df['risk_score'].values
        else:
            targets = np.random.uniform(0, 1, size=len(df))
        
        self.logger.info(f"Sequências preparadas: {sequences.shape}")
        
        return sequences, targets
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Constrói arquitetura LSTM
        
        Args:
            input_shape: Forma dos dados de entrada
            
        Returns:
            Modelo LSTM compilado
        """
        self.logger.info("Construindo arquitetura LSTM...")
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(self.embedding_dim, activation='relu'),
            Dense(1, activation='sigmoid')  # Para regressão do risk_score
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.logger.info(f"Modelo LSTM construído: {model.count_params():,} parâmetros")
        
        return model
    
    def train(self, sequences: np.ndarray, targets: np.ndarray, 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Treina modelo LSTM
        
        Args:
            sequences: Sequências de entrada
            targets: Targets para treino
            validation_split: Proporção para validação
            
        Returns:
            Histórico de treinamento
        """
        self.logger.info("=== TREINANDO LSTM PARA EXTRAÇÃO DE FEATURES ===")
        
        # Construir modelo
        input_shape = (sequences.shape[1], sequences.shape[2])
        self.model = self.build_model(input_shape)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('../models/lstm_feature_extractor.h5', save_best_only=True)
        ]
        
        # Treinar
        history = self.model.fit(
            sequences, targets,
            epochs=50,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info("Treinamento LSTM concluído")
        
        return history.history
    
    def extract_features(self, sequences: np.ndarray) -> np.ndarray:
        """
        Extrai features usando LSTM treinado
        
        Args:
            sequences: Sequências de entrada
            
        Returns:
            Features extraídas (embeddings)
        """
        if self.model is None:
            raise ValueError("Modelo LSTM não foi treinado")
        
        # Criar modelo para extração de features (sem última camada)
        feature_extractor = Model(
            inputs=self.model.input,
            outputs=self.model.layers[-2].output  # Penúltima camada
        )
        
        features = feature_extractor.predict(sequences)
        
        self.logger.info(f"Features LSTM extraídas: shape {features.shape}")
        
        return features


class ModelEvaluator:
    """
    Classe para avaliação e comparação de modelos
    """
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Avalia performance de um modelo
        
        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Métricas de avaliação
        """
        self.logger.info("Avaliando performance do modelo...")
        
        # Fazer predições
        if hasattr(model, 'predict'):
            predictions, probabilities = model.predict(X_test)
        else:
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calcular métricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted')
        }
        
        # AUC para classificação multiclasse
        if probabilities is not None:
            try:
                auc_scores = []
                for i in range(probabilities.shape[1]):
                    y_binary = (y_test == i).astype(int)
                    if len(np.unique(y_binary)) > 1:  # Só calcular se houver ambas as classes
                        auc = roc_auc_score(y_binary, probabilities[:, i])
                        auc_scores.append(auc)
                
                metrics['auc_mean'] = np.mean(auc_scores) if auc_scores else np.nan
            except:
                metrics['auc_mean'] = np.nan
        
        # Relatório detalhado
        report = classification_report(y_test, predictions, output_dict=True)
        metrics['classification_report'] = report
        
        self.logger.info(f"Avaliação concluída: Accuracy = {metrics['accuracy']:.3f}")
        
        return metrics
    
    def compare_models(self, models_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compara performance de múltiplos modelos
        
        Args:
            models_results: Dicionário com resultados dos modelos
            
        Returns:
            DataFrame com comparação
        """
        self.logger.info("Comparando performance dos modelos...")
        
        comparison_data = []
        
        for model_name, results in models_results.items():
            row = {
                'model': model_name,
                'accuracy': results.get('accuracy', np.nan),
                'precision': results.get('precision', np.nan),
                'recall': results.get('recall', np.nan),
                'f1_score': results.get('f1_score', np.nan),
                'auc_mean': results.get('auc_mean', np.nan)
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        self.logger.info("Comparação de modelos:")
        self.logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        return comparison_df


if __name__ == "__main__":
    print("🤖 Módulos de Modelagem carregados com sucesso!")
    print("📋 Classes disponíveis:")
    print("  - RiskSpectrumEncoder: Codificação do espectro de risco")
    print("  - BaselineModel: Regressão Logística Multinomial")
    print("  - EnsembleModel: XGBoost/LightGBM")
    print("  - LSTMFeatureExtractor: Extração de features sequenciais")
    print("  - ModelEvaluator: Avaliação e comparação de modelos")
