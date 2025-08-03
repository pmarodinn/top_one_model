"""
Módulo de Engenharia de Dados e Features
=======================================

Este módulo implementa a integração entre dados internos e externos,
criação de features avançadas e preparação dos dados para modelagem.

Funcionalidades:
- Integração de dados internos com dados regionais
- Criação de features sintéticas (ex: Estresse Financeiro)
- Normalização e limpeza de dados geográficos
- Pipeline de feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import yaml
import re
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')


class DataIntegrator:
    """
    Classe responsável pela integração de dados internos e externos
    """
    
    def __init__(self, config_path: str = "../config/config.yaml"):
        """
        Inicializa o integrador de dados
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.geolocator = Nominatim(user_agent="top_one_model")
        
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configurações do arquivo YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logging.warning(f"Config file not found: {config_path}")
            return {}
    
    def setup_logging(self):
        """Configurar sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def normalize_city_names(self, df: pd.DataFrame, city_column: str) -> pd.DataFrame:
        """
        Normaliza nomes de cidades para facilitar merge
        
        Args:
            df: DataFrame com coluna de cidades
            city_column: Nome da coluna de cidades
            
        Returns:
            DataFrame com nomes de cidades normalizados
        """
        self.logger.info(f"Normalizando nomes de cidades na coluna: {city_column}")
        
        df = df.copy()
        
        if city_column not in df.columns:
            self.logger.error(f"Coluna {city_column} não encontrada no DataFrame")
            return df
        
        # Converter para string e remover espaços
        df[city_column] = df[city_column].astype(str).str.strip()
        
        # Converter para título (primeira letra maiúscula)
        df[city_column] = df[city_column].str.title()
        
        # Remover acentos e caracteres especiais
        df[f'{city_column}_normalized'] = df[city_column].str.normalize('NFKD')\
                                                         .str.encode('ascii', errors='ignore')\
                                                         .str.decode('ascii')
        
        # Correções específicas de cidades conhecidas
        city_corrections = {
            'Sao Paulo': 'São Paulo',
            'Rio De Janeiro': 'Rio de Janeiro', 
            'Belo Horizonte': 'Belo Horizonte',
            'Porto Alegre': 'Porto Alegre'
        }
        
        for wrong, correct in city_corrections.items():
            df[city_column] = df[city_column].replace(wrong, correct)
        
        self.logger.info(f"Normalização concluída: {df[city_column].nunique()} cidades únicas")
        
        return df
    
    def merge_internal_external_data(self, 
                                   df_internal: pd.DataFrame,
                                   df_combustiveis: pd.DataFrame,
                                   df_utilities: pd.DataFrame,
                                   df_indicadores: pd.DataFrame,
                                   city_column: str) -> pd.DataFrame:
        """
        Integra dados internos com dados externos coletados
        
        Args:
            df_internal: Dataset interno
            df_combustiveis: Dados de combustíveis
            df_utilities: Dados de utilities
            df_indicadores: Indicadores econômicos
            city_column: Nome da coluna de cidade no dataset interno
            
        Returns:
            DataFrame integrado
        """
        self.logger.info("=== INICIANDO INTEGRAÇÃO DE DADOS ===")
        
        # Normalizar nomes de cidades em todos os DataFrames
        df_internal = self.normalize_city_names(df_internal, city_column)
        df_combustiveis = self.normalize_city_names(df_combustiveis, 'municipio')
        df_utilities = self.normalize_city_names(df_utilities, 'municipio')
        df_indicadores = self.normalize_city_names(df_indicadores, 'municipio')
        
        # Começar com o dataset interno
        df_merged = df_internal.copy()
        initial_rows = len(df_merged)
        
        # Merge com dados de combustíveis
        self.logger.info("Integrando dados de combustíveis...")
        df_merged = df_merged.merge(
            df_combustiveis.rename(columns={'municipio': city_column}),
            on=city_column,
            how='left',
            suffixes=('', '_combustivel')
        )
        
        # Merge com dados de utilities
        self.logger.info("Integrando dados de utilities...")
        df_merged = df_merged.merge(
            df_utilities.rename(columns={'municipio': city_column}),
            on=city_column,
            how='left',
            suffixes=('', '_utility')
        )
        
        # Merge com indicadores econômicos
        self.logger.info("Integrando indicadores econômicos...")
        df_merged = df_merged.merge(
            df_indicadores.rename(columns={'municipio': city_column}),
            on=city_column,
            how='left',
            suffixes=('', '_indicador')
        )
        
        final_rows = len(df_merged)
        
        # Estatísticas do merge
        self.logger.info(f"Integração concluída:")
        self.logger.info(f"  - Registros iniciais: {initial_rows:,}")
        self.logger.info(f"  - Registros finais: {final_rows:,}")
        
        # Verificar cobertura dos dados externos
        self._check_external_data_coverage(df_merged)
        
        return df_merged
    
    def _check_external_data_coverage(self, df_merged: pd.DataFrame):
        """Verifica cobertura dos dados externos após merge"""
        
        external_cols = [
            'gasolina_comum', 'energia_kwh', 'pib_per_capita'
        ]
        
        for col in external_cols:
            if col in df_merged.columns:
                coverage = (1 - df_merged[col].isna().mean()) * 100
                self.logger.info(f"  - Cobertura {col}: {coverage:.1f}%")


class FeatureEngineer:
    """
    Classe responsável pela criação de features avançadas
    """
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar sistema de logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_financial_stress_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria a feature sintética de 'Estresse Financeiro'
        
        EstresseFinanceiro = DívidaTotal / (RendaMensal - CustoVidaRegional)
        
        Args:
            df: DataFrame com dados integrados
            
        Returns:
            DataFrame com feature de estresse financeiro
        """
        self.logger.info("Criando feature de Estresse Financeiro...")
        
        df = df.copy()
        
        # Identificar colunas relevantes (ajustar conforme dataset real)
        possible_debt_cols = ['divida_total', 'valor_devido', 'debt_amount']
        possible_income_cols = ['renda_mensal', 'salario', 'income']
        
        debt_col = None
        income_col = None
        
        for col in possible_debt_cols:
            if col in df.columns:
                debt_col = col
                break
                
        for col in possible_income_cols:
            if col in df.columns:
                income_col = col
                break
        
        if debt_col and income_col:
            # Calcular custo de vida regional (proxy)
            df['custo_vida_regional'] = (
                df.get('cesta_basica', 500) + 
                df.get('aluguel_medio_m2', 30) * 50 +  # Assumindo 50m²
                df.get('energia_kwh', 0.6) * 150 +     # Assumindo 150kWh/mês
                df.get('transporte_publico', 4) * 22   # Assumindo 22 dias úteis
            )
            
            # Calcular renda disponível
            df['renda_disponivel'] = df[income_col] - df['custo_vida_regional']
            
            # Calcular estresse financeiro
            df['estresse_financeiro'] = np.where(
                df['renda_disponivel'] > 0,
                df[debt_col] / df['renda_disponivel'],
                np.inf  # Estresse infinito se renda disponível <= 0
            )
            
            # Limitar valores extremos
            df['estresse_financeiro'] = np.clip(df['estresse_financeiro'], 0, 10)
            
            self.logger.info(f"Feature criada - Estresse médio: {df['estresse_financeiro'].mean():.2f}")
            
        else:
            self.logger.warning("Colunas de dívida/renda não encontradas. Feature não criada.")
            df['estresse_financeiro'] = np.nan
        
        return df
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features comportamentais baseadas no histórico
        
        Args:
            df: DataFrame com dados do cliente
            
        Returns:
            DataFrame com features comportamentais
        """
        self.logger.info("Criando features comportamentais...")
        
        df = df.copy()
        
        # Feature: Volatilidade de pagamentos
        payment_cols = [col for col in df.columns if 'pagamento' in col.lower() or 'payment' in col.lower()]
        if payment_cols:
            df['volatilidade_pagamentos'] = df[payment_cols].std(axis=1, skipna=True)
        
        # Feature: Razão produto/renda (proxy de impulsividade)
        if 'valor_produto' in df.columns and 'renda_mensal' in df.columns:
            df['razao_produto_renda'] = df['valor_produto'] / df['renda_mensal'].replace(0, np.nan)
        
        # Feature: Tempo de relacionamento (em dias)
        if 'data_primeiro_contrato' in df.columns:
            df['data_primeiro_contrato'] = pd.to_datetime(df['data_primeiro_contrato'], errors='coerce')
            df['tempo_relacionamento_dias'] = (datetime.now() - df['data_primeiro_contrato']).dt.days
        
        # Feature: Idade estimada do produto
        if 'data_compra' in df.columns:
            df['data_compra'] = pd.to_datetime(df['data_compra'], errors='coerce')
            df['idade_produto_dias'] = (datetime.now() - df['data_compra']).dt.days
        
        self.logger.info("Features comportamentais criadas")
        
        return df
    
    def create_regional_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features econômicas regionais derivadas
        
        Args:
            df: DataFrame com dados regionais
            
        Returns:
            DataFrame com features econômicas derivadas
        """
        self.logger.info("Criando features econômicas regionais...")
        
        df = df.copy()
        
        # Feature: Índice de custo de combustível
        combustivel_cols = ['gasolina_comum', 'etanol', 'diesel_s10']
        available_combustivel_cols = [col for col in combustivel_cols if col in df.columns]
        
        if available_combustivel_cols:
            df['indice_custo_combustivel'] = df[available_combustivel_cols].mean(axis=1, skipna=True)
        
        # Feature: Índice de custo de vida
        if 'cesta_basica' in df.columns and 'energia_kwh' in df.columns:
            df['indice_custo_vida'] = (
                df['cesta_basica'] * 0.4 +  # 40% peso para alimentação
                df.get('aluguel_medio_m2', 0) * 50 * 0.3 +  # 30% para moradia
                df['energia_kwh'] * 150 * 0.15 +  # 15% para energia
                df.get('transporte_publico', 0) * 22 * 0.15  # 15% para transporte
            )
        
        # Feature: Poder de compra regional
        if 'pib_per_capita' in df.columns and 'indice_custo_vida' in df.columns:
            df['poder_compra_regional'] = df['pib_per_capita'] / df['indice_custo_vida']
        
        # Feature: Classificação de região econômica
        if 'pib_per_capita' in df.columns:
            df['classificacao_economica'] = pd.cut(
                df['pib_per_capita'], 
                bins=[0, 20000, 35000, 50000, np.inf],
                labels=['Baixa', 'Média-Baixa', 'Média-Alta', 'Alta']
            )
        
        self.logger.info("Features econômicas regionais criadas")
        
        return df
    
    def apply_feature_engineering_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica pipeline completo de feature engineering
        
        Args:
            df: DataFrame integrado
            
        Returns:
            DataFrame com todas as features criadas
        """
        self.logger.info("=== INICIANDO PIPELINE DE FEATURE ENGINEERING ===")
        
        # Criar features financeiras
        df = self.create_financial_stress_feature(df)
        
        # Criar features comportamentais
        df = self.create_behavioral_features(df)
        
        # Criar features econômicas regionais
        df = self.create_regional_economic_features(df)
        
        # Estatísticas finais
        initial_cols = len([col for col in df.columns if not col.startswith('estresse_') 
                           and not col.startswith('volatilidade_')
                           and not col.startswith('indice_')])
        final_cols = len(df.columns)
        new_features = final_cols - initial_cols
        
        self.logger.info(f"Pipeline concluído:")
        self.logger.info(f"  - Features originais: {initial_cols}")
        self.logger.info(f"  - Features finais: {final_cols}")
        self.logger.info(f"  - Novas features criadas: {new_features}")
        
        return df


class DataCleaner:
    """
    Classe responsável pela limpeza e preparação final dos dados
    """
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar sistema de logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'smart') -> pd.DataFrame:
        """
        Trata valores faltantes com estratégias inteligentes
        
        Args:
            df: DataFrame para limpeza
            strategy: Estratégia de tratamento ('smart', 'drop', 'fill')
            
        Returns:
            DataFrame limpo
        """
        self.logger.info("Tratando valores faltantes...")
        
        df = df.copy()
        initial_rows = len(df)
        
        if strategy == 'smart':
            # Estratégias específicas por tipo de coluna
            for col in df.columns:
                missing_pct = df[col].isna().mean()
                
                if missing_pct > 0:
                    self.logger.info(f"  - {col}: {missing_pct:.1%} faltantes")
                    
                    if missing_pct > 0.5:
                        # Muitos valores faltantes - considerar remoção
                        self.logger.warning(f"    Removendo coluna {col} (>50% faltantes)")
                        df.drop(columns=[col], inplace=True)
                        
                    elif df[col].dtype in ['int64', 'float64']:
                        # Variáveis numéricas - usar mediana
                        df[col].fillna(df[col].median(), inplace=True)
                        
                    elif df[col].dtype == 'object':
                        # Variáveis categóricas - usar moda
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            df[col].fillna(mode_val[0], inplace=True)
                        else:
                            df[col].fillna('Desconhecido', inplace=True)
        
        elif strategy == 'drop':
            df.dropna(inplace=True)
            
        elif strategy == 'fill':
            df.fillna(method='forward', inplace=True)
            df.fillna(0, inplace=True)
        
        final_rows = len(df)
        self.logger.info(f"Limpeza concluída: {initial_rows - final_rows} registros removidos")
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers das variáveis numéricas
        
        Args:
            df: DataFrame para limpeza
            method: Método de detecção ('iqr', 'zscore')
            
        Returns:
            DataFrame sem outliers
        """
        self.logger.info(f"Removendo outliers (método: {method})...")
        
        df = df.copy()
        initial_rows = len(df)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers_mask = z_scores > 3
            
            outliers_count = outliers_mask.sum()
            if outliers_count > 0:
                self.logger.info(f"  - {col}: {outliers_count} outliers removidos")
                df = df[~outliers_mask]
        
        final_rows = len(df)
        total_removed = initial_rows - final_rows
        
        self.logger.info(f"Remoção de outliers concluída: {total_removed} registros removidos ({total_removed/initial_rows:.1%})")
        
        return df
    
    def prepare_final_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preparação final do dataset para modelagem
        
        Args:
            df: DataFrame limpo
            
        Returns:
            DataFrame pronto para modelagem
        """
        self.logger.info("=== PREPARAÇÃO FINAL DO DATASET ===")
        
        df = df.copy()
        
        # Remover colunas de identificação/administrativas
        id_cols = [col for col in df.columns if any(x in col.lower() for x in ['id', 'nome', 'cpf', 'data_coleta'])]
        if id_cols:
            self.logger.info(f"Removendo colunas administrativas: {id_cols}")
            df.drop(columns=id_cols, inplace=True)
        
        # Converter variáveis categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            self.logger.info(f"Convertendo {len(categorical_cols)} variáveis categóricas")
            df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
        
        # Garantir que todas as colunas são numéricas
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    df.drop(columns=[col], inplace=True)
        
        # Estatísticas finais
        self.logger.info(f"Dataset final:")
        self.logger.info(f"  - Shape: {df.shape}")
        self.logger.info(f"  - Memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        self.logger.info(f"  - Valores faltantes: {df.isna().sum().sum()}")
        
        return df


if __name__ == "__main__":
    # Exemplo de uso dos módulos
    print("🔧 Módulos de Engenharia de Dados carregados com sucesso!")
    print("📋 Classes disponíveis:")
    print("  - DataIntegrator: Integração de dados internos/externos")
    print("  - FeatureEngineer: Criação de features avançadas") 
    print("  - DataCleaner: Limpeza e preparação final")
