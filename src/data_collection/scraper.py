"""
Sistema de Web Scraping para Dados Regionais
============================================

Este módulo implementa o sistema de coleta automatizada de dados
macroeconômicos e de custo de vida por região/município.

Funcionalidades:
- Coleta de preços de combustíveis
- Dados de utilities (energia, água, gás)
- Indicadores econômicos regionais
- Sistema de validação e qualidade de dados
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional
import yaml
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
from datetime import datetime, timedelta


class RegionalDataScraper:
    """
    Classe principal para web scraping de dados regionais
    """
    
    def __init__(self, config_path: str = "../config/config.yaml"):
        """
        Inicializa o scraper com configurações
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configurações do arquivo YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logging.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Configurações padrão caso arquivo não seja encontrado"""
        return {
            'scraping': {
                'frequency_days': 45,
                'combustiveis': {'frequency_days': 7},
                'utilities': {'frequency_days': 30}
            }
        }
    
    def setup_logging(self):
        """Configurar sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('../logs/scraping.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_chrome_driver(self) -> webdriver.Chrome:
        """
        Configura e retorna driver do Chrome para Selenium
        """
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Executar sem interface gráfica
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        
        return webdriver.Chrome(options=chrome_options)
    
    def scrape_combustiveis_anp(self, municipios: List[str]) -> pd.DataFrame:
        """
        Coleta preços de combustíveis da ANP por município
        
        Args:
            municipios: Lista de municípios para coleta
            
        Returns:
            DataFrame com preços de combustíveis por município
        """
        self.logger.info(f"Iniciando coleta de preços de combustíveis para {len(municipios)} municípios")
        
        combustiveis_data = []
        
        for municipio in municipios:
            try:
                # Simular coleta (implementar URLs reais da ANP)
                data = {
                    'municipio': municipio,
                    'gasolina_comum': np.random.uniform(5.0, 6.5),  # Valores simulados
                    'gasolina_aditivada': np.random.uniform(5.5, 7.0),
                    'etanol': np.random.uniform(3.5, 4.5),
                    'diesel_s10': np.random.uniform(4.8, 6.0),
                    'data_coleta': datetime.now(),
                    'fonte': 'ANP'
                }
                combustiveis_data.append(data)
                
                # Delay para evitar sobrecarga
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Erro ao coletar dados de {municipio}: {str(e)}")
                continue
        
        df_combustiveis = pd.DataFrame(combustiveis_data)
        self.logger.info(f"Coleta concluída: {len(df_combustiveis)} registros coletados")
        
        return df_combustiveis
    
    def scrape_utilities_data(self, municipios: List[str]) -> pd.DataFrame:
        """
        Coleta dados de utilities (energia, água, gás) por município
        
        Args:
            municipios: Lista de municípios para coleta
            
        Returns:
            DataFrame com dados de utilities por município
        """
        self.logger.info(f"Iniciando coleta de dados de utilities para {len(municipios)} municípios")
        
        utilities_data = []
        
        for municipio in municipios:
            try:
                # Simular coleta (implementar APIs/sites reais)
                data = {
                    'municipio': municipio,
                    'energia_kwh': np.random.uniform(0.4, 0.8),  # R$/kWh
                    'agua_tarifa_basica': np.random.uniform(25, 80),  # R$/mês
                    'gas_botijao_13kg': np.random.uniform(80, 120),  # R$/botijão
                    'data_coleta': datetime.now(),
                    'fonte': 'ANEEL/SNIS'
                }
                utilities_data.append(data)
                
                time.sleep(0.3)
                
            except Exception as e:
                self.logger.error(f"Erro ao coletar utilities de {municipio}: {str(e)}")
                continue
        
        df_utilities = pd.DataFrame(utilities_data)
        self.logger.info(f"Coleta de utilities concluída: {len(df_utilities)} registros")
        
        return df_utilities
    
    def scrape_indicadores_economicos(self, municipios: List[str]) -> pd.DataFrame:
        """
        Coleta indicadores econômicos regionais
        
        Args:
            municipios: Lista de municípios
            
        Returns:
            DataFrame com indicadores econômicos
        """
        self.logger.info(f"Coletando indicadores econômicos para {len(municipios)} municípios")
        
        indicadores_data = []
        
        for municipio in municipios:
            try:
                # Simular dados (implementar APIs do IBGE/BACEN)
                data = {
                    'municipio': municipio,
                    'pib_per_capita': np.random.uniform(15000, 45000),
                    'taxa_desemprego': np.random.uniform(8, 18),
                    'inflacao_regional': np.random.uniform(3, 8),
                    'cesta_basica': np.random.uniform(400, 800),
                    'aluguel_medio_m2': np.random.uniform(15, 50),
                    'transporte_publico': np.random.uniform(3.5, 6.5),
                    'data_coleta': datetime.now(),
                    'fonte': 'IBGE/BACEN'
                }
                indicadores_data.append(data)
                
                time.sleep(0.2)
                
            except Exception as e:
                self.logger.error(f"Erro ao coletar indicadores de {municipio}: {str(e)}")
                continue
        
        df_indicadores = pd.DataFrame(indicadores_data)
        self.logger.info(f"Coleta de indicadores concluída: {len(df_indicadores)} registros")
        
        return df_indicadores
    
    def validate_data_quality(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Valida qualidade dos dados coletados
        
        Args:
            df: DataFrame para validação
            data_type: Tipo de dados (combustiveis, utilities, indicadores)
            
        Returns:
            DataFrame validado e limpo
        """
        self.logger.info(f"Validando qualidade dos dados: {data_type}")
        
        initial_rows = len(df)
        
        # Remover duplicatas
        df = df.drop_duplicates(subset=['municipio'])
        
        # Remover valores extremos (outliers)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'data_coleta':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        
        self.logger.info(f"Validação concluída: {removed_rows} registros removidos ({removed_rows/initial_rows*100:.1f}%)")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """
        Salva dados coletados em arquivo
        
        Args:
            df: DataFrame para salvar
            filename: Nome do arquivo
        """
        os.makedirs("../data/external", exist_ok=True)
        filepath = f"../data/external/{filename}"
        
        df.to_csv(filepath, index=False, encoding='utf-8')
        self.logger.info(f"Dados salvos em: {filepath}")
    
    def run_full_collection(self, municipios: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Executa coleta completa de todos os tipos de dados
        
        Args:
            municipios: Lista de municípios para coleta
            
        Returns:
            Dicionário com DataFrames coletados
        """
        self.logger.info("=== INICIANDO COLETA COMPLETA DE DADOS REGIONAIS ===")
        
        results = {}
        
        # Coletar combustíveis
        try:
            df_combustiveis = self.scrape_combustiveis_anp(municipios)
            df_combustiveis = self.validate_data_quality(df_combustiveis, 'combustiveis')
            self.save_data(df_combustiveis, f"combustiveis_{datetime.now().strftime('%Y%m%d')}.csv")
            results['combustiveis'] = df_combustiveis
        except Exception as e:
            self.logger.error(f"Falha na coleta de combustíveis: {str(e)}")
        
        # Coletar utilities
        try:
            df_utilities = self.scrape_utilities_data(municipios)
            df_utilities = self.validate_data_quality(df_utilities, 'utilities')
            self.save_data(df_utilities, f"utilities_{datetime.now().strftime('%Y%m%d')}.csv")
            results['utilities'] = df_utilities
        except Exception as e:
            self.logger.error(f"Falha na coleta de utilities: {str(e)}")
        
        # Coletar indicadores econômicos
        try:
            df_indicadores = self.scrape_indicadores_economicos(municipios)
            df_indicadores = self.validate_data_quality(df_indicadores, 'indicadores')
            self.save_data(df_indicadores, f"indicadores_{datetime.now().strftime('%Y%m%d')}.csv")
            results['indicadores'] = df_indicadores
        except Exception as e:
            self.logger.error(f"Falha na coleta de indicadores: {str(e)}")
        
        self.logger.info("=== COLETA COMPLETA FINALIZADA ===")
        return results


def get_unique_cities_from_internal_data(df_internal: pd.DataFrame) -> List[str]:
    """
    Extrai lista única de cidades do dataset interno
    
    Args:
        df_internal: DataFrame do dataset interno
        
    Returns:
        Lista de cidades únicas
    """
    # Assumindo que existe uma coluna 'cidade' ou similar
    # Ajustar conforme estrutura real do dataset
    possible_city_columns = ['cidade', 'municipio', 'city', 'municipality']
    
    city_column = None
    for col in possible_city_columns:
        if col in df_internal.columns:
            city_column = col
            break
    
    if city_column:
        cities = df_internal[city_column].dropna().unique().tolist()
        return [city.strip().title() for city in cities if isinstance(city, str)]
    else:
        print("⚠️ Coluna de cidade não encontrada no dataset interno")
        return []


if __name__ == "__main__":
    # Exemplo de uso
    scraper = RegionalDataScraper()
    
    # Lista de municípios para teste
    municipios_teste = [
        "São Paulo", "Rio de Janeiro", "Belo Horizonte", 
        "Porto Alegre", "Salvador", "Recife", "Fortaleza"
    ]
    
    # Executar coleta
    results = scraper.run_full_collection(municipios_teste)
    
    # Mostrar resultados
    for data_type, df in results.items():
        print(f"\n{data_type.upper()}:")
        print(f"Shape: {df.shape}")
        print(df.head())
