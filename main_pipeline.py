"""
Pipeline Principal - Top One Model
=================================

Script principal que orquestra todo o pipeline do projeto:
1. Análise do dataset interno
2. Coleta de dados regionais via web scraping  
3. Engenharia de features e integração de dados
4. Treinamento dos modelos
5. Validação e backtesting
6. Sistema de predição para novas pessoas

Uso:
    python main_pipeline.py --mode [analysis|training|prediction]
"""

import sys
import os
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Adicionar src ao path
sys.path.append('src')

# Importar módulos do projeto
try:
    from data_collection.scraper import RegionalDataScraper
    from data_engineering.data_integration import DataIntegrator, FeatureEngineer, DataCleaner
    from modeling.models import RiskSpectrumEncoder, BaselineModel, EnsembleModel, ModelEvaluator
    from validation.model_validation import TemporalValidator, BacktestingEngine, DriftDetector
    from prediction.prediction_system import PersonRiskPredictor
except ImportError as e:
    print(f"❌ Erro ao importar módulos: {e}")
    print("Certifique-se de que todas as dependências estão instaladas")
    sys.exit(1)


def setup_logging():
    """Configurar sistema de logging"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class MainPipeline:
    """
    Classe principal que orquestra todo o pipeline
    """
    
    def __init__(self):
        self.logger = setup_logging()
        self.logger.info("=== INICIANDO PIPELINE TOP ONE MODEL ===")
        
        # Inicializar componentes
        self.scraper = RegionalDataScraper()
        self.data_integrator = DataIntegrator()
        self.feature_engineer = FeatureEngineer()
        self.data_cleaner = DataCleaner()
        self.risk_encoder = RiskSpectrumEncoder()
        self.evaluator = ModelEvaluator()
        
        # Variáveis para armazenar dados processados
        self.df_internal = None
        self.df_final = None
        self.models = {}
        
    def load_internal_dataset(self, dataset_path: str = "data/internal_data/dataset_interno_top_one.xlsx"):
        """
        Carrega e faz análise inicial do dataset interno
        
        Args:
            dataset_path: Caminho para o dataset interno
        """
        self.logger.info("🔍 FASE 1: CARREGANDO DATASET INTERNO")
        
        try:
            # Carregar dataset
            self.logger.info(f"Carregando dataset de: {dataset_path}")
            self.df_internal = pd.read_excel(dataset_path)
            
            self.logger.info(f"✅ Dataset carregado: {self.df_internal.shape[0]:,} linhas x {self.df_internal.shape[1]} colunas")
            
            # Análise básica
            self.logger.info("📊 Análise básica do dataset:")
            self.logger.info(f"  - Tamanho em memória: {self.df_internal.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
            self.logger.info(f"  - Valores faltantes: {self.df_internal.isnull().sum().sum():,}")
            self.logger.info(f"  - Tipos de dados: {self.df_internal.dtypes.value_counts().to_dict()}")
            
            # Identificar coluna de cidade
            possible_city_cols = ['cidade', 'municipio', 'city', 'municipality']
            city_col = None
            for col in possible_city_cols:
                if col in self.df_internal.columns:
                    city_col = col
                    break
            
            if city_col:
                unique_cities = self.df_internal[city_col].nunique()
                self.logger.info(f"  - Cidades únicas encontradas: {unique_cities}")
                self.city_column = city_col
            else:
                self.logger.warning("⚠️ Coluna de cidade não identificada automaticamente")
                self.city_column = None
                
            return True
            
        except FileNotFoundError:
            self.logger.error(f"❌ Arquivo não encontrado: {dataset_path}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dataset: {str(e)}")
            return False
    
    def collect_regional_data(self):
        """
        Coleta dados regionais via web scraping
        """
        self.logger.info("🌐 FASE 2: COLETANDO DADOS REGIONAIS")
        
        if self.df_internal is None:
            self.logger.error("❌ Dataset interno não carregado")
            return False
        
        try:
            # Extrair lista de cidades únicas
            if self.city_column:
                cities = self.df_internal[self.city_column].dropna().unique()
                cities = [str(city).strip().title() for city in cities if str(city) != 'nan']
                cities = list(set(cities))  # Remover duplicatas
                
                self.logger.info(f"📍 Coletando dados para {len(cities)} cidades")
                
                # Limitar número de cidades para demonstração (remover em produção)
                if len(cities) > 20:
                    cities = cities[:20]
                    self.logger.info(f"⚠️ Limitando para {len(cities)} cidades para demonstração")
                
                # Executar coleta
                regional_data = self.scraper.run_full_collection(cities)
                
                if regional_data:
                    self.logger.info("✅ Dados regionais coletados com sucesso")
                    self.regional_data = regional_data
                    return True
                else:
                    self.logger.error("❌ Falha na coleta de dados regionais")
                    return False
            else:
                self.logger.error("❌ Coluna de cidade não identificada")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro na coleta de dados regionais: {str(e)}")
            return False
    
    def integrate_and_engineer_features(self):
        """
        Integra dados e aplica engenharia de features
        """
        self.logger.info("🔧 FASE 3: INTEGRAÇÃO E ENGENHARIA DE FEATURES")
        
        try:
            # Integrar dados internos com externos
            self.logger.info("Integrando dados internos e externos...")
            
            if hasattr(self, 'regional_data') and self.regional_data:
                df_combustiveis = self.regional_data.get('combustiveis', pd.DataFrame())
                df_utilities = self.regional_data.get('utilities', pd.DataFrame()) 
                df_indicadores = self.regional_data.get('indicadores', pd.DataFrame())
            else:
                # Criar DataFrames vazios se dados não coletados
                df_combustiveis = pd.DataFrame()
                df_utilities = pd.DataFrame()
                df_indicadores = pd.DataFrame()
            
            # Integrar dados
            if not df_combustiveis.empty and not df_utilities.empty and not df_indicadores.empty:
                df_integrated = self.data_integrator.merge_internal_external_data(
                    self.df_internal, df_combustiveis, df_utilities, df_indicadores, self.city_column
                )
            else:
                self.logger.warning("⚠️ Dados regionais não disponíveis. Usando apenas dados internos.")
                df_integrated = self.df_internal.copy()
            
            # Aplicar engenharia de features
            self.logger.info("Aplicando engenharia de features...")
            df_engineered = self.feature_engineer.apply_feature_engineering_pipeline(df_integrated)
            
            # Limpeza e preparação final
            self.logger.info("Limpeza e preparação final dos dados...")
            self.df_final = self.data_cleaner.handle_missing_values(df_engineered)
            self.df_final = self.data_cleaner.remove_outliers(self.df_final)
            self.df_final = self.data_cleaner.prepare_final_dataset(self.df_final)
            
            self.logger.info(f"✅ Dados finais preparados: {self.df_final.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro na engenharia de features: {str(e)}")
            return False
    
    def create_target_variable(self):
        """
        Cria variável-alvo baseada no espectro de risco
        """
        self.logger.info("🎯 FASE 4: CRIANDO VARIÁVEL-ALVO")
        
        try:
            # Identificar colunas de histórico de pagamento
            payment_cols = [col for col in self.df_final.columns 
                          if any(keyword in col.lower() for keyword in ['pagamento', 'payment', 'atraso', 'delay'])]
            
            if not payment_cols:
                self.logger.warning("⚠️ Colunas de histórico de pagamento não encontradas. Simulando dados.")
                # Simular colunas de pagamento para demonstração
                np.random.seed(42)
                for i in range(12):  # 12 meses de histórico
                    col_name = f'dias_atraso_mes_{i+1}'
                    self.df_final[col_name] = np.random.poisson(5, size=len(self.df_final))
                payment_cols = [col for col in self.df_final.columns if 'dias_atraso' in col]
            
            # Criar variável-alvo
            self.df_final = self.risk_encoder.create_target_variable(self.df_final, payment_cols)
            
            self.logger.info("✅ Variável-alvo criada com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao criar variável-alvo: {str(e)}")
            return False
    
    def train_models(self):
        """
        Treina modelos conforme arquitetura especificada
        """
        self.logger.info("🤖 FASE 5: TREINAMENTO DE MODELOS")
        
        try:
            # Preparar dados para treinamento
            target_col = 'risk_region_numeric'
            if target_col not in self.df_final.columns:
                self.logger.error(f"❌ Coluna target '{target_col}' não encontrada")
                return False
            
            # Separar features e target
            feature_cols = [col for col in self.df_final.columns 
                          if col not in [target_col, 'risk_region', 'risk_score']]
            
            X = self.df_final[feature_cols]
            y = self.df_final[target_col]
            
            self.logger.info(f"Features para treinamento: {len(feature_cols)}")
            self.logger.info(f"Amostras para treinamento: {len(X)}")
            
            # Split treino/teste
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 1. Modelo Baseline (Regressão Logística)
            self.logger.info("Treinando modelo baseline (Logistic Regression)...")
            baseline_model = BaselineModel()
            baseline_metrics = baseline_model.train(X_train, y_train)
            self.models['baseline'] = baseline_model
            
            # 2. Modelo XGBoost
            self.logger.info("Treinando modelo XGBoost...")
            xgb_model = EnsembleModel('xgboost')
            X_train_val, X_val, y_train_val, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            xgb_metrics = xgb_model.train(X_train_val, y_train_val, X_val, y_val)
            self.models['xgboost'] = xgb_model
            
            # 3. Modelo LightGBM
            self.logger.info("Treinando modelo LightGBM...")
            lgb_model = EnsembleModel('lightgbm')
            lgb_metrics = lgb_model.train(X_train_val, y_train_val, X_val, y_val)
            self.models['lightgbm'] = lgb_model
            
            # Salvar dados de teste para validação
            self.X_test = X_test
            self.y_test = y_test
            
            self.logger.info("✅ Todos os modelos treinados com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro no treinamento de modelos: {str(e)}")
            return False
    
    def validate_models(self):
        """
        Valida modelos usando estratégias robustas
        """
        self.logger.info("✅ FASE 6: VALIDAÇÃO DE MODELOS")
        
        try:
            # Avaliar cada modelo
            model_results = {}
            
            for model_name, model in self.models.items():
                self.logger.info(f"Avaliando modelo: {model_name}")
                
                # Avaliar no conjunto de teste
                results = self.evaluator.evaluate_model(model, self.X_test, self.y_test)
                model_results[model_name] = results
                
                self.logger.info(f"  - Accuracy: {results['accuracy']:.3f}")
                self.logger.info(f"  - F1-Score: {results['f1_score']:.3f}")
            
            # Comparar modelos
            comparison_df = self.evaluator.compare_models(model_results)
            
            # Identificar melhor modelo
            best_model_name = comparison_df.iloc[0]['model']
            self.best_model = self.models[best_model_name]
            self.best_model_name = best_model_name
            
            self.logger.info(f"🏆 Melhor modelo: {best_model_name}")
            
            # Salvar melhor modelo
            os.makedirs('models', exist_ok=True)
            self.best_model.save_model(f'models/best_model_{best_model_name}.pkl')
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro na validação: {str(e)}")
            return False
    
    def run_analysis_mode(self):
        """Executa modo de análise completa"""
        self.logger.info("📊 EXECUTANDO MODO: ANÁLISE COMPLETA")
        
        # Executar todas as fases
        success = True
        success &= self.load_internal_dataset()
        success &= self.collect_regional_data()
        success &= self.integrate_and_engineer_features()
        success &= self.create_target_variable()
        
        if success:
            self.logger.info("✅ ANÁLISE COMPLETA CONCLUÍDA COM SUCESSO")
            
            # Salvar dados processados
            os.makedirs('data/processed', exist_ok=True)
            self.df_final.to_csv('data/processed/dataset_final.csv', index=False)
            self.logger.info("💾 Dados processados salvos em: data/processed/dataset_final.csv")
        else:
            self.logger.error("❌ FALHAS DURANTE A ANÁLISE")
        
        return success
    
    def run_training_mode(self):
        """Executa modo de treinamento"""
        self.logger.info("🤖 EXECUTANDO MODO: TREINAMENTO")
        
        # Verificar se dados processados existem
        processed_data_path = 'data/processed/dataset_final.csv'
        if os.path.exists(processed_data_path):
            self.logger.info("Carregando dados processados existentes...")
            self.df_final = pd.read_csv(processed_data_path)
        else:
            self.logger.info("Dados processados não encontrados. Executando análise completa...")
            if not self.run_analysis_mode():
                return False
        
        # Treinar e validar modelos
        success = True
        success &= self.create_target_variable() if 'risk_region_numeric' not in self.df_final.columns else True
        success &= self.train_models()
        success &= self.validate_models()
        
        if success:
            self.logger.info("✅ TREINAMENTO CONCLUÍDO COM SUCESSO")
        else:
            self.logger.error("❌ FALHAS DURANTE O TREINAMENTO")
        
        return success
    
    def run_prediction_mode(self):
        """Executa modo de predição"""
        self.logger.info("🎯 EXECUTANDO MODO: PREDIÇÃO")
        
        try:
            # Inicializar sistema de predição
            predictor = PersonRiskPredictor()
            
            # Exemplo de predição
            person_example = {
                'nome': 'João Silva',
                'idade': 35,
                'renda_mensal': 4000.0,
                'divida_total': 8000.0,
                'cidade': 'São Paulo',
                'estado': 'SP',
                'valor_produto': 3000.0,
                'tempo_emprego': 24
            }
            
            self.logger.info("Executando predição de exemplo...")
            results = predictor.predict_new_person(person_example)
            
            if 'error' not in results:
                prediction = results['prediction']
                self.logger.info(f"✅ Predição concluída:")
                self.logger.info(f"  - Região de Risco: {prediction['risk_region']}")
                self.logger.info(f"  - Score: {prediction['risk_score']:.3f}")
                self.logger.info(f"  - Classe: {prediction['prediction_class']}")
            else:
                self.logger.error(f"❌ Erro na predição: {results['error']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro no modo predição: {str(e)}")
            return False


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Pipeline Top One Model')
    parser.add_argument('--mode', choices=['analysis', 'training', 'prediction', 'all'],
                       default='analysis', help='Modo de execução')
    
    args = parser.parse_args()
    
    # Inicializar pipeline
    pipeline = MainPipeline()
    
    # Executar modo selecionado
    success = False
    
    if args.mode == 'analysis':
        success = pipeline.run_analysis_mode()
    elif args.mode == 'training':
        success = pipeline.run_training_mode()
    elif args.mode == 'prediction':
        success = pipeline.run_prediction_mode()
    elif args.mode == 'all':
        success = pipeline.run_analysis_mode()
        if success:
            success = pipeline.run_training_mode()
        if success:
            success = pipeline.run_prediction_mode()
    
    # Resultado final
    if success:
        pipeline.logger.info("🎉 PIPELINE EXECUTADO COM SUCESSO!")
        sys.exit(0)
    else:
        pipeline.logger.error("💥 PIPELINE FALHOU!")
        sys.exit(1)


if __name__ == "__main__":
    main()
