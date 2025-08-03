"""
Sistema de Predição para Novas Pessoas
=====================================

Este módulo implementa o sistema completo para classificar novas pessoas
no espectro de risco, incluindo coleta automática de dados regionais,
engenharia de features e interface de usuário.

Funcionalidades:
- Interface para entrada de dados de nova pessoa
- Coleta automática de dados macro regionais
- Aplicação de pipeline de feature engineering
- Predição usando modelo treinado
- Explicabilidade local com LIME/SHAP
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
import os
from datetime import datetime
import yaml
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Importar módulos internos
import sys
sys.path.append('../')

try:
    from src.data_collection.scraper import RegionalDataScraper
    from src.data_engineering.data_integration import DataIntegrator, FeatureEngineer, DataCleaner
    from src.modeling.models import RiskSpectrumEncoder
except ImportError:
    print("⚠️ Módulos internos não encontrados. Certifique-se de que estão no PATH.")

# Interpretabilidade
try:
    import shap
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    INTERPRETABILITY_AVAILABLE = False
    print("⚠️ Bibliotecas de interpretabilidade não disponíveis.")


class PersonRiskPredictor:
    """
    Classe principal para predição de risco de novas pessoas
    """
    
    def __init__(self, model_path: str = "../models/final_model.pkl",
                 config_path: str = "../config/config.yaml"):
        """
        Inicializa o preditor
        
        Args:
            model_path: Caminho para modelo treinado
            config_path: Caminho para configurações
        """
        self.model_path = model_path
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.risk_encoder = RiskSpectrumEncoder(config_path)
        self.setup_logging()
        
        # Inicializar componentes
        self.scraper = RegionalDataScraper(config_path)
        self.data_integrator = DataIntegrator(config_path)
        self.feature_engineer = FeatureEngineer()
        self.data_cleaner = DataCleaner()
        
        # Carregar modelo se disponível
        self.load_model()
        
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configurações"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return {}
    
    def setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Carrega modelo treinado"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data.get('model')
                self.feature_names = model_data.get('feature_names', [])
                self.scaler = model_data.get('scaler')
                self.logger.info(f"Modelo carregado de: {self.model_path}")
            else:
                self.logger.warning(f"Modelo não encontrado em: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")
    
    def collect_person_data(self, person_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Coleta dados da pessoa e informações regionais
        
        Args:
            person_info: Dicionário com informações da pessoa
            
        Returns:
            DataFrame com dados completos
        """
        self.logger.info("Coletando dados da pessoa e informações regionais...")
        
        # Criar DataFrame base com informações da pessoa
        person_df = pd.DataFrame([person_info])
        
        # Extrair cidade para coleta de dados regionais
        cidade = person_info.get('cidade', person_info.get('municipio', ''))
        
        if cidade:
            try:
                # Coletar dados regionais
                self.logger.info(f"Coletando dados regionais para: {cidade}")
                regional_data = self.scraper.run_full_collection([cidade])
                
                # Integrar dados regionais
                if regional_data:
                    # Dados de combustíveis
                    if 'combustiveis' in regional_data:
                        combustiveis_row = regional_data['combustiveis'].iloc[0:1]
                        person_df = person_df.merge(combustiveis_row, left_on='cidade', 
                                                   right_on='municipio', how='left', suffixes=('', '_comb'))
                    
                    # Dados de utilities
                    if 'utilities' in regional_data:
                        utilities_row = regional_data['utilities'].iloc[0:1]
                        person_df = person_df.merge(utilities_row, left_on='cidade',
                                                   right_on='municipio', how='left', suffixes=('', '_util'))
                    
                    # Indicadores econômicos
                    if 'indicadores' in regional_data:
                        indicadores_row = regional_data['indicadores'].iloc[0:1]
                        person_df = person_df.merge(indicadores_row, left_on='cidade',
                                                   right_on='municipio', how='left', suffixes=('', '_ind'))
                
            except Exception as e:
                self.logger.error(f"Erro ao coletar dados regionais: {str(e)}")
                # Preencher com valores padrão/médios
                self._fill_default_regional_data(person_df)
        else:
            self.logger.warning("Cidade não informada. Usando valores padrão.")
            self._fill_default_regional_data(person_df)
        
        return person_df
    
    def _fill_default_regional_data(self, df: pd.DataFrame):
        """Preenche dados regionais com valores padrão"""
        default_values = {
            'gasolina_comum': 5.5,
            'etanol': 4.0,
            'diesel_s10': 5.2,
            'energia_kwh': 0.6,
            'agua_tarifa_basica': 50.0,
            'gas_botijao_13kg': 100.0,
            'pib_per_capita': 25000,
            'taxa_desemprego': 12.0,
            'cesta_basica': 600.0,
            'aluguel_medio_m2': 25.0,
            'transporte_publico': 4.5
        }
        
        for col, value in default_values.items():
            if col not in df.columns:
                df[col] = value
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica engenharia de features
        
        Args:
            df: DataFrame com dados brutos
            
        Returns:
            DataFrame com features engineered
        """
        self.logger.info("Aplicando engenharia de features...")
        
        # Aplicar pipeline de feature engineering
        df_engineered = self.feature_engineer.apply_feature_engineering_pipeline(df)
        
        # Limpar e preparar dados
        df_final = self.data_cleaner.prepare_final_dataset(df_engineered)
        
        return df_final
    
    def predict_risk(self, person_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Faz predição de risco para a pessoa
        
        Args:
            person_data: DataFrame com dados da pessoa processados
            
        Returns:
            Resultados da predição
        """
        if self.model is None:
            raise ValueError("Modelo não carregado. Treine um modelo primeiro.")
        
        self.logger.info("Fazendo predição de risco...")
        
        try:
            # Alinhar features com o modelo treinado
            model_features = self.feature_names
            
            # Selecionar apenas features do modelo
            available_features = [f for f in model_features if f in person_data.columns]
            missing_features = [f for f in model_features if f not in person_data.columns]
            
            if missing_features:
                self.logger.warning(f"Features faltantes: {missing_features}")
                # Preencher com zeros ou valores padrão
                for feat in missing_features:
                    person_data[feat] = 0
            
            # Preparar dados para predição
            X = person_data[model_features]
            
            # Aplicar scaling se disponível
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
            
            # Fazer predição
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0] if hasattr(self.model, 'predict_proba') else None
            
            # Converter para espectro de risco
            risk_score = probabilities.max() if probabilities is not None else 0.5
            risk_region = self.risk_encoder._map_score_to_region(pd.Series([risk_score]))[0]
            
            # Compilar resultados
            results = {
                'risk_score': float(risk_score),
                'risk_region': risk_region,
                'prediction_class': int(prediction),
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'features_used': available_features,
                'features_missing': missing_features
            }
            
            self.logger.info(f"Predição concluída: {risk_region} (Score: {risk_score:.3f})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro na predição: {str(e)}")
            raise
    
    def explain_prediction(self, person_data: pd.DataFrame, 
                          prediction_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gera explicação da predição usando LIME/SHAP
        
        Args:
            person_data: Dados da pessoa
            prediction_results: Resultados da predição
            
        Returns:
            Explicações da predição
        """
        if not INTERPRETABILITY_AVAILABLE:
            return {"error": "Bibliotecas de interpretabilidade não disponíveis"}
        
        self.logger.info("Gerando explicação da predição...")
        
        try:
            # Preparar dados
            X = person_data[self.feature_names]
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
            
            explanations = {}
            
            # LIME explanation
            try:
                # Criar explainer (usar dados de treino se disponível)
                explainer = LimeTabularExplainer(
                    X_scaled,
                    feature_names=self.feature_names,
                    class_names=[f'Region_{i}' for i in range(5)],
                    mode='classification'
                )
                
                # Gerar explicação
                exp = explainer.explain_instance(
                    X_scaled[0], 
                    self.model.predict_proba,
                    num_features=10
                )
                
                # Extrair features mais importantes
                lime_features = []
                for feature, importance in exp.as_list():
                    lime_features.append({
                        'feature': feature,
                        'importance': float(importance),
                        'impact': 'Positive' if importance > 0 else 'Negative'
                    })
                
                explanations['lime'] = {
                    'top_features': lime_features[:5],
                    'all_features': lime_features
                }
                
            except Exception as e:
                self.logger.error(f"Erro ao gerar explicação LIME: {str(e)}")
                explanations['lime'] = {"error": str(e)}
            
            # Feature importance do modelo (se disponível)
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = []
                for i, importance in enumerate(self.model.feature_importances_):
                    if i < len(self.feature_names):
                        feature_importance.append({
                            'feature': self.feature_names[i],
                            'importance': float(importance),
                            'value': float(X.iloc[0, i])
                        })
                
                # Ordenar por importância
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                explanations['feature_importance'] = feature_importance[:10]
            
            return explanations
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar explicações: {str(e)}")
            return {"error": str(e)}
    
    def predict_new_person(self, person_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pipeline completo para predição de nova pessoa
        
        Args:
            person_info: Informações da pessoa
            
        Returns:
            Resultados completos da análise
        """
        self.logger.info("=== INICIANDO ANÁLISE DE NOVA PESSOA ===")
        
        try:
            # 1. Coletar dados completos
            person_data = self.collect_person_data(person_info)
            
            # 2. Aplicar engenharia de features
            person_processed = self.engineer_features(person_data)
            
            # 3. Fazer predição
            prediction_results = self.predict_risk(person_processed)
            
            # 4. Gerar explicações
            explanations = self.explain_prediction(person_processed, prediction_results)
            
            # 5. Compilar resultados finais
            final_results = {
                'person_info': person_info,
                'prediction': prediction_results,
                'explanations': explanations,
                'analysis_timestamp': datetime.now().isoformat(),
                'model_version': getattr(self.model, 'version', 'unknown')
            }
            
            self.logger.info("=== ANÁLISE CONCLUÍDA ===")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Erro na análise: {str(e)}")
            return {"error": str(e)}


class StreamlitInterface:
    """
    Interface Streamlit para o sistema de predição
    """
    
    def __init__(self, predictor: PersonRiskPredictor):
        """
        Inicializa interface
        
        Args:
            predictor: Instância do preditor
        """
        self.predictor = predictor
    
    def render_interface(self):
        """Renderiza interface principal"""
        st.title("🎯 Sistema de Análise de Risco de Crédito")
        st.subtitle("Classificação no Espectro de Risco - Top One Model")
        
        # Sidebar com informações
        st.sidebar.markdown("## 📊 Espectro de Risco")
        st.sidebar.markdown("""
        **Regiões de Classificação:**
        - 🟢 **Adimplente Pontual** (0.0-0.2)
        - 🟡 **Adimplente Lucrativo** (0.2-0.4)  
        - 🟠 **Risco de Abandono** (0.4-0.6)
        - 🔴 **Inadimplente Parcial** (0.6-0.8)
        - ⚫ **Inadimplente Total** (0.8-1.0)
        """)
        
        # Formulário de entrada
        st.header("📝 Informações da Pessoa")
        
        with st.form("person_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                nome = st.text_input("Nome")
                idade = st.number_input("Idade", min_value=18, max_value=100, value=30)
                renda_mensal = st.number_input("Renda Mensal (R$)", min_value=0.0, value=3000.0)
                divida_total = st.number_input("Dívida Total (R$)", min_value=0.0, value=5000.0)
                
            with col2:
                cidade = st.text_input("Cidade")
                estado = st.selectbox("Estado", 
                                    ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "PE", "CE"])
                valor_produto = st.number_input("Valor do Produto (R$)", min_value=0.0, value=2000.0)
                tempo_emprego = st.number_input("Tempo de Emprego (meses)", min_value=0, value=12)
            
            submitted = st.form_submit_button("🔍 Analisar Risco")
        
        if submitted and nome and cidade:
            # Preparar dados
            person_info = {
                'nome': nome,
                'idade': idade,
                'renda_mensal': renda_mensal,
                'divida_total': divida_total,
                'cidade': cidade,
                'estado': estado,
                'valor_produto': valor_produto,
                'tempo_emprego': tempo_emprego
            }
            
            # Fazer análise
            with st.spinner("Analisando dados... Coletando informações regionais..."):
                results = self.predictor.predict_new_person(person_info)
            
            if 'error' not in results:
                self._display_results(results)
            else:
                st.error(f"Erro na análise: {results['error']}")
    
    def _display_results(self, results: Dict[str, Any]):
        """Exibe resultados da análise"""
        prediction = results['prediction']
        explanations = results.get('explanations', {})
        
        # Resultado principal
        st.header("📊 Resultado da Análise")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_score = prediction['risk_score']
            st.metric("Score de Risco", f"{risk_score:.3f}")
        
        with col2:
            risk_region = prediction['risk_region']
            # Definir cor baseada na região
            region_colors = {
                'Adimplente Pontual': '🟢',
                'Adimplente Lucrativo': '🟡',
                'Risco de Abandono Precoce': '🟠',
                'Inadimplente Parcial': '🔴',
                'Inadimplente Total': '⚫'
            }
            color = region_colors.get(risk_region, '🔵')
            st.metric("Classificação", f"{color} {risk_region}")
        
        with col3:
            if prediction.get('probabilities'):
                confidence = max(prediction['probabilities'])
                st.metric("Confiança", f"{confidence:.1%}")
        
        # Gráfico de probabilidades
        if prediction.get('probabilities'):
            st.subheader("📈 Distribuição de Probabilidades")
            
            region_names = [
                'Adimplente Pontual', 'Adimplente Lucrativo', 'Risco de Abandono',
                'Inadimplente Parcial', 'Inadimplente Total'
            ]
            
            fig = px.bar(
                x=region_names,
                y=prediction['probabilities'],
                title="Probabilidade por Região de Risco"
            )
            fig.update_layout(
                xaxis_title="Região de Risco",
                yaxis_title="Probabilidade",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Explicações
        if explanations and 'feature_importance' in explanations:
            st.subheader("🔍 Principais Fatores de Influência")
            
            top_features = explanations['feature_importance'][:5]
            
            feature_names = [f['feature'] for f in top_features]
            feature_importance = [f['importance'] for f in top_features]
            feature_values = [f['value'] for f in top_features]
            
            # Gráfico de importância
            fig = px.bar(
                x=feature_importance,
                y=feature_names,
                orientation='h',
                title="Top 5 Features Mais Importantes"
            )
            fig.update_layout(
                xaxis_title="Importância",
                yaxis_title="Feature",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela detalhada
            st.subheader("📋 Detalhamento das Features")
            feature_df = pd.DataFrame(top_features)
            st.dataframe(feature_df, use_container_width=True)
        
        # Informações técnicas (expansível)
        with st.expander("🔧 Informações Técnicas"):
            st.json({
                'features_utilizadas': len(prediction.get('features_used', [])),
                'features_faltantes': len(prediction.get('features_missing', [])),
                'timestamp_analise': results.get('analysis_timestamp'),
                'versao_modelo': results.get('model_version')
            })


def create_sample_interface():
    """Cria interface de exemplo"""
    
    # Inicializar preditor
    predictor = PersonRiskPredictor()
    
    # Criar interface
    interface = StreamlitInterface(predictor)
    
    # Renderizar
    interface.render_interface()


if __name__ == "__main__":
    print("🎯 Sistema de Predição carregado com sucesso!")
    print("📋 Classes disponíveis:")
    print("  - PersonRiskPredictor: Predição completa de risco")
    print("  - StreamlitInterface: Interface web interativa")
    print("\n🚀 Para executar a interface:")
    print("  streamlit run prediction_system.py")
