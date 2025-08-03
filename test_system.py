#!/usr/bin/env python3
"""
Script de Teste - Top One Model
==============================

Script para testar rapidamente o sistema com dados de exemplo.
Útil para demonstrações e validação inicial do pipeline.

Uso:
    python test_system.py [--quick]
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Adicionar src ao path
sys.path.append('src')

def create_sample_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    Cria dataset de exemplo para testes
    
    Args:
        n_samples: Número de amostras
        
    Returns:
        DataFrame com dados simulados
    """
    print(f"📊 Criando dataset de exemplo com {n_samples:,} registros...")
    
    np.random.seed(42)
    
    # Listas de cidades brasileiras para simulação
    cidades = [
        'São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Salvador', 
        'Brasília', 'Fortaleza', 'Recife', 'Porto Alegre', 'Curitiba',
        'Manaus', 'Belém', 'Goiânia', 'Guarulhos', 'Campinas',
        'São Luís', 'Maceió', 'Natal', 'João Pessoa', 'Aracaju'
    ]
    
    # Gerar dados
    data = {
        'id_cliente': range(1, n_samples + 1),
        'nome': [f'Cliente_{i}' for i in range(1, n_samples + 1)],
        'idade': np.random.randint(18, 70, n_samples),
        'renda_mensal': np.random.lognormal(mean=8.2, sigma=0.8, size=n_samples),  # Média ~3500
        'divida_total': np.random.lognormal(mean=8.8, sigma=1.0, size=n_samples),  # Média ~6000
        'valor_produto': np.random.lognormal(mean=7.8, sigma=0.7, size=n_samples), # Média ~2500
        'tempo_emprego': np.random.randint(1, 120, n_samples),  # Meses
        'cidade': np.random.choice(cidades, n_samples),
        'estado': np.random.choice(['SP', 'RJ', 'MG', 'BA', 'PR', 'RS', 'PE', 'CE'], n_samples),
        'data_contrato': [
            datetime.now() - timedelta(days=np.random.randint(30, 1095))
            for _ in range(n_samples)
        ]
    }
    
    # Adicionar histórico de pagamentos (12 meses)
    for mes in range(1, 13):
        # Dias de atraso (0 = em dia, valores maiores = atraso)
        # Distribuição realista: maioria paga em dia, alguns com atraso
        atrasos = np.random.choice(
            [0, 0, 0, 0, 0, 1, 2, 3, 5, 10, 15, 30, 60, 90], 
            size=n_samples,
            p=[0.4, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.005, 0.003, 0.001, 0.001]
        )
        data[f'dias_atraso_mes_{mes}'] = atrasos
    
    df = pd.DataFrame(data)
    
    # Adicionar algumas correlações realistas
    # Pessoas com renda maior tendem a ter menos atraso
    for mes in range(1, 13):
        col = f'dias_atraso_mes_{mes}'
        # Ajustar atraso baseado na renda (quanto maior a renda, menor o atraso)
        renda_normalizada = (df['renda_mensal'] - df['renda_mensal'].min()) / (df['renda_mensal'].max() - df['renda_mensal'].min())
        reduction_factor = renda_normalizada * 0.5  # Redução de até 50% no atraso
        df[col] = np.maximum(0, df[col] * (1 - reduction_factor)).astype(int)
    
    print(f"✅ Dataset criado com {len(df)} registros e {len(df.columns)} colunas")
    return df


def test_data_pipeline():
    """Testa pipeline de dados"""
    print("\n🔧 TESTANDO PIPELINE DE DADOS")
    print("=" * 50)
    
    try:
        from data_collection.scraper import RegionalDataScraper
        from data_engineering.data_integration import DataIntegrator, FeatureEngineer
        
        # Criar dados de exemplo
        df_test = create_sample_dataset(100)  # Pequeno para teste rápido
        
        # Testar scraper
        print("🌐 Testando web scraper...")
        scraper = RegionalDataScraper()
        cities_sample = ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte']
        regional_data = scraper.run_full_collection(cities_sample)
        
        if regional_data:
            print("✅ Web scraping funcionando")
        else:
            print("⚠️ Web scraping com problemas (usando dados simulados)")
        
        # Testar engenharia de features
        print("🔧 Testando feature engineering...")
        feature_engineer = FeatureEngineer()
        df_engineered = feature_engineer.apply_feature_engineering_pipeline(df_test)
        
        print(f"✅ Features criadas: {len(df_engineered.columns)} colunas")
        print(f"   Novas features: {len(df_engineered.columns) - len(df_test.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de pipeline: {str(e)}")
        return False


def test_modeling_pipeline():
    """Testa pipeline de modelagem"""
    print("\n🤖 TESTANDO PIPELINE DE MODELAGEM")
    print("=" * 50)
    
    try:
        from modeling.models import RiskSpectrumEncoder, BaselineModel
        
        # Criar dados sintéticos
        df_test = create_sample_dataset(500)
        
        # Testar criação de target
        print("🎯 Testando criação de variável-alvo...")
        risk_encoder = RiskSpectrumEncoder()
        payment_cols = [col for col in df_test.columns if 'dias_atraso' in col]
        df_with_target = risk_encoder.create_target_variable(df_test, payment_cols)
        
        print(f"✅ Target criado: {df_with_target['risk_region'].value_counts().to_dict()}")
        
        # Testar modelo baseline
        print("📊 Testando modelo baseline...")
        
        # Preparar dados para modelo
        feature_cols = ['idade', 'renda_mensal', 'divida_total', 'valor_produto', 'tempo_emprego']
        X = df_with_target[feature_cols].fillna(0)
        y = df_with_target['risk_region_numeric']
        
        # Treinar modelo baseline
        baseline_model = BaselineModel()
        metrics = baseline_model.train(X, y)
        
        print(f"✅ Modelo baseline treinado:")
        print(f"   Accuracy CV: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de modelagem: {str(e)}")
        return False


def test_prediction_system():
    """Testa sistema de predição"""
    print("\n🎯 TESTANDO SISTEMA DE PREDIÇÃO")
    print("=" * 50)
    
    try:
        from prediction.prediction_system import PersonRiskPredictor
        
        # Dados de pessoa fictícia
        person_test = {
            'nome': 'João Teste',
            'idade': 35,
            'renda_mensal': 4000.0,
            'divida_total': 8000.0,
            'cidade': 'São Paulo',
            'estado': 'SP',
            'valor_produto': 3000.0,
            'tempo_emprego': 24
        }
        
        print(f"👤 Testando predição para: {person_test['nome']}")
        
        # Inicializar preditor (pode falhar se modelo não existir)
        predictor = PersonRiskPredictor()
        
        # Testar coleta de dados
        print("📊 Testando coleta de dados...")
        person_data = predictor.collect_person_data(person_test)
        print(f"✅ Dados coletados: {person_data.shape[1]} features")
        
        # Testar engenharia de features
        print("🔧 Testando feature engineering...")
        person_processed = predictor.engineer_features(person_data)
        print(f"✅ Features processadas: {person_processed.shape[1]} features")
        
        print("✅ Sistema de predição testado com sucesso")
        print("⚠️ Predição completa requer modelo treinado")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de predição: {str(e)}")
        print("💡 Dica: Execute 'python main_pipeline.py --mode training' primeiro")
        return False


def run_quick_test():
    """Executa teste rápido do sistema"""
    print("🚀 EXECUTANDO TESTE RÁPIDO")
    print("=" * 60)
    
    # Criar dataset de exemplo
    df_sample = create_sample_dataset(100)
    
    # Salvar para uso posterior
    os.makedirs('data/internal_data', exist_ok=True)
    sample_path = 'data/internal_data/dataset_interno_top_one.xlsx'
    df_sample.to_excel(sample_path, index=False)
    print(f"💾 Dataset de exemplo salvo em: {sample_path}")
    
    # Testar componentes individuais
    tests = [
        ("Pipeline de Dados", test_data_pipeline),
        ("Pipeline de Modelagem", test_modeling_pipeline), 
        ("Sistema de Predição", test_prediction_system)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Falha em {test_name}: {str(e)}")
            results[test_name] = False
    
    # Resumo dos testes
    print("\n📊 RESUMO DOS TESTES")
    print("=" * 30)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        all_passed &= passed
    
    if all_passed:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("💡 Próximo passo: python main_pipeline.py --mode all")
    else:
        print("\n⚠️ ALGUNS TESTES FALHARAM")
        print("💡 Verifique as dependências: pip install -r requirements.txt")
    
    return all_passed


def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Script de teste do Top One Model')
    parser.add_argument('--quick', action='store_true', help='Executar apenas teste rápido')
    
    args = parser.parse_args()
    
    print("🧪 TOP ONE MODEL - SCRIPT DE TESTE")
    print("=" * 60)
    print(f"⏰ Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.quick:
        success = run_quick_test()
    else:
        # Teste completo
        print("🔍 Executando teste completo do sistema...")
        success = run_quick_test()
        
        if success:
            print("\n🚀 Executando pipeline completo...")
            os.system("python main_pipeline.py --mode analysis")
    
    if success:
        print("\n✅ TESTE CONCLUÍDO COM SUCESSO!")
    else:
        print("\n❌ TESTE FALHOU!")
        sys.exit(1)


if __name__ == "__main__":
    main()
