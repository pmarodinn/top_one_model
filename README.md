# 🎯 Top One Model - Sistema de Modelagem de Risco de Crédito

## 📋 Visão Geral

Sistema avançado de modelagem de risco de crédito que vai além da classificação binária tradicional, implementando um **espectro contínuo de risco** com 5 regiões e integração de dados macroeconômicos regionais via web scraping.

### 🎯 Características Principais

- **Espectro de Risco Contínuo**: 5 regiões (Adimplente Pontual → Inadimplente Total)
- **Dados Regionais**: Web scraping automático de combustíveis, utilities e indicadores econômicos
- **Feature Engineering Avançada**: Features sintéticas como "Estresse Financeiro"
- **Modelos Múltiplos**: Baseline, XGBoost, LightGBM com arquitetura híbrida
- **Validação Temporal**: Backtesting robusto e detecção de drift
- **Interpretabilidade**: LIME/SHAP para explicação das predições
- **Interface Web**: Sistema interativo para classificação de novas pessoas

---

## 🏗️ Estrutura do Projeto

```
top_one_model/
├── 📁 data/
│   ├── internal_data/          # Dataset interno (.xlsx)
│   ├── external/              # Dados coletados via web scraping
│   └── processed/             # Dados processados finais
├── 📁 src/
│   ├── data_collection/       # 🌐 Módulos de web scraping
│   ├── data_engineering/      # 🔧 Engenharia de features
│   ├── modeling/              # 🤖 Modelos de ML
│   ├── validation/            # ✅ Validação e testes
│   └── prediction/            # 🎯 Sistema de predição
├── 📁 notebooks/              # 📊 Análises exploratórias
├── 📁 models/                 # 💾 Modelos treinados
├── 📁 config/                 # ⚙️ Configurações
├── 📁 logs/                   # 📝 Logs de execução
├── main_pipeline.py           # 🚀 Pipeline principal
├── requirements.txt           # 📦 Dependências
└── README.md                  # 📖 Este arquivo
```

---

## ⚡ Quick Start

### 1. Instalação

```bash
# Clonar/navegar para o diretório do projeto
cd top_one_model/

# Instalar dependências
pip install -r requirements.txt

# Criar diretórios necessários
mkdir -p logs models data/processed
```

### 2. Preparar Dataset Interno

Coloque seu dataset interno em:
```
data/internal_data/dataset_interno_top_one.xlsx
```

**Colunas esperadas:**
- `cidade` ou `municipio`: Para associação com dados regionais
- `renda_mensal`, `divida_total`: Para feature "Estresse Financeiro"
- Colunas de histórico de pagamento (opcional - serão simuladas se não existirem)

### 3. Executar Pipeline

```bash
# Análise completa + Treinamento + Predição
python main_pipeline.py --mode all

# Ou executar etapas individuais:
python main_pipeline.py --mode analysis    # Análise e feature engineering
python main_pipeline.py --mode training    # Treinamento de modelos
python main_pipeline.py --mode prediction  # Sistema de predição
```

### 4. Interface Web (Opcional)

```bash
# Instalar Streamlit
pip install streamlit

# Executar interface
streamlit run src/prediction/prediction_system.py
```

---

## 🔄 Pipeline Detalhado

### Fase 1: Análise do Dataset Interno
- ✅ Carregamento e validação do dataset (.xlsx)
- ✅ Análise estatística descritiva
- ✅ Identificação de colunas geográficas
- ✅ Verificação de qualidade dos dados

### Fase 2: Coleta de Dados Regionais (Web Scraping)
- 🌐 **Combustíveis**: Gasolina, etanol, diesel por município
- ⚡ **Utilities**: Energia elétrica, água, gás por região
- 📊 **Indicadores**: PIB per capita, desemprego, cesta básica
- 🔄 **Frequência**: Configurável (padrão: 45 dias)

### Fase 3: Engenharia de Features
- 🔧 **Integração**: Merge de dados internos + externos por cidade
- 💰 **Feature Sintética**: Estresse Financeiro = Dívida / (Renda - Custo Regional)
- 📈 **Features Comportamentais**: Volatilidade de pagamentos, razão produto/renda
- 🏙️ **Features Regionais**: Índices de custo de vida e poder de compra

### Fase 4: Modelagem (Arquitetura em Estágios)
- 📊 **Baseline**: Regressão Logística Multinomial
- 🚀 **Ensemble**: XGBoost e LightGBM otimizados
- 🧠 **Opcional**: LSTM para extração de features sequenciais
- 🎯 **Target**: Espectro contínuo 0-1 mapeado em 5 regiões

### Fase 5: Validação Temporal
- ⏰ **Validação Cruzada**: Sliding window temporal
- 🧪 **Backtesting**: Hold-out dos últimos 6 meses
- 📊 **Métricas**: Accuracy, F1, AUC, PSI para drift detection
- 🔍 **Monitoramento**: Detecção automática de concept drift

### Fase 6: Sistema de Predição
- 👤 **Nova Pessoa**: Interface para entrada de dados
- 🌐 **Coleta Auto**: Dados regionais em tempo real
- 🎯 **Classificação**: Espectro de risco + confiança
- 🔍 **Explicabilidade**: Top 5 fatores de influência

---

## 📊 Espectro de Risco

| Score | Região | Descrição | Ação Recomendada |
|-------|--------|-----------|------------------|
| 0.0-0.2 | 🟢 **Adimplente Pontual** | Pagamentos consistentes em dia | Ofertas de novos produtos |
| 0.2-0.4 | 🟡 **Adimplente Lucrativo** | Atrasos sistemáticos mas paga | Monitoramento preventivo |
| 0.4-0.6 | 🟠 **Risco de Abandono** | Cessação precoce de pagamentos | Ação de cobrança imediata |
| 0.6-0.8 | 🔴 **Inadimplente Parcial** | Recuperação parcial possível | Estratégia de renegociação |
| 0.8-1.0 | ⚫ **Inadimplente Total** | Perda total ou quase total | Provisão de perda |

---

## 🛠️ Configurações

### Arquivo `config/config.yaml`

```yaml
# Configurações de Web Scraping
scraping:
  frequency_days: 45
  combustiveis:
    frequency_days: 7
  utilities:
    frequency_days: 30

# Configurações de Modelagem  
modeling:
  retrain_frequency_days: 45
  validation_window_months: 18

# Monitoramento
monitoring:
  drift_threshold: 0.15
  performance_window_days: 30
```

---

## 🎯 Uso do Sistema de Predição

### Via Código Python

```python
from src.prediction.prediction_system import PersonRiskPredictor

# Inicializar preditor
predictor = PersonRiskPredictor()

# Dados da nova pessoa
person_info = {
    'nome': 'Maria Santos',
    'idade': 28,
    'renda_mensal': 3500.0,
    'divida_total': 6000.0,
    'cidade': 'Rio de Janeiro',
    'estado': 'RJ',
    'valor_produto': 2500.0,
    'tempo_emprego': 18
}

# Fazer predição completa
results = predictor.predict_new_person(person_info)

# Acessar resultados
prediction = results['prediction']
print(f"Região de Risco: {prediction['risk_region']}")
print(f"Score: {prediction['risk_score']:.3f}")
print(f"Confiança: {max(prediction['probabilities']):.1%}")
```

### Via Interface Web

1. Execute: `streamlit run src/prediction/prediction_system.py`
2. Acesse: `http://localhost:8501`
3. Preencha o formulário com dados da pessoa
4. Visualize resultados e explicações interativas

---

## 📈 Monitoramento e MLOps

### Sistema de Atualização Contínua

- **Coleta Automática**: A cada 45 dias, novos dados são coletados
- **Detecção de Drift**: Testes KS e PSI para identificar mudanças
- **Retreinamento**: Automático quando drift > 15%
- **A/B Testing**: Framework Champion/Challenger
- **Versionamento**: Modelos com rastreabilidade completa

### Métricas Monitoradas

- **Performance**: Accuracy, Precision, Recall, F1-Score
- **Estabilidade**: Population Stability Index (PSI)
- **Drift**: Kolmogorov-Smirnov test
- **Cobertura**: % dados regionais disponíveis

---

## 🔧 Desenvolvimento e Customização

### Adicionando Novas Fontes de Dados

1. Edite `src/data_collection/scraper.py`
2. Adicione nova função de coleta
3. Atualize `run_full_collection()`
4. Configure frequência em `config.yaml`

### Criando Novas Features

1. Edite `src/data_engineering/data_integration.py`
2. Adicione função na classe `FeatureEngineer`
3. Inclua no pipeline `apply_feature_engineering_pipeline()`

### Adicionando Novos Modelos

1. Crie classe em `src/modeling/models.py`
2. Implemente métodos `train()` e `predict()`
3. Adicione ao pipeline principal

---

## ⚠️ Troubleshooting

### Problemas Comuns

**1. Erro de importação de bibliotecas**
```bash
pip install -r requirements.txt
```

**2. Dataset interno não encontrado**
- Verifique se está em `data/internal_data/dataset_interno_top_one.xlsx`
- Confirme formato Excel (.xlsx)

**3. Falha no web scraping**
- Verifique conexão de internet
- Alguns sites podem ter mudado estrutura HTML
- Dados simulados serão usados como fallback

**4. Modelo não carregado**
- Execute primeiro: `python main_pipeline.py --mode training`
- Verifique se existe `models/best_model_*.pkl`

**5. Interface Streamlit não abre**
```bash
pip install streamlit
streamlit run src/prediction/prediction_system.py
```

---

## 📚 Referências Técnicas

- **Aljadani et al. (2023)**: *Mathematical Modeling and Analysis of Credit Scoring Using the LIME Explainer*
- **Bielecki & Rutkowski (2001)**: *Credit Risk: Modeling, Valuation, and Hedging*
- **Crouhy, Galai, & Mark (2000)**: *A comparative analysis of current credit risk models*

---

## 👥 Autores

**Pedro Schuves Marodin**  
**Enzo Holtzmann Gaio**

📧 Para suporte ou dúvidas, consulte os logs em `logs/` ou execute com `--mode analysis` para diagnóstico completo.

---

## 🎉 Próximos Passos

1. **Dados Reais**: Substitua dados simulados por APIs reais (ANP, ANEEL, IBGE)
2. **Deployment**: Configure ambiente de produção com Docker
3. **Dashboard**: Implemente dashboard executivo com métricas de portfólio
4. **APIs**: Crie endpoints REST para integração com sistemas existentes
5. **Alertas**: Sistema de notificações para drift e performance

---

**🚀 Sistema pronto para uso! Execute `python main_pipeline.py --mode all` para começar.**
