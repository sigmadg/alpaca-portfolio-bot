"""
Cartera de Inversión de Corto Plazo (1 mes) - ALTO RIESGO
Estrategia: Alto Riesgo con Potencial de Altas Ganancias Mensuales
⚠️ ADVERTENCIA: Esta estrategia conlleva mayor riesgo de pérdidas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Intentar importar yfinance
try:
    import yfinance as yf
    DATOS_REALES = True
except ImportError:
    print("⚠️  yfinance no está instalado. Usando datos simulados.")
    DATOS_REALES = False


# ============================================================================
# FUNCIONES DE OPTIMIZACIÓN
# ============================================================================

def lagrange_quadratic(C, B, b):
    """Resuelve optimización cuadrática con restricciones lineales."""
    iC = np.linalg.pinv(C)
    M1 = B @ iC
    M2 = np.linalg.pinv(M1 @ B.T)
    lambda_star = 2 * b.T @ M2
    x_star = (lambda_star / 2) @ M1
    return x_star, lambda_star


def optimizar_cartera_rendimiento_objetivo(C, m, rendimiento_objetivo):
    """Optimiza cartera para un rendimiento objetivo (sin ventas en corto)."""
    n = len(C)
    B = np.vstack([m.reshape(1, -1), np.ones((1, n))])
    b = np.array([[rendimiento_objetivo], [1]])
    w_opt, lambda_opt = lagrange_quadratic(C, B, b)
    w_opt = w_opt.flatten()
    
    # Eliminar ventas en corto (pesos negativos) y renormalizar
    w_opt = np.maximum(w_opt, 0)
    if np.sum(w_opt) > 0:
        w_opt = w_opt / np.sum(w_opt)
    else:
        w_opt = np.ones(n) / n
    
    riesgo = np.sqrt(w_opt @ C @ w_opt.T)
    rendimiento = w_opt @ m
    return {'pesos': w_opt, 'riesgo': riesgo, 'rendimiento': rendimiento}


def optimizar_cartera_maximo_rendimiento(C, m):
    """Optimiza cartera para máximo rendimiento (alto riesgo)."""
    n = len(C)
    # Encontrar el activo con mayor rendimiento esperado
    idx_max = np.argmax(m)
    w_opt = np.zeros(n)
    w_opt[idx_max] = 1.0
    
    riesgo = np.sqrt(w_opt @ C @ w_opt.T)
    rendimiento = w_opt @ m
    return {'pesos': w_opt, 'riesgo': riesgo, 'rendimiento': rendimiento}


# ============================================================================
# FUENTE ÚNICA DE TICKERS ALTO RIESGO (análisis y ejecución Alpaca)
# ============================================================================

def get_tickers_alto_riesgo():
    """Fuente única de tickers alto riesgo (para análisis y ejecución Alpaca)."""
    tickers = {
        # NOTA: ETFs AMPLIOS y TECH BLUE CHIPS movidos a cartera_largo_plazo
        'NVDA': 'NVIDIA Corp. (Semiconductores/IA - Muy volátil)',
        'TSLA': 'Tesla Inc. (EV líder - Muy volátil)',
        'PLTR': 'Palantir Technologies (IA/Analytics - Muy volátil)',
        'SNOW': 'Snowflake Inc. (Cloud Data - Muy volátil)',
        'NET': 'Cloudflare Inc. (Cloud Security - Volátil)',
        'CRWD': 'CrowdStrike Holdings (Cybersecurity - Muy volátil)',
        'DDOG': 'Datadog Inc. (Cloud Monitoring - Muy volátil)',
        'AMD': 'Advanced Micro Devices (Muy volátil)',
        'UNH': 'UnitedHealth Group (Salud - Líder)',
        'LLY': 'Eli Lilly & Co. (Farmacéutica - Crecimiento)',
        'ABBV': 'AbbVie Inc. (Farmacéutica - Dividendo)',
        'TMO': 'Thermo Fisher Scientific (Ciencias - Estable)',
        'NEE': 'NextEra Energy (Energía Renovable - Alta volatilidad)',
        'BABA': 'Alibaba Group (E-commerce/Cloud - Líder China)',
        'JD': 'JD.com Inc. (E-commerce/Logística - Líder)',
        'PDD': 'Pinduoduo Inc. (E-commerce Social - Crecimiento)',
        'BIDU': 'Baidu Inc. (Búsqueda/IA - Google de China)',
        'NIO': 'NIO Inc. (Vehículos Eléctricos - Premium)',
        'XPEV': 'XPeng Inc. (Vehículos Eléctricos - Tech)',
        'LI': 'Li Auto Inc. (Vehículos Eléctricos - Híbrido)',
        'BYDDY': 'BYD Co. Ltd. (Vehículos Eléctricos - ADR en US; reemplazo de BYDDF OTC)',
        'RIVN': 'Rivian Automotive',
        'LCID': 'Lucid Group',
        'BILI': 'Bilibili Inc. (Streaming/Video - Generación Z)',
        'TME': 'Tencent Music Entertainment (Música Streaming)',
        'YMM': 'Full Truck Alliance (Logística Digital - Líder)',
        'BZ': 'Kanzhun Limited (Reclutamiento Digital - Líder)',
        'BGNE': 'BeiGene Ltd. (Biotech - Cáncer)',
        'ZLAB': 'Zai Lab Limited (Biotech - Farmacéutica)',
        'TAL': 'TAL Education Group (Educación - Recuperación)',
        'EDU': 'New Oriental Education (Educación - Recuperación)',
        'SHOP': 'Shopify Inc.',
        'HOOD': 'Robinhood Markets',
        'AFRM': 'Affirm Holdings',
        'UPST': 'Upstart Holdings',
        'ARKK': 'ARK Innovation ETF',
        'SOXL': 'Direxion Daily Semiconductor Bull 3X',
        'COIN': 'Coinbase Global',
        'MSTR': 'MicroStrategy',
    }
    return tickers


# ============================================================================
# FUNCIÓN PRINCIPAL: CARTERA ALTO RIESGO CORTO PLAZO
# ============================================================================

def analizar_cartera_alto_riesgo(capital_inicial=10000, riesgo_max=0.20, generar_graficos=True):
    """
    Análisis de cartera de corto plazo (1 mes) con ALTO RIESGO.
    
    ⚠️ ADVERTENCIA: Esta estrategia conlleva mayor riesgo de pérdidas significativas.
    
    Args:
        capital_inicial: Capital disponible
        riesgo_max: Riesgo máximo mensual aceptado (default 20%)
        generar_graficos: Si True, genera gráficos (default True). False para ejecución rápida.
    """
    global DATOS_REALES
    
    print("=" * 80)
    print("CARTERA DE INVERSIÓN - CORTO PLAZO (1 MES) - ALTO RIESGO")
    print("Estrategia: Acciones Chinas (ADRs) - Alto Riesgo")
    print("=" * 80)
    print("\n⚠️  ADVERTENCIA: Esta estrategia conlleva ALTO RIESGO de pérdidas.")
    print("   Solo invierte capital que puedas permitirte perder completamente.")
    print("   ⚠️  ADRs chinos pueden tener mayor volatilidad y riesgo regulatorio.")
    print("=" * 80)
    print(f"\n💰 Capital inicial: ${capital_inicial:,.2f}")
    print(f"📅 Período: 1 mes (21 días de trading)")
    print(f"⚠️  Riesgo máximo aceptado: {riesgo_max*100:.2f}% mensual")
    print(f"🎯 Objetivo: Máximo rendimiento con riesgo controlado")
    
    # ========================================================================
    # PARTE 1: SELECCIÓN DE ACTIVOS DE ALTO RIESGO
    # ========================================================================
    print("\n" + "=" * 80)
    print("PARTE 1: SELECCIÓN DE ACCIONES CHINAS (ADRs) - ALTO RIESGO")
    print("=" * 80)
    print("📌 Nota: Estas son ADRs (American Depositary Receipts)")
    print("   que representan acciones chinas cotizando en el mercado US")
    print("=" * 80)
    
    # Fuente única de tickers (mismo que Alpaca)
    tickers = get_tickers_alto_riesgo()
    ticker_symbols = list(tickers.keys())
    nombres = list(tickers.values())
    
    print("\n📊 Activos seleccionados (alto riesgo - alta volatilidad):")
    for ticker, nombre in tickers.items():
        print(f"   {ticker}: {nombre}")
    
    # ========================================================================
    # PARTE 2: OBTENER DATOS (últimos 6 meses)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PARTE 2: OBTENIENDO DATOS HISTÓRICOS")
    print("=" * 80)
    
    if DATOS_REALES:
        print("\n📊 Descargando datos históricos (6 meses)...")
        try:
            data = yf.download(ticker_symbols, period="6mo", progress=False)
            
            if 'Close' in data.columns:
                precios = data['Close']
            else:
                precios = data
            
            # Limpieza menos agresiva: conserva tickers con datos suficientes
            precios = precios.dropna(how="all")  # quita filas totalmente vacías
            precios = precios.dropna(axis=1, thresh=int(0.8 * len(precios)))  # exige >=80% datos por ticker
            precios = precios.dropna()  # ahora sí, quita filas con huecos restantes

            print(f"✅ Datos descargados: {len(precios)} días de trading")
            print(f"   Período: {precios.index[0].date()} a {precios.index[-1].date()}")
            
        except Exception as e:
            print(f"⚠️  Error descargando datos: {e}")
            print("   Usando datos simulados...")
            DATOS_REALES = False
    
    if not DATOS_REALES:
        print("\n📊 Generando datos simulados (alto riesgo)...")
        np.random.seed(42)
        n_dias = 126  # 6 meses
        fechas = pd.date_range(end=datetime.now(), periods=n_dias, freq='B')
        
        # Parámetros simulados para acciones (alta volatilidad)
        parametros = {
            # ========================================================================
            # NOTA: ETFs AMPLIOS y TECH BLUE CHIPS movidos a cartera_largo_plazo (bajo riesgo)
            # SPY, VTI, QQQ, VEA, MSFT, AAPL, GOOGL, AMZN ahora están en bajo riesgo
            # ========================================================================
            
            # ========================================================================
            # TECH US (Alta volatilidad - Solo alto riesgo)
            # ========================================================================
            'NVDA': {'mu': 0.0026, 'sigma': 0.046},  # Semiconductores/IA muy volátil
            
            # Tech/Cloud (alta volatilidad)
            'TSLA': {'mu': 0.0025, 'sigma': 0.045},  # EV líder muy volátil
            'PLTR': {'mu': 0.0023, 'sigma': 0.044},  # IA/Analytics muy volátil
            'SNOW': {'mu': 0.0022, 'sigma': 0.043},  # Cloud Data muy volátil
            'NET': {'mu': 0.0021, 'sigma': 0.042},   # Cloud Security volátil
            'CRWD': {'mu': 0.0024, 'sigma': 0.044},  # Cybersecurity muy volátil
            'DDOG': {'mu': 0.0023, 'sigma': 0.043},  # Cloud Monitoring muy volátil
            
            # ========================================================================
            # NOTA: FINANCIERAS/PAGOS ESTABLES movidas a cartera_largo_plazo (bajo riesgo)
            # V, MA, JPM ahora están en bajo riesgo
            # ========================================================================
            
            # ========================================================================
            # SEMICONDUCTORES US (Alta volatilidad - Solo alto riesgo)
            # ========================================================================
            'AMD': {'mu': 0.0024, 'sigma': 0.044},   # Semiconductores muy volátil
            # NOTA: AVGO, ASML, QCOM, TSM movidos a cartera_largo_plazo (bajo riesgo)
            
            # ========================================================================
            # SALUD/FARMACÉUTICAS US (Alta volatilidad - Solo alto riesgo)
            # ========================================================================
            'UNH': {'mu': 0.0014, 'sigma': 0.023},   # UnitedHealth - Líder
            'LLY': {'mu': 0.0018, 'sigma': 0.028},   # Eli Lilly - Crecimiento
            'ABBV': {'mu': 0.0012, 'sigma': 0.020},   # AbbVie - Dividendo
            'TMO': {'mu': 0.0013, 'sigma': 0.021},   # Thermo Fisher - Estable
            # NOTA: JNJ movido a cartera_largo_plazo (bajo riesgo)
            
            # ========================================================================
            # NOTA: CONSUMO, DIVERSIFICADO/ENERGÍA ESTABLE, DEFENSA, BONOS movidos a bajo riesgo
            # PG, KO, WMT, BRK.B, XOM, CVX, LMT, BND, TLT, GLD ahora están en bajo riesgo
            # ========================================================================
            
            # ========================================================================
            # DIVERSIFICADO/ENERGÍA US (Alta volatilidad - Solo alto riesgo)
            # ========================================================================
            'NEE': {'mu': 0.0014, 'sigma': 0.022},   # NextEra - Energía renovable (alta volatilidad)
            
            # ========================================================================
            # EMPRESAS CHINAS - E-COMMERCE/TECH (Alta volatilidad)
            # ========================================================================
            'BABA': {'mu': 0.0018, 'sigma': 0.038},  # Alibaba - E-commerce/Cloud volátil
            'JD': {'mu': 0.0015, 'sigma': 0.035},    # JD.com - E-commerce/Logística volátil
            'PDD': {'mu': 0.0021, 'sigma': 0.041},   # Pinduoduo - Social commerce muy volátil
            'BIDU': {'mu': 0.0018, 'sigma': 0.038},  # Baidu - Búsqueda/IA volátil
            
            # ========================================================================
            # EMPRESAS CHINAS - VEHÍCULOS ELÉCTRICOS (Muy alta volatilidad)
            # ========================================================================
            'NIO': {'mu': 0.002, 'sigma': 0.04},     # NIO - EV muy volátil
            'XPEV': {'mu': 0.0022, 'sigma': 0.042},  # XPeng - EV muy volátil
            'LI': {'mu': 0.0019, 'sigma': 0.039},    # Li Auto - EV volátil
            'BYDDY': {'mu': 0.002, 'sigma': 0.040},  # BYD - EV líder mundial volátil (ADR en US)
            
            # EVs (muy volátiles, alto potencial)
            'RIVN': {'mu': 0.0023, 'sigma': 0.045},  # Rivian - EV startup muy volátil
            'LCID': {'mu': 0.0021, 'sigma': 0.043},  # Lucid - EV premium muy volátil
            
            # ========================================================================
            # EMPRESAS CHINAS - ENTERTAINMENT/MEDIA (Alta volatilidad)
            # ========================================================================
            'BILI': {'mu': 0.002, 'sigma': 0.04},    # Bilibili - Streaming/Entertainment volátil
            'TME': {'mu': 0.0017, 'sigma': 0.037},   # Tencent Music - Música streaming volátil
            
            # ========================================================================
            # EMPRESAS CHINAS - TECNOLOGÍA/IA (Alta volatilidad)
            # ========================================================================
            # 'TCEHY': {'mu': 0.0019, 'sigma': 0.039},  # Tencent Holdings - OTC, no soportado en Alpaca

            # ========================================================================
            # EMPRESAS CHINAS - LOGÍSTICA/TRANSPORTE (Alta volatilidad)
            # ========================================================================
            'YMM': {'mu': 0.0018, 'sigma': 0.038},   # Full Truck Alliance - Logística digital volátil
            
            # ========================================================================
            # EMPRESAS CHINAS - RECLUTAMIENTO (Alta volatilidad)
            # ========================================================================
            'BZ': {'mu': 0.0016, 'sigma': 0.036},     # Kanzhun Limited - Reclutamiento digital volátil
            
            # ========================================================================
            # EMPRESAS CHINAS - BIOTECH (Muy alta volatilidad)
            # ========================================================================
            'BGNE': {'mu': 0.0025, 'sigma': 0.048},  # BeiGene - Biotech extremadamente volátil
            'ZLAB': {'mu': 0.0022, 'sigma': 0.043},  # Zai Lab - Biotech muy volátil
            
            # ========================================================================
            # EMPRESAS CHINAS - CONSUMO/EDUCACIÓN (Alta volatilidad)
            # ========================================================================
            'TAL': {'mu': 0.002, 'sigma': 0.041},    # TAL Education - Educación muy volátil
            'EDU': {'mu': 0.0019, 'sigma': 0.039},   # New Oriental Education - Educación volátil
            
            # E-commerce/Retail (volátil, crecimiento)
            'SHOP': {'mu': 0.0022, 'sigma': 0.043},  # Shopify - E-commerce platform muy volátil
            
            # Fintech (volátil, crecimiento)
            'HOOD': {'mu': 0.0023, 'sigma': 0.044},  # Robinhood - Trading platform muy volátil
            'AFRM': {'mu': 0.0022, 'sigma': 0.043},  # Affirm - BNPL muy volátil
            'UPST': {'mu': 0.0025, 'sigma': 0.048},  # Upstart - AI Lending extremadamente volátil
            
            # ETFs (diversificación)
            'ARKK': {'mu': 0.0023, 'sigma': 0.043},  # ARK Innovation - ETF tech disruptivo volátil
            'SOXL': {'mu': 0.003, 'sigma': 0.055},   # Direxion 3x - ETF apalancado extremo riesgo
            
            # Cripto/Bitcoin (extremo riesgo)
            'COIN': {'mu': 0.0028, 'sigma': 0.052},  # Coinbase - Crypto Exchange extremadamente volátil
            'MSTR': {'mu': 0.003, 'sigma': 0.058},   # MicroStrategy - Bitcoin Play extremadamente volátil
        }
        
        precios = pd.DataFrame(index=fechas)
        for ticker in ticker_symbols:
            params = parametros[ticker]
            rendimientos = np.random.normal(params['mu'], params['sigma'], n_dias)
            precios_ticker = [100]
            for r in rendimientos:
                precios_ticker.append(precios_ticker[-1] * (1 + r))
            precios[ticker] = precios_ticker[1:]
        
        print(f"✅ Datos simulados generados: {len(precios)} días")

    # Alinear tickers con columnas reales de precios (evita pesos vs precios desalineados)
    ticker_symbols = list(precios.columns)
    nombres = [tickers[t] for t in ticker_symbols]
    removed = set(tickers.keys()) - set(ticker_symbols)
    if removed:
        print("⚠️ removidos por falta de datos:", sorted(removed))
    
    # ========================================================================
    # PARTE 3: CALCULAR ESTADÍSTICAS MENSUALES
    # ========================================================================
    print("\n" + "=" * 80)
    print("PARTE 3: ANÁLISIS DE RENDIMIENTOS MENSUALES")
    print("=" * 80)
    
    # Calcular rendimientos diarios
    rendimientos = precios.pct_change().dropna()
    
    # Convertir a rendimientos mensuales (21 días de trading)
    rendimientos_mensuales = rendimientos.mean() * 21
    volatilidades_mensuales = rendimientos.std() * np.sqrt(21)
    
    print("\n📈 Rendimientos y Riesgos Mensuales (ALTO RIESGO):")
    print("-" * 80)
    for i, (ticker, nombre) in enumerate(zip(ticker_symbols, nombres)):
        print(f"{ticker:6s} ({nombre[:45]:45s}): "
              f"Rendimiento: {rendimientos_mensuales[ticker]*100:6.2f}% | "
              f"Riesgo: {volatilidades_mensuales[ticker]*100:6.2f}%")
    
    # Matriz de covarianza mensual
    C_mensual = rendimientos.cov().values * 21  # Mensualizada
    m_mensual = rendimientos_mensuales.values
    
    print(f"\n📊 Matriz de Correlación:")
    correlaciones = rendimientos.corr()
    print(correlaciones.round(3))
    
    # ========================================================================
    # PARTE 4: OPTIMIZACIÓN PARA ALTO RENDIMIENTO
    # ========================================================================
    print("\n" + "=" * 80)
    print("PARTE 4: OPTIMIZACIÓN DE CARTERA (ALTO RENDIMIENTO)")
    print("=" * 80)
    
    # Cartera de máximo rendimiento (más agresiva)
    resultado_max_rendimiento = optimizar_cartera_maximo_rendimiento(C_mensual, m_mensual)
    
    print("\n🎯 CARTERA DE MÁXIMO RENDIMIENTO (100% en el activo más rentable):")
    print("-" * 80)
    for i, (ticker, nombre) in enumerate(zip(ticker_symbols, nombres)):
        peso_pct = resultado_max_rendimiento['pesos'][i] * 100
        inversion = capital_inicial * resultado_max_rendimiento['pesos'][i]
        if abs(peso_pct) > 0.01:
            print(f"  {ticker:6s}: {peso_pct:6.2f}% (${inversion:,.2f})")
    print(f"\n  ⚠️  Riesgo mensual: {resultado_max_rendimiento['riesgo']*100:.2f}%")
    print(f"  🎯 Rendimiento esperado mensual: {resultado_max_rendimiento['rendimiento']*100:.2f}%")
    print(f"  💰 Ganancia esperada (1 mes): ${capital_inicial * resultado_max_rendimiento['rendimiento']:,.2f}")
    
    # Optimizar con restricción de riesgo máximo pero buscando alto rendimiento
    rendimiento_objetivo = np.mean(m_mensual) * 1.2  # 120% del promedio para alto rendimiento
    max_weight_ticker = 0.12  # Peso máximo por ticker
    cap_soxl = 0.10  # Cap SOXL
    cap_cripto_conjunto = 0.10  # Cap conjunto MSTR + COIN

    print(f"\n🎯 CARTERA OPTIMIZADA (Riesgo máximo: {riesgo_max*100:.2f}%):")
    print(f"   Rendimiento objetivo inicial: {rendimiento_objetivo*100:.2f}% mensual")
    print("-" * 80)

    def aplicar_caps_y_max_weight(pesos_arr):
        """Aplica max_weight por ticker, cap SOXL y cap conjunto MSTR+COIN."""
        w = np.array(pesos_arr, dtype=float)
        # Max weight por ticker
        w = np.minimum(w, max_weight_ticker)
        if np.sum(w) > 0:
            w = w / np.sum(w)
        # Cap SOXL
        if 'SOXL' in ticker_symbols:
            idx_soxl = ticker_symbols.index('SOXL')
            if w[idx_soxl] > cap_soxl:
                exceso = w[idx_soxl] - cap_soxl
                w[idx_soxl] = cap_soxl
                otros = np.ones_like(w, dtype=bool)
                otros[idx_soxl] = False
                if np.sum(w[otros]) > 0:
                    w[otros] += exceso * (w[otros] / np.sum(w[otros]))
                w = w / np.sum(w)
        # Cap conjunto MSTR + COIN (máximo 10% entre ambos)
        idx_mstr = ticker_symbols.index('MSTR') if 'MSTR' in ticker_symbols else None
        idx_coin = ticker_symbols.index('COIN') if 'COIN' in ticker_symbols else None
        if idx_mstr is not None or idx_coin is not None:
            suma_cripto = (w[idx_mstr] if idx_mstr is not None else 0) + (w[idx_coin] if idx_coin is not None else 0)
            if suma_cripto > cap_cripto_conjunto and suma_cripto > 0:
                factor = cap_cripto_conjunto / suma_cripto
                if idx_mstr is not None:
                    w[idx_mstr] *= factor
                if idx_coin is not None:
                    w[idx_coin] *= factor
                otros = np.ones_like(w, dtype=bool)
                if idx_mstr is not None:
                    otros[idx_mstr] = False
                if idx_coin is not None:
                    otros[idx_coin] = False
                exceso = suma_cripto - cap_cripto_conjunto
                if np.sum(w[otros]) > 0:
                    w[otros] += exceso * (w[otros] / np.sum(w[otros]))
                w = w / np.sum(w)
        return w

    try:
        resultado_optimizado = optimizar_cartera_rendimiento_objetivo(C_mensual, m_mensual, rendimiento_objetivo)
        # Loop: ajustar rendimiento objetivo hasta que riesgo <= riesgo_max
        max_iter = 20
        for _ in range(max_iter):
            if resultado_optimizado['riesgo'] <= riesgo_max:
                break
            rendimiento_objetivo *= 0.85  # Reducir ~15%
            if rendimiento_objetivo < np.min(m_mensual):
                rendimiento_objetivo = np.mean(m_mensual) * 0.5
            resultado_optimizado = optimizar_cartera_rendimiento_objetivo(C_mensual, m_mensual, rendimiento_objetivo)
        if resultado_optimizado['riesgo'] > riesgo_max:
            print(f"⚠️  Riesgo ({resultado_optimizado['riesgo']*100:.2f}%) aún excede máximo ({riesgo_max*100:.2f}%); usando cartera mínimo riesgo implícito.")
        # Aplicar caps y max_weight
        pesos_capped = aplicar_caps_y_max_weight(resultado_optimizado['pesos'])
        resultado_optimizado['pesos'] = pesos_capped
        resultado_optimizado['riesgo'] = np.sqrt(pesos_capped @ C_mensual @ pesos_capped.T)
        resultado_optimizado['rendimiento'] = pesos_capped @ m_mensual

        for i, (ticker, nombre) in enumerate(zip(ticker_symbols, nombres)):
            peso_pct = resultado_optimizado['pesos'][i] * 100
            inversion = capital_inicial * resultado_optimizado['pesos'][i]
            # Mostrar todos los tickers con peso > 0.001 (0.1%) para mejor visibilidad
            if abs(peso_pct) > 0.01:
                print(f"  {ticker:6s}: {peso_pct:6.2f}% (${inversion:,.2f})")
            elif abs(peso_pct) > 0.001:
                print(f"  {ticker:6s}: {peso_pct:6.3f}% (${inversion:,.2f}) [peso mínimo]")
        
        print(f"\n  ⚠️  Riesgo mensual: {resultado_optimizado['riesgo']*100:.2f}%")
        print(f"  🎯 Rendimiento esperado mensual: {resultado_optimizado['rendimiento']*100:.2f}%")
        print(f"  💰 Ganancia esperada (1 mes): ${capital_inicial * resultado_optimizado['rendimiento']:,.2f}")
        print(f"  📈 Retorno anualizado estimado: {resultado_optimizado['rendimiento']*12*100:.2f}%")
        
    except Exception as e:
        print(f"⚠️  Error en optimización: {e}")
        resultado_optimizado = resultado_max_rendimiento
    
    # ========================================================================
    # PARTE 5: SIMULACIÓN MENSUAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("PARTE 5: SIMULACIÓN DE ESCENARIOS (1 MES) - ALTO RIESGO")
    print("=" * 80)
    
    def simular_precios_mensuales(precio_actual, mu_mensual, sigma_mensual, num_sim=1000):
        """Simula precios a 1 mes (≈21 días hábiles) usando GBM con parámetros mensuales."""
        dias_mes = 21

        # Convertir parámetros mensuales -> diarios (asumiendo 21 días hábiles por mes)
        mu_diario = mu_mensual / dias_mes
        sigma_diario = sigma_mensual / np.sqrt(dias_mes)

        precios_sim = np.empty(num_sim, dtype=float)

        for i in range(num_sim):
            precio = float(precio_actual)
            for _ in range(dias_mes):
                z = np.random.normal()
                # dt = 1 día
                precio *= np.exp((mu_diario - 0.5 * sigma_diario**2) + sigma_diario * z)
            precios_sim[i] = precio

        return precios_sim
    
    print("\n🔮 Simulando 1000 escenarios para 1 mes (ALTO RIESGO)...")
    
    precios_actuales = precios.iloc[-1].values
    simulaciones = {}
    
    for i, ticker in enumerate(ticker_symbols):
        mu_mensual = rendimientos_mensuales[ticker]
        sigma_mensual = volatilidades_mensuales[ticker]
        sim = simular_precios_mensuales(precios_actuales[i], mu_mensual, sigma_mensual)
        simulaciones[ticker] = sim
    
    # Calcular valor futuro de la cartera
    pesos_optimos = resultado_optimizado['pesos']
    valores_futuros_cartera = np.zeros(1000)
    
    for i, ticker in enumerate(ticker_symbols):
        num_acciones = (capital_inicial * pesos_optimos[i]) / precios_actuales[i]
        valores_futuros_cartera += num_acciones * simulaciones[ticker]
    
    valor_futuro_promedio = np.mean(valores_futuros_cartera)
    valor_futuro_std = np.std(valores_futuros_cartera)
    ganancia_promedio = valor_futuro_promedio - capital_inicial
    
    # Calcular percentiles para análisis de riesgo
    percentil_5 = np.percentile(valores_futuros_cartera, 5)
    percentil_95 = np.percentile(valores_futuros_cartera, 95)
    
    print(f"\n📊 Resultados de la Simulación (1 mes) - ALTO RIESGO:")
    print(f"   Valor actual: ${capital_inicial:,.2f}")
    print(f"   Valor esperado (1 mes): ${valor_futuro_promedio:,.2f}")
    print(f"   Ganancia esperada: ${ganancia_promedio:,.2f}")
    print(f"   Rendimiento esperado: {(ganancia_promedio/capital_inicial)*100:.2f}%")
    print(f"   ⚠️  Intervalo de confianza 95%: "
          f"${valor_futuro_promedio - 1.96*valor_futuro_std:,.2f} - "
          f"${valor_futuro_promedio + 1.96*valor_futuro_std:,.2f}")
    print(f"   📉 Peor escenario (5%): ${percentil_5:,.2f} (Pérdida: ${percentil_5 - capital_inicial:,.2f})")
    print(f"   📈 Mejor escenario (95%): ${percentil_95:,.2f} (Ganancia: ${percentil_95 - capital_inicial:,.2f})")
    print(f"   🎯 Probabilidad de ganancia: "
          f"{(valores_futuros_cartera > capital_inicial).sum() / len(valores_futuros_cartera)*100:.1f}%")
    print(f"   ⚠️  Probabilidad de pérdida: "
          f"{(valores_futuros_cartera < capital_inicial).sum() / len(valores_futuros_cartera)*100:.1f}%")
    print(f"   💥 Probabilidad de pérdida > 20%: "
          f"{(valores_futuros_cartera < capital_inicial * 0.8).sum() / len(valores_futuros_cartera)*100:.1f}%")
    
    # ========================================================================
    # PARTE 6: VISUALIZACIONES (Opcional)
    # ========================================================================
    if generar_graficos:
        print("\n" + "=" * 80)
        print("PARTE 6: GENERANDO VISUALIZACIONES")
        print("=" * 80)
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Distribución de pesos
        ax1 = plt.subplot(2, 3, 1)
        pesos_pct = resultado_optimizado['pesos'] * 100
        colores = plt.cm.Reds(range(len(ticker_symbols)))
        bars = ax1.barh(ticker_symbols, pesos_pct, color=colores)
        ax1.set_xlabel('Peso (%)', fontweight='bold')
        ax1.set_title('Distribución de la Cartera\n(ALTO RIESGO)', fontweight='bold', color='red')
        ax1.grid(True, alpha=0.3, axis='x')
        for i, (bar, peso) in enumerate(zip(bars, pesos_pct)):
            if abs(peso) > 0.01:
                ax1.text(peso, i, f'{peso:.1f}%', va='center', fontweight='bold')
        
        # 2. Riesgo vs Rendimiento (Mensual)
        ax2 = plt.subplot(2, 3, 2)
        ax2.scatter(volatilidades_mensuales*100, rendimientos_mensuales*100, 
                   s=200, alpha=0.6, color='red')
        for i, ticker in enumerate(ticker_symbols):
            ax2.annotate(ticker, (volatilidades_mensuales[i]*100, rendimientos_mensuales[i]*100),
                        fontsize=10, fontweight='bold')
        ax2.scatter(resultado_optimizado['riesgo']*100, 
                   resultado_optimizado['rendimiento']*100,
                   s=400, color='darkred', marker='*', label='Cartera Óptima', zorder=5)
        ax2.set_xlabel('Riesgo Mensual (%)', fontweight='bold')
        ax2.set_ylabel('Rendimiento Mensual (%)', fontweight='bold')
        ax2.set_title('Riesgo vs. Rendimiento (ALTO RIESGO)', fontweight='bold', color='red')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribución de valores futuros
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(valores_futuros_cartera, bins=50, alpha=0.7, edgecolor='black', color='crimson')
        ax3.axvline(valor_futuro_promedio, color='red', linestyle='--', 
                    linewidth=2, label=f'Promedio: ${valor_futuro_promedio:,.0f}')
        ax3.axvline(capital_inicial, color='blue', linestyle='--', 
                    linewidth=2, label=f'Inicial: ${capital_inicial:,.0f}')
        ax3.axvline(percentil_5, color='darkred', linestyle=':', 
                    linewidth=2, label=f'Peor 5%: ${percentil_5:,.0f}')
        ax3.set_xlabel('Valor de la Cartera ($)', fontweight='bold')
        ax3.set_ylabel('Frecuencia', fontweight='bold')
        ax3.set_title('Distribución de Valores (1 Mes) - ALTO RIESGO', fontweight='bold', color='red')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Evolución de precios (últimos 3 meses)
        ax4 = plt.subplot(2, 3, 4)
        precios_recientes = precios.tail(63)  # Últimos 3 meses
        for ticker in ticker_symbols:
            precios_norm = (precios_recientes[ticker] / precios_recientes[ticker].iloc[0]) * 100
            ax4.plot(precios_recientes.index, precios_norm, label=ticker, linewidth=1.5)
        ax4.set_title('Evolución Reciente (3 meses)', fontweight='bold')
        ax4.set_xlabel('Fecha')
        ax4.set_ylabel('Precio Normalizado (Base=100)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Matriz de correlación
        ax5 = plt.subplot(2, 3, 5)
        im = ax5.imshow(correlaciones.values, cmap='Reds', vmin=-1, vmax=1, aspect='auto')
        ax5.set_xticks(range(len(ticker_symbols)))
        ax5.set_yticks(range(len(ticker_symbols)))
        ax5.set_xticklabels(ticker_symbols)
        ax5.set_yticklabels(ticker_symbols)
        ax5.set_title('Matriz de Correlación', fontweight='bold')
        for i in range(len(ticker_symbols)):
            for j in range(len(ticker_symbols)):
                ax5.text(j, i, f'{correlaciones.iloc[i, j]:.2f}',
                        ha='center', va='center', fontweight='bold', fontsize=9)
        plt.colorbar(im, ax=ax5)
        
        # 6. Resumen de inversión
        ax6 = plt.subplot(2, 3, 6)
        inversiones = capital_inicial * resultado_optimizado['pesos']
        ax6.pie(inversiones, labels=ticker_symbols, autopct='%1.1f%%', 
               startangle=90, colors=colores)
        ax6.set_title(f'Distribución de Inversión\n${capital_inicial:,.0f} - ALTO RIESGO', 
                     fontweight='bold', color='red')
        
        plt.tight_layout()
        
        # Guardar gráfico
        nombre_archivo = f'cartera_alto_riesgo_{datetime.now().strftime("%Y%m%d")}.png'
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        plt.close()  # Cerrar figura para liberar memoria
        print(f"\n✅ Visualizaciones guardadas en: {nombre_archivo}")
    else:
        print("\n⏩ Saltando generación de gráficos (modo rápido)")
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESUMEN FINAL - CARTERA ALTO RIESGO CORTO PLAZO")
    print("=" * 80)
    
    print(f"\n💰 DISTRIBUCIÓN DE INVERSIÓN:")
    print("-" * 80)
    total_invertido = 0
    for i, (ticker, nombre) in enumerate(zip(ticker_symbols, nombres)):
        inversion = capital_inicial * resultado_optimizado['pesos'][i]
        if abs(inversion) > 1:
            num_acciones = inversion / precios_actuales[i]
            print(f"  {ticker:6s}: ${inversion:>10,.2f} → {num_acciones:.2f} acciones @ ${precios_actuales[i]:.2f}")
            total_invertido += abs(inversion)
    
    print(f"\n  Total invertido: ${total_invertido:,.2f}")
    print(f"  Efectivo restante: ${capital_inicial - total_invertido:,.2f}")
    
    print(f"\n📊 MÉTRICAS MENSUALES (ALTO RIESGO):")
    print("-" * 80)
    print(f"  🎯 Rendimiento esperado: {resultado_optimizado['rendimiento']*100:.2f}%")
    print(f"  ⚠️  Riesgo (volatilidad): {resultado_optimizado['riesgo']*100:.2f}%")
    print(f"  💰 Ganancia esperada: ${capital_inicial * resultado_optimizado['rendimiento']:,.2f}")
    print(f"  📈 Ratio Sharpe mensual: "
          f"{(resultado_optimizado['rendimiento'] - 0.0025) / resultado_optimizado['riesgo']:.2f}")
    print(f"  (Asumiendo tasa libre de riesgo mensual: 0.25%)")
    
    print(f"\n🔮 PROYECCIÓN A 1 MES (ALTO RIESGO):")
    print("-" * 80)
    print(f"  Valor esperado: ${valor_futuro_promedio:,.2f}")
    print(f"  Ganancia esperada: ${ganancia_promedio:,.2f}")
    print(f"  🎯 Probabilidad de ganancia: "
          f"{(valores_futuros_cartera > capital_inicial).sum() / len(valores_futuros_cartera)*100:.1f}%")
    print(f"  ⚠️  Probabilidad de pérdida: "
          f"{(valores_futuros_cartera < capital_inicial).sum() / len(valores_futuros_cartera)*100:.1f}%")
    print(f"  💥 Probabilidad de pérdida > 20%: "
          f"{(valores_futuros_cartera < capital_inicial * 0.8).sum() / len(valores_futuros_cartera)*100:.1f}%")
    print(f"  📉 Peor caso (5%): ${percentil_5:,.2f}")
    print(f"  📈 Mejor caso (95%): ${percentil_95:,.2f}")
    
    print(f"\n⚠️  ADVERTENCIAS IMPORTANTES:")
    print("-" * 80)
    print("  1. Esta es una estrategia de ALTO RIESGO")
    print("  2. Puedes perder una parte significativa de tu capital")
    print("  3. Solo invierte dinero que puedas permitirte perder")
    print("  4. Monitorea diariamente - los movimientos pueden ser extremos")
    print("  5. Ten un plan de salida claro (stop-loss)")
    print("  6. Considera diversificar incluso dentro de alto riesgo")
    print("  7. Los costos de transacción pueden ser significativos")
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO")
    print("=" * 80)
    
    return {
        'precios': precios,
        'rendimientos_mensuales': rendimientos_mensuales,
        'rendimientos': rendimientos,
        'cartera_optimizada': resultado_optimizado,
        'simulaciones': valores_futuros_cartera,
        'ganancia_esperada': ganancia_promedio,
        'percentil_5': percentil_5,
        'percentil_95': percentil_95
    }


# ============================================================================
# EJECUTAR ANÁLISIS
# ============================================================================

if __name__ == "__main__":
    # Configuración
    CAPITAL = 10000  # Capital disponible
    RIESGO_MAX = 0.20  # Riesgo máximo mensual (20% = 0.20) - ALTO RIESGO
    
    print("\n" + "⚠️" * 40)
    print("ADVERTENCIA: ESTRATEGIA DE ALTO RIESGO")
    print("⚠️" * 40)
    print("\nEsta estrategia conlleva un riesgo significativo de pérdidas.")
    print("Solo invierte capital que puedas permitirte perder completamente.")
    print("\n¿Deseas continuar? (El script se ejecutará automáticamente)")
    print("=" * 80 + "\n")
    
    # Ejecutar análisis
    resultados = analizar_cartera_alto_riesgo(
        capital_inicial=CAPITAL,
        riesgo_max=RIESGO_MAX
    )
    
    print("\n✅ Análisis de cartera alto riesgo completado!")
    print("⚠️  Recuerda: Alto riesgo = Alto potencial de pérdidas")

