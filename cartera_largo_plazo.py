"""
Cartera de Inversión de Largo Plazo (1 año o más)
Estrategia: Minimización de Riesgo y Maximización de Sharpe Ratio
Basado en Teoría de Markowitz y CAPM (Tema 9 y Tema 10)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

GRAFICOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_graficos')


def _guardar_grafica(fig, slug, stamp):
    os.makedirs(GRAFICOS_DIR, exist_ok=True)
    path = os.path.join(GRAFICOS_DIR, f'{slug}_{stamp}.png')
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


def get_tickers_largo_plazo():
    """Universo CORE / largo plazo (no overlap operativo con alto riesgo)."""
    return {
        'SPY', 'VTI', 'QQQ', 'VEA',
        'AAPL', 'MSFT', 'GOOGL', 'AMZN',
        'BND', 'TLT', 'GLD',
        'JNJ', 'PG', 'KO', 'WMT', 'VZ', 'LMT',
        'V', 'MA', 'JPM',
        'AVGO', 'ASML', 'QCOM', 'TSM',
        'BRK.B', 'XOM', 'CVX',
    }

# Intentar importar yfinance
try:
    import yfinance as yf
    DATOS_REALES = True
except ImportError:
    print("⚠️  yfinance no está instalado. Usando datos simulados.")
    DATOS_REALES = False


# ============================================================================
# FUNCIONES DE OPTIMIZACIÓN (Basadas en Teoría)
# ============================================================================

def lagrange_quadratic(C, B, b):
    """
    Resuelve optimización cuadrática con restricciones lineales.
    Basado en Tema 9: Minimización de riesgo con restricciones.
    
    Args:
        C: Matriz de covarianzas (n x n)
        B: Matriz de restricciones (k x n)
        b: Vector de restricciones (k x 1)
    
    Returns:
        x_star: Vector de pesos óptimos
        lambda_star: Multiplicadores de Lagrange
    """
    iC = np.linalg.pinv(C)
    M1 = B @ iC
    M2 = np.linalg.pinv(M1 @ B.T)
    lambda_star = 2 * b.T @ M2
    x_star = (lambda_star / 2) @ M1
    return x_star, lambda_star


def optimizar_cartera_minimo_riesgo(C, m=None):
    """
    Optimiza cartera para mínimo riesgo (Tema 9).
    
    Problema:
    min w*C*w^T
    s.a. 1*w^T = 1
    
    Solución:
    w* = (1*C^-1) / (1*C^-1*1^T)
    
    Args:
        C: Matriz de covarianzas
        m: Vector de rendimientos esperados (opcional)
    
    Returns:
        dict: Pesos óptimos, riesgo mínimo, rendimiento esperado
    """
    n = len(C)
    v1 = np.ones(n)
    iC = np.linalg.pinv(C)

    # Calcular pesos óptimos según teoría
    denominador = v1 @ iC @ v1.T
    w_opt = (v1 @ iC) / denominador
    w_opt = w_opt.flatten()
    
    # Calcular riesgo mínimo
    riesgo_min = np.sqrt(w_opt @ C @ w_opt.T)
    
    resultado = {'pesos': w_opt, 'riesgo': riesgo_min}
    
    if m is not None:
        rendimiento = w_opt @ m
        resultado['rendimiento'] = rendimiento
    
    return resultado


def optimizar_cartera_maximo_sharpe(C, m, tasa_libre_riesgo=0.02):
    """
    Optimiza cartera para máximo Sharpe Ratio (Tema 10 - CAPM).
    
    Problema:
    max S = (m*w^T - μ_rf) / sqrt(w*C*w^T)
    s.a. 1*w^T = 1
    
    Solución:
    w* = (m - μ_rf*1) * C^-1 / ((m - μ_rf*1) * C^-1 * 1^T)
    
    Args:
        C: Matriz de covarianzas
        m: Vector de rendimientos esperados
        tasa_libre_riesgo: Tasa libre de riesgo anual (default 2%)
    
    Returns:
        dict: Pesos óptimos, Sharpe Ratio, riesgo, rendimiento
    """
    n = len(m)
    v1 = np.ones(n)
    iC = np.linalg.pinv(C)

    # Calcular numerador: (m - μ_rf*1) * C^-1
    numerador = (m - tasa_libre_riesgo * v1) @ iC
    
    # Calcular denominador: (m - μ_rf*1) * C^-1 * 1^T
    denominador = numerador @ v1.T
    
    # Pesos óptimos
    w_opt = numerador / denominador
    w_opt = w_opt.flatten()
    
    # Eliminar ventas en corto y renormalizar
    w_opt = np.maximum(w_opt, 0)
    if np.sum(w_opt) > 0:
        w_opt = w_opt / np.sum(w_opt)
    else:
        w_opt = np.ones(n) / n
    
    # Calcular métricas
    rendimiento = w_opt @ m
    riesgo = np.sqrt(w_opt @ C @ w_opt.T)
    sharpe_ratio = (rendimiento - tasa_libre_riesgo) / riesgo if riesgo > 0 else 0
    
    return {
        'pesos': w_opt,
        'sharpe_ratio': sharpe_ratio,
        'riesgo': riesgo,
        'rendimiento': rendimiento
    }


def optimizar_cartera_rendimiento_objetivo(C, m, rendimiento_objetivo):
    """
    Optimiza cartera para un rendimiento objetivo con mínimo riesgo.
    
    Problema:
    min w*C*w^T
    s.a. m*w^T = μ
         1*w^T = 1
    
    Args:
        C: Matriz de covarianzas
        m: Vector de rendimientos esperados
        rendimiento_objetivo: Rendimiento objetivo
    
    Returns:
        dict: Pesos óptimos, riesgo, rendimiento
    """
    n = len(C)
    B = np.vstack([m.reshape(1, -1), np.ones((1, n))])
    b = np.array([[rendimiento_objetivo], [1]])
    
    w_opt, lambda_opt = lagrange_quadratic(C, B, b)
    w_opt = w_opt.flatten()
    
    # Eliminar ventas en corto y renormalizar
    w_opt = np.maximum(w_opt, 0)
    if np.sum(w_opt) > 0:
        w_opt = w_opt / np.sum(w_opt)
    else:
        w_opt = np.ones(n) / n
    
    riesgo = np.sqrt(w_opt @ C @ w_opt.T)
    rendimiento = w_opt @ m
    
    return {'pesos': w_opt, 'riesgo': riesgo, 'rendimiento': rendimiento}


# ============================================================================
# FUNCIÓN PRINCIPAL: CARTERA LARGO PLAZO
# ============================================================================

def _shrinkage_covariance(C, alpha=0.85):
    """Estabiliza covarianza: C_shrink = alpha*C + (1-alpha)*diag(C)."""
    if alpha >= 1.0:
        return C
    d = np.diag(C).copy()
    return alpha * C + (1 - alpha) * np.diag(d)


def _aplicar_limites_peso(w, min_weight=0.0, max_weight=0.15):
    """Aplica límites por activo y renormaliza."""
    w = np.clip(w, min_weight, max_weight)
    if np.sum(w) > 0:
        w = w / np.sum(w)
    return w


def analizar_cartera_largo_plazo(capital_inicial=10000, riesgo_max=0.15, 
                                  estrategia='sharpe', generar_graficos=True,
                                  max_weight=0.15, min_weight=0.0, shrinkage_alpha=0.85):
    """
    Análisis de cartera de largo plazo (1 año o más) con optimización teórica.
    
    Basado en:
    - Tema 9: Minimización de riesgo
    - Tema 10: Maximización de Sharpe Ratio (CAPM)
    
    Args:
        capital_inicial: Capital disponible
        riesgo_max: Riesgo máximo anual aceptado (default 15%)
        estrategia: 'minimo_riesgo', 'sharpe', o 'rendimiento_objetivo'
        generar_graficos: Si True, genera gráficos
        max_weight: Peso máximo por activo (default 0.15)
        min_weight: Peso mínimo por activo (default 0.0)
        shrinkage_alpha: Alpha para shrinkage de covarianza (default 0.85)
    
    Returns:
        dict: Resultados de la optimización
    """
    global DATOS_REALES
    
    print("=" * 80)
    print("CARTERA DE INVERSIÓN - LARGO PLAZO (1 AÑO O MÁS)")
    print("Estrategia: Optimización Teórica (Markowitz + CAPM)")
    print("=" * 80)
    print(f"\n💰 Capital inicial: ${capital_inicial:,.2f}")
    print(f"📅 Período: Largo plazo (1 año o más)")
    print(f"⚠️  Riesgo máximo aceptado: {riesgo_max*100:.2f}% anual")
    print(f"📊 Estrategia: {estrategia}")
    
    # ========================================================================
    # PARTE 1: SELECCIÓN DE ACTIVOS PARA LARGO PLAZO
    # ========================================================================
    print("\n" + "=" * 80)
    print("PARTE 1: SELECCIÓN DE ACTIVOS PARA LARGO PLAZO")
    print("=" * 80)
    
    # Activos estables y diversificados para largo plazo (SOLO BAJO RIESGO)
    tickers = {
        # ========================================================================
        # ETFs AMPLIOS (Diversificación - Baja volatilidad)
        # ========================================================================
        'SPY': 'SPDR S&P 500 ETF (Índice amplio - Bajo riesgo)',
        'VTI': 'Vanguard Total Stock Market (Diversificación total)',
        'QQQ': 'Invesco QQQ Trust (Nasdaq 100 - Tech estable)',
        'VEA': 'Vanguard FTSE Developed Markets (Internacional)',
        
        # ========================================================================
        # BLUE CHIPS TECH (Empresas consolidadas - Estables)
        # ========================================================================
        'AAPL': 'Apple Inc. (Tech líder - Estable)',
        'MSFT': 'Microsoft Corp. (Tech líder - Estable)',
        'GOOGL': 'Alphabet Inc. (Tech líder - Estable)',
        'AMZN': 'Amazon.com Inc. (E-commerce líder)',
        
        # ========================================================================
        # BONOS Y ACTIVOS DEFENSIVOS (Muy baja volatilidad)
        # ========================================================================
        'BND': 'Vanguard Total Bond Market (Bonos - Bajo riesgo)',
        'TLT': 'iShares 20+ Year Treasury Bond (Bonos gubernamentales)',
        'GLD': 'SPDR Gold Trust (Oro - Hedge inflación)',
        
        # ========================================================================
        # SALUD/FARMACÉUTICAS (Defensivos - Baja volatilidad)
        # ========================================================================
        'JNJ': 'Johnson & Johnson (Salud - Defensivo)',
        
        # ========================================================================
        # CONSUMO (Defensivos - Baja volatilidad)
        # ========================================================================
        'PG': 'Procter & Gamble (Consumo - Defensivo)',
        'KO': 'Coca-Cola (Dividendo estable)',
        'WMT': 'Walmart (Retail líder - Defensivo)',
        
        # ========================================================================
        # TELECOMUNICACIONES (Dividendos - Baja volatilidad)
        # ========================================================================
        'VZ': 'Verizon Communications (Telecom - Dividendo)',
        
        # ========================================================================
        # DEFENSA (Estable - Baja volatilidad)
        # ========================================================================
        'LMT': 'Lockheed Martin (Defensa estable)',
        
        # ========================================================================
        # FINANCIERAS ESTABLES (Moderada-baja volatilidad)
        # ========================================================================
        'V': 'Visa (Pagos - Estable)',
        'MA': 'Mastercard (Pagos - Estable)',
        'JPM': 'JPMorgan (Banca líder - Estable)',
        
        # ========================================================================
        # SEMICONDUCTORES ESTABLES (Moderada volatilidad)
        # ========================================================================
        'AVGO': 'Broadcom (Semiconductores - Estable)',
        'ASML': 'ASML (Líder mundial - Estable)',
        'QCOM': 'Qualcomm (Semiconductores móviles - Estable)',
        'TSM': 'Taiwan Semiconductor (Líder - Estable)',
        
        # ========================================================================
        # DIVERSIFICADO/ENERGÍA ESTABLE (Moderada volatilidad)
        # ========================================================================
        'BRK.B': 'Berkshire Hathaway (Diversificado - Estable)',
        'XOM': 'Exxon Mobil (Petróleo estable)',
        'CVX': 'Chevron (Petróleo dividendo)',
    }
    
    ticker_symbols = list(tickers.keys())
    nombres = list(tickers.values())
    
    print("\n📊 Activos seleccionados (largo plazo - bajo riesgo):")
    for ticker, nombre in tickers.items():
        print(f"   {ticker}: {nombre}")
    
    # ========================================================================
    # PARTE 2: OBTENER DATOS HISTÓRICOS (1 año para mejor estimación)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PARTE 2: OBTENIENDO DATOS HISTÓRICOS")
    print("=" * 80)
    
    if DATOS_REALES:
        print("\n📊 Descargando datos históricos (1 año)...")
        try:
            data = yf.download(ticker_symbols, period="1y", progress=False)
            
            if 'Close' in data.columns:
                precios = data['Close']
            else:
                precios = data
            
            # Limpieza menos agresiva: conserva tickers con datos suficientes
            precios = precios.dropna(how="all")
            precios = precios.dropna(axis=1, thresh=int(0.8 * len(precios)))
            precios = precios.dropna()

            print(f"✅ Datos descargados: {len(precios)} días de trading")
            print(f"   Período: {precios.index[0].date()} a {precios.index[-1].date()}")
            
        except Exception as e:
            print(f"⚠️  Error descargando datos: {e}")
            print("   Usando datos simulados...")
            DATOS_REALES = False
    
    if not DATOS_REALES:
        print("\n📊 Generando datos simulados (largo plazo - bajo riesgo)...")
        np.random.seed(42)
        n_dias = 252  # 1 año de trading
        fechas = pd.date_range(end=datetime.now(), periods=n_dias, freq='B')
        
        # Parámetros simulados para activos estables (baja volatilidad)
        parametros = {
            'SPY': {'mu': 0.0008, 'sigma': 0.012},  # ETF S&P 500
            'VTI': {'mu': 0.0007, 'sigma': 0.011},  # Total Stock Market
            'QQQ': {'mu': 0.001, 'sigma': 0.014},   # Nasdaq 100
            'VEA': {'mu': 0.0006, 'sigma': 0.013},  # Internacional
            'AAPL': {'mu': 0.0009, 'sigma': 0.015}, # Apple
            'MSFT': {'mu': 0.0008, 'sigma': 0.014}, # Microsoft
            'GOOGL': {'mu': 0.0009, 'sigma': 0.016}, # Google
            'AMZN': {'mu': 0.001, 'sigma': 0.017},  # Amazon
            'BND': {'mu': 0.0003, 'sigma': 0.005},  # Bonos
            'TLT': {'mu': 0.0004, 'sigma': 0.008},  # Bonos largo plazo
            'GLD': {'mu': 0.0005, 'sigma': 0.010},  # Oro
            'JNJ': {'mu': 0.0006, 'sigma': 0.011},  # Johnson & Johnson
            'PG': {'mu': 0.0005, 'sigma': 0.010},   # Procter & Gamble
            'VZ': {'mu': 0.0004, 'sigma': 0.009},   # Verizon
            'KO': {'mu': 0.0005, 'sigma': 0.009},   # Coca-Cola (consumo defensivo)
            'WMT': {'mu': 0.0006, 'sigma': 0.010},   # Walmart (retail estable)
            'MA': {'mu': 0.0007, 'sigma': 0.012},   # Mastercard (pagos)
            'JPM': {'mu': 0.0007, 'sigma': 0.013},  # JPMorgan (banca)
            'AVGO': {'mu': 0.0008, 'sigma': 0.014}, # Broadcom (semis estable)
            'ASML': {'mu': 0.0009, 'sigma': 0.015}, # ASML (semis líder)
            'QCOM': {'mu': 0.0008, 'sigma': 0.014},  # Qualcomm (semis móviles)
            'TSM': {'mu': 0.0008, 'sigma': 0.015},  # Taiwan Semiconductor
            'BRK.B': {'mu': 0.0007, 'sigma': 0.012}, # Berkshire (diversificado)
            'XOM': {'mu': 0.0006, 'sigma': 0.012},  # Exxon (energía)
            'CVX': {'mu': 0.0006, 'sigma': 0.011},   # Chevron (energía dividendo)
            'V': {'mu': 0.0007, 'sigma': 0.012},    # Visa (pagos)
            'LMT': {'mu': 0.0005, 'sigma': 0.010},   # Lockheed (defensa)
        }
        
        # Generar precios simulados
        precios = pd.DataFrame(index=fechas)
        precios_iniciales = {t: 100.0 for t in ticker_symbols}
        
        for ticker in ticker_symbols:
            params = parametros.get(ticker, {'mu': 0.0007, 'sigma': 0.012})
            retornos = np.random.normal(params['mu'], params['sigma'], n_dias)
            precios_ticker = [precios_iniciales[ticker]]
            for ret in retornos:
                precios_ticker.append(precios_ticker[-1] * (1 + ret))
            precios[ticker] = precios_ticker[1:]

    # Alinear tickers con columnas reales de precios (evita pesos vs precios desalineados)
    ticker_symbols = list(precios.columns)
    nombres = [tickers[t] for t in ticker_symbols]
    removed = set(tickers.keys()) - set(ticker_symbols)
    if removed:
        print("⚠️ removidos por falta de datos:", sorted(removed))
    
    # ========================================================================
    # PARTE 3: CALCULAR RENDIMIENTOS Y COVARIANZAS
    # ========================================================================
    print("\n" + "=" * 80)
    print("PARTE 3: CALCULANDO RENDIMIENTOS Y COVARIANZAS")
    print("=" * 80)
    
    # Calcular rendimientos diarios
    rendimientos = precios.pct_change().dropna()
    
    # Calcular rendimientos esperados (media anualizada)
    rendimientos_esperados = rendimientos.mean() * 252  # Anualizar
    
    # Calcular matriz de covarianzas (anualizada) y aplicar shrinkage
    matriz_covarianzas_raw = rendimientos.cov() * 252
    matriz_covarianzas = pd.DataFrame(
        _shrinkage_covariance(matriz_covarianzas_raw.values, alpha=shrinkage_alpha),
        index=matriz_covarianzas_raw.index,
        columns=matriz_covarianzas_raw.columns,
    )
    
    print(f"\n✅ Rendimientos esperados anuales:")
    for i, ticker in enumerate(ticker_symbols):
        print(f"   {ticker}: {rendimientos_esperados.iloc[i]*100:.2f}%")
    
    print(f"\n✅ Riesgo (volatilidad anual):")
    for i, ticker in enumerate(ticker_symbols):
        riesgo = np.sqrt(matriz_covarianzas.iloc[i, i])
        print(f"   {ticker}: {riesgo*100:.2f}%")
    
    # ========================================================================
    # PARTE 4: OPTIMIZACIÓN DE CARTERA (Basada en Teoría)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PARTE 4: OPTIMIZACIÓN DE CARTERA")
    print("=" * 80)
    
    C = matriz_covarianzas.values
    m = rendimientos_esperados.values
    tasa_libre_riesgo = 0.02  # 2% anual
    
    if estrategia == 'minimo_riesgo':
        print("\n📊 Optimizando para mínimo riesgo (Tema 9)...")
        resultado = optimizar_cartera_minimo_riesgo(C, m)
        print(f"   ✅ Riesgo mínimo: {resultado['riesgo']*100:.2f}% anual")
        print(f"   ✅ Rendimiento esperado: {resultado['rendimiento']*100:.2f}% anual")
        
    elif estrategia == 'sharpe':
        print("\n📊 Optimizando para máximo Sharpe Ratio (Tema 10 - CAPM)...")
        resultado = optimizar_cartera_maximo_sharpe(C, m, tasa_libre_riesgo)
        print(f"   ✅ Sharpe Ratio: {resultado['sharpe_ratio']:.3f}")
        print(f"   ✅ Riesgo: {resultado['riesgo']*100:.2f}% anual")
        print(f"   ✅ Rendimiento esperado: {resultado['rendimiento']*100:.2f}% anual")
        
    elif estrategia == 'rendimiento_objetivo':
        rendimiento_objetivo = 0.10  # 10% anual
        print(f"\n📊 Optimizando para rendimiento objetivo: {rendimiento_objetivo*100:.2f}% anual...")
        resultado = optimizar_cartera_rendimiento_objetivo(C, m, rendimiento_objetivo)
        print(f"   ✅ Riesgo: {resultado['riesgo']*100:.2f}% anual")
        print(f"   ✅ Rendimiento esperado: {resultado['rendimiento']*100:.2f}% anual")
    else:
        # Por defecto, usar Sharpe
        resultado = optimizar_cartera_maximo_sharpe(C, m, tasa_libre_riesgo)
    
    pesos_optimos = _aplicar_limites_peso(resultado['pesos'].copy(), min_weight=min_weight, max_weight=max_weight)
    resultado['pesos'] = pesos_optimos
    resultado['riesgo'] = np.sqrt(pesos_optimos @ C @ pesos_optimos.T)
    if m is not None:
        resultado['rendimiento'] = pesos_optimos @ m
    
    # Filtrar pesos significativos (> 1%)
    pesos_filtrados = {}
    for i, ticker in enumerate(ticker_symbols):
        if pesos_optimos[i] > 0.01:
            pesos_filtrados[ticker] = pesos_optimos[i]
    
    print(f"\n✅ Cartera optimizada ({len(pesos_filtrados)} activos):")
    pesos_ordenados = sorted(pesos_filtrados.items(), key=lambda x: x[1], reverse=True)
    for ticker, peso in pesos_ordenados:
        print(f"   {ticker}: {peso*100:.2f}%")
    
    # ========================================================================
    # PARTE 5: SIMULACIÓN Y PROYECCIÓN
    # ========================================================================
    print("\n" + "=" * 80)
    print("PARTE 5: SIMULACIÓN Y PROYECCIÓN")
    print("=" * 80)
    
    # Calcular rendimiento y riesgo de la cartera optimizada
    rendimiento_cartera = resultado.get('rendimiento', pesos_optimos @ m)
    riesgo_cartera = resultado.get('riesgo', np.sqrt(pesos_optimos @ C @ pesos_optimos.T))
    
    # Simulación Monte Carlo
    n_simulaciones = 1000
    rendimientos_simulados = np.random.normal(
        rendimiento_cartera,
        riesgo_cartera,
        n_simulaciones
    )
    
    valores_futuros = capital_inicial * (1 + rendimientos_simulados)
    
    print(f"\n📊 Resultados de simulación ({n_simulaciones} escenarios):")
    print(f"   Rendimiento esperado: {rendimiento_cartera*100:.2f}% anual")
    print(f"   Riesgo (desviación estándar): {riesgo_cartera*100:.2f}% anual")
    print(f"   Valor esperado después de 1 año: ${np.mean(valores_futuros):,.2f}")
    print(f"   Percentil 5% (peor caso): ${np.percentile(valores_futuros, 5):,.2f}")
    print(f"   Percentil 95% (mejor caso): ${np.percentile(valores_futuros, 95):,.2f}")
    
    probabilidad_ganancia = np.mean(rendimientos_simulados > 0) * 100
    print(f"   Probabilidad de ganancia: {probabilidad_ganancia:.1f}%")
    
    # ========================================================================
    # PARTE 6: GRÁFICOS (si se solicita)
    # ========================================================================
    grafico_path = None
    grafico_paths = []
    if generar_graficos:
        print("\n📊 Generando gráficos (uno por imagen)...")
        try:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tickers_grafico = [t for t, p in pesos_ordenados]
            pesos_grafico = [p * 100 for t, p in pesos_ordenados]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(tickers_grafico, pesos_grafico)
            ax.set_xlabel('Peso (%)')
            ax.set_title('Distribución de Pesos Optimizados')
            ax.grid(True, alpha=0.3)
            grafico_paths.append({
                'path': _guardar_grafica(fig, 'lp_pesos', stamp),
                'caption': 'Distribución de pesos',
            })

            fig, ax = plt.subplots(figsize=(10, 6))
            for i, ticker in enumerate(ticker_symbols):
                riesgo_activo = np.sqrt(C[i, i])
                rend_activo = m[i]
                ax.scatter(riesgo_activo * 100, rend_activo * 100, alpha=0.6)
                ax.annotate(ticker, (riesgo_activo * 100, rend_activo * 100), fontsize=8)
            ax.scatter(
                riesgo_cartera * 100, rendimiento_cartera * 100,
                color='red', s=200, marker='*', label='Cartera Optimizada',
            )
            ax.set_xlabel('Riesgo (%)')
            ax.set_ylabel('Rendimiento Esperado (%)')
            ax.set_title('Riesgo vs Rendimiento')
            ax.legend()
            ax.grid(True, alpha=0.3)
            grafico_paths.append({
                'path': _guardar_grafica(fig, 'lp_riesgo_rendimiento', stamp),
                'caption': 'Riesgo vs rendimiento',
            })

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(valores_futuros, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(capital_inicial, color='red', linestyle='--', label='Capital Inicial')
            ax.axvline(np.mean(valores_futuros), color='green', linestyle='--', label='Valor Esperado')
            ax.set_xlabel('Valor ($)')
            ax.set_ylabel('Frecuencia')
            ax.set_title('Distribución de Valores Futuros (1 año)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            grafico_paths.append({
                'path': _guardar_grafica(fig, 'lp_valores_futuros', stamp),
                'caption': 'Valores futuros (1 año)',
            })

            fig, ax = plt.subplots(figsize=(10, 8))
            matriz_correlacion = rendimientos.corr()
            im = ax.imshow(matriz_correlacion, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax.set_xticks(range(len(ticker_symbols)))
            ax.set_yticks(range(len(ticker_symbols)))
            ax.set_xticklabels(ticker_symbols, rotation=45, ha='right')
            ax.set_yticklabels(ticker_symbols)
            ax.set_title('Matriz de Correlación')
            fig.colorbar(im, ax=ax)
            grafico_paths.append({
                'path': _guardar_grafica(fig, 'lp_correlacion', stamp),
                'caption': 'Matriz de correlación',
            })

            fig, ax = plt.subplots(figsize=(10, 6))
            precios_normalizados = precios / precios.iloc[0]
            for ticker in tickers_grafico[:5]:
                if ticker in precios_normalizados.columns:
                    ax.plot(precios_normalizados.index, precios_normalizados[ticker], label=ticker, alpha=0.7)
            ax.set_xlabel('Fecha')
            ax.set_ylabel('Precio Normalizado')
            ax.set_title('Evolución de Precios (Top 5)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            grafico_paths.append({
                'path': _guardar_grafica(fig, 'lp_evolucion', stamp),
                'caption': 'Evolución de precios (Top 5)',
            })

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('off')
            resumen_texto = (
                "RESUMEN DE CARTERA LARGO PLAZO\n\n"
                f"Capital Inicial: ${capital_inicial:,.2f}\n"
                f"Estrategia: {estrategia.upper()}\n\n"
                f"Rendimiento Esperado: {rendimiento_cartera*100:.2f}% anual\n"
                f"Riesgo: {riesgo_cartera*100:.2f}% anual\n"
                f"Sharpe Ratio: {resultado.get('sharpe_ratio', 0):.3f}\n\n"
                f"Valor Esperado (1 año): ${np.mean(valores_futuros):,.2f}\n"
                f"Ganancia Esperada: ${np.mean(valores_futuros) - capital_inicial:,.2f}\n\n"
                f"Probabilidad de Ganancia: {probabilidad_ganancia:.1f}%\n"
                f"Peor Caso (5%): ${np.percentile(valores_futuros, 5):,.2f}\n"
                f"Mejor Caso (95%): ${np.percentile(valores_futuros, 95):,.2f}\n\n"
                f"Activos en Cartera: {len(pesos_filtrados)}"
            )
            ax.text(0.08, 0.5, resumen_texto, fontsize=12, verticalalignment='center', family='monospace')
            grafico_paths.append({
                'path': _guardar_grafica(fig, 'lp_resumen', stamp),
                'caption': 'Resumen de cartera',
            })

            grafico_path = grafico_paths[0]['path'] if grafico_paths else None
            print(f"   ✅ {len(grafico_paths)} gráficos guardados por separado")
        except Exception as e:
            print(f"   ⚠️  Error generando gráficos: {e}")
            try:
                plt.close('all')
            except Exception:
                pass
    
    # ========================================================================
    # RESULTADOS FINALES
    # ========================================================================
    resultados = {
        'cartera_optimizada': resultado,
        'pesos_optimizados': dict(zip(ticker_symbols, pesos_optimos)),
        'rendimientos_anuales': rendimientos_esperados,
        'rendimientos': rendimientos,
        'matriz_covarianzas': matriz_covarianzas,
        'rendimiento_esperado': rendimiento_cartera,
        'riesgo': riesgo_cartera,
        'valores_futuros_simulados': valores_futuros,
        'probabilidad_ganancia': probabilidad_ganancia,
        'tickers': ticker_symbols,
        'nombres': nombres,
        'grafico_path': grafico_path,
        'grafico_paths': grafico_paths,
    }
    
    return resultados


# ============================================================================
# EJECUTAR
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CARTERA DE INVERSIÓN - LARGO PLAZO")
    print("Optimización Teórica (Markowitz + CAPM)")
    print("=" * 80)
    
    # Configuración
    CAPITAL = 10000
    RIESGO_MAX = 0.15  # 15% anual
    ESTRATEGIA = 'sharpe'  # 'minimo_riesgo', 'sharpe', o 'rendimiento_objetivo'
    
    # Ejecutar análisis
    resultados = analizar_cartera_largo_plazo(
        capital_inicial=CAPITAL,
        riesgo_max=RIESGO_MAX,
        estrategia=ESTRATEGIA,
        generar_graficos=True
    )
    
    if resultados:
        print("\n✅ Análisis completado exitosamente")

