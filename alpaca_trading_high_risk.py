#!/usr/bin/env python3
"""
Trading automatizado con Alpaca - CARTERA ALTO RIESGO (Corto Plazo).
Script COMPLETO y autónomo: no depende de alpaca_trading_base.
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("❌ Instala: pip install numpy pandas")
    sys.exit(1)

try:
    from cartera_alto_riesgo_corto_plazo import analizar_cartera_alto_riesgo, get_tickers_alto_riesgo
except ImportError as e:
    print("❌ No se pudo importar cartera_alto_riesgo_corto_plazo:", e)
    sys.exit(1)

try:
    from cartera_largo_plazo import analizar_cartera_largo_plazo
except ImportError:
    analizar_cartera_largo_plazo = None

try:
    from alpaca_trading_helpers import (
        get_portfolio_config,
        risk_checks,
        position_size_from_risk,
        generate_mini_report,
        generate_big_report,
        apply_cooldown_and_max_trades,
        load_state,
        save_state,
        safe_api_call,
        compute_drawdown,
        mini_report_if_due,
        full_report_if_due,
        check_stop_and_trailing,
        should_skip_buy_due_to_cooldown,
        update_trailing_state_for_position,
        normalize_state_positions,
    )
except ImportError:
    get_portfolio_config = risk_checks = position_size_from_risk = None
    generate_mini_report = generate_big_report = apply_cooldown_and_max_trades = None
    load_state = save_state = safe_api_call = compute_drawdown = None
    mini_report_if_due = full_report_if_due = check_stop_and_trailing = None
    should_skip_buy_due_to_cooldown = update_trailing_state_for_position = None
    normalize_state_positions = None

try:
    import requests
    REQUESTS_DISPONIBLE = True
except ImportError:
    REQUESTS_DISPONIBLE = False

# ============== CONFIG (centralizada + helpers) ==============
CONFIG = {
    'modo_paper': True,
    'capital_inicial': 10000,
    'riesgo_max': 0.10,
    'riesgo_max_mensual': 0.10,
    'mini_monitor_interval_seconds': 60,
    'full_report_interval_minutes': 30,
    'usar_top_5_acciones': True,
    'top_5_numero': 4,   # Solo las 4 mejores (no comprar las que están cayendo)
    'top_5_criterio': 'sharpe',
    'stop_loss_porcentaje': 0.07,   # 7% (ajuste: menos ventas en correcciones pequeñas)
    'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
    'enviar_telegram': os.getenv('ENVIAR_TELEGRAM', 'true').lower() in ('true', '1', 'yes'),
    'etiqueta_cartera': 'ALTO RIESGO',
}
# Merge helpers; fuente de verdad para reporte grande: full_report_interval_minutes
if get_portfolio_config is not None:
    _base = get_portfolio_config('alto_riesgo', CONFIG)
    for k, v in _base.items():
        if k == "report_interval_seconds":
            continue
        if k not in CONFIG or CONFIG.get(k) is None:
            CONFIG[k] = v
CONFIG["report_interval_seconds"] = CONFIG["full_report_interval_minutes"] * 60
if get_portfolio_config is not None and "state_file" not in CONFIG:
    CONFIG["state_file"] = get_portfolio_config("alto_riesgo", {}).get("state_file", "state_high_risk.json")
# Opcional: desactivar trailing (solo stop fijo): CONFIG["trailing_stop_porcentaje"] = None

logger = logging.getLogger("alpaca_trading")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

ALPACA_CONFIG = {
    'api_key': os.getenv('ALPACA_API_KEY', ''),
    'secret_key': os.getenv('ALPACA_SECRET_KEY', ''),
    'base_url': 'https://paper-api.alpaca.markets',
}

TICKERS_ALTO_RIESGO = list(get_tickers_alto_riesgo().keys())


# ============== TELEGRAM ==============
def enviar_mensaje_telegram(bot_token, chat_id, mensaje, parse_mode='HTML'):
    """Envía un mensaje a Telegram. Devuelve True si se envió correctamente."""
    if not REQUESTS_DISPONIBLE or not bot_token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={'chat_id': chat_id, 'text': mensaje, 'parse_mode': parse_mode},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"   ⚠️ Telegram: {e}")
        return False


def _state_ts(state, key):
    """Convierte last_mini_report_ts o last_full_report_ts del estado a float (unix ts). Lee de state['meta'] o state."""
    v = (state.get("meta") or {}).get(key) if isinstance(state.get("meta"), dict) else None
    if v is None:
        v = state.get(key)
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None
    return None


def _verificar_stop_loss(trading, state, config, price_cache, state_file, warmup_only=False):
    """
    Stop-loss y trailing stop: actualiza state por símbolo, vende si se activa y registra last_stop_ts (cooldown).
    Si warmup_only=True: solo actualiza max_price_since_entry (no evalúa stop ni vende). Para warm-up inicial.
    """
    if not trading or not trading.api:
        return 0
    state.setdefault("symbols", {})
    positions = safe_api_call(trading.api.list_positions) if safe_api_call else trading.api.list_positions()
    if not positions:
        return 0
    if warmup_only:
        now_ts = time.time()
        for p in positions:
            symbol = getattr(p, "symbol", None)
            if not symbol:
                continue
            qty = float(getattr(p, "qty", 0) or 0)
            if qty <= 0:
                continue
            entry_price = None
            if hasattr(p, "avg_entry_price") and p.avg_entry_price:
                entry_price = float(p.avg_entry_price)
            elif hasattr(p, "cost_basis") and p.cost_basis and qty:
                entry_price = float(p.cost_basis) / qty
            if entry_price is None or entry_price <= 0:
                continue
            current_price = None
            if safe_api_call:
                trade = safe_api_call(trading.api.get_latest_trade, symbol)
                if trade and hasattr(trade, "price"):
                    current_price = float(trade.price)
                if current_price is None:
                    bar = safe_api_call(trading.api.get_latest_bar, symbol)
                    if bar and hasattr(bar, "c"):
                        current_price = float(bar.c)
            if current_price is None and price_cache and symbol in price_cache:
                current_price = price_cache[symbol][0] if isinstance(price_cache[symbol], (list, tuple)) else price_cache[symbol]
            if current_price is None or current_price <= 0:
                continue
            if price_cache is not None:
                price_cache[symbol] = (current_price, now_ts)
            if update_trailing_state_for_position:
                update_trailing_state_for_position(state, symbol, entry_price, current_price, now_ts)
        return 0
    if not check_stop_and_trailing or not save_state:
        return 0
    stop_pct = config.get("stop_loss_porcentaje") or 0.05
    trailing_pct = config.get("trailing_stop_porcentaje")
    sold = 0
    now_ts = time.time()
    for p in positions:
        symbol = getattr(p, "symbol", None)
        if not symbol:
            continue
        qty = float(getattr(p, "qty", 0) or 0)
        if qty <= 0:
            continue
        entry_price = None
        if hasattr(p, "avg_entry_price") and p.avg_entry_price:
            entry_price = float(p.avg_entry_price)
        elif hasattr(p, "cost_basis") and p.cost_basis and qty:
            entry_price = float(p.cost_basis) / qty
        if entry_price is None or entry_price <= 0:
            continue
        current_price = None
        if safe_api_call:
            trade = safe_api_call(trading.api.get_latest_trade, symbol)
            if trade and hasattr(trade, "price"):
                current_price = float(trade.price)
            if current_price is None:
                bar = safe_api_call(trading.api.get_latest_bar, symbol)
                if bar and hasattr(bar, "c"):
                    current_price = float(bar.c)
        if current_price is None and price_cache and symbol in price_cache:
            current_price = price_cache[symbol][0] if isinstance(price_cache[symbol], (list, tuple)) else price_cache[symbol]
        if current_price is None or current_price <= 0:
            continue
        if price_cache is not None:
            price_cache[symbol] = (current_price, now_ts)
        if update_trailing_state_for_position:
            update_trailing_state_for_position(state, symbol, entry_price, current_price, now_ts)
        pos_state = state["symbols"].get(symbol) or {}
        max_p = pos_state.get("max_price_since_entry") or current_price
        try:
            max_p = float(max_p)
        except (TypeError, ValueError):
            max_p = current_price
        result = check_stop_and_trailing(entry_price, max_p, current_price, stop_pct, trailing_stop_pct=trailing_pct)
        if result:
            reason, trigger_price, pct_used = result
            try:
                trading.api.submit_order(symbol=symbol, qty=int(qty), side="sell", type="market", time_in_force="day")
                sold += 1
                state["symbols"].setdefault(symbol, {})["last_stop_ts"] = now_ts
                state["symbols"][symbol]["last_action_ts"] = now_ts
                logger.info(
                    "[HIGH_RISK] %s | entry=%.2f max_since=%.2f current=%.2f trigger=%.2f pct=%.2f%% reason=%s",
                    symbol, entry_price, max_p, current_price, trigger_price, pct_used * 100, reason,
                )
                save_state(state_file, state)
            except Exception as e:
                logger.warning("[HIGH_RISK] Venta stop %s: %s", symbol, e)
    return sold


# ============== ALPACA API ==============
def _get_api():
    """Conexión a Alpaca (import dentro para evitar fallos al inicio)."""
    try:
        import alpaca_trade_api as tradeapi
        return tradeapi.REST(
            key_id=ALPACA_CONFIG['api_key'],
            secret_key=ALPACA_CONFIG['secret_key'],
            base_url=ALPACA_CONFIG['base_url'],
            api_version='v2',
        )
    except Exception as e:
        print(f"⚠️ Alpaca no disponible: {e}")
        return None


class TradingAlpaca:
    """Clase mínima de trading Alpaca (contenida en este script)."""
    def __init__(self, modo_paper=True):
        self.modo_paper = modo_paper
        self.api = _get_api()

    def obtener_precio_actual(self, symbol, mostrar_warnings=True):
        if not self.api:
            return None
        try:
            trade = self.api.get_latest_trade(symbol)
            if trade and hasattr(trade, 'price'):
                return float(trade.price)
            bar = self.api.get_latest_bar(symbol)
            if bar and hasattr(bar, 'c'):
                return float(bar.c)
        except Exception as e:
            if mostrar_warnings:
                print(f"   ⚠️ Precio {symbol}: {e}")
        return None

    def obtener_todas_posiciones(self):
        if not self.api:
            return {}
        try:
            positions = self.api.list_positions()
            return {p.symbol: float(p.qty) for p in positions if float(p.qty) != 0}
        except Exception:
            return {}

    def obtener_resumen(self, enviar_telegram=False):
        if not self.api:
            return
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            cash = float(account.cash)
            port_value = float(account.portfolio_value)
            etq = CONFIG.get('etiqueta_cartera', 'ALTO RIESGO')
            print(f"\n   [{etq}] 💰 Cash: ${cash:,.2f} | Portfolio: ${port_value:,.2f}")
            for p in positions[:10]:
                print(f"      {p.symbol}: {p.qty} @ ${float(p.market_value or 0):,.2f}")
            if enviar_telegram and CONFIG.get('enviar_telegram') and CONFIG.get('telegram_bot_token') and CONFIG.get('telegram_chat_id'):
                msg = f"<b>📊 [{etq}] Cartera Alto Riesgo (corto plazo)</b>\n"
                msg += f"💰 Cash: ${cash:,.2f}\n"
                msg += f"📈 Portfolio: ${port_value:,.2f}\n"
                msg += f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                for p in positions[:8]:
                    msg += f"  {p.symbol}: {p.qty} @ ${float(p.market_value or 0):,.2f}\n"
                enviar_mensaje_telegram(CONFIG['telegram_bot_token'], CONFIG['telegram_chat_id'], msg)
        except Exception as e:
            print(f"   ⚠️ Resumen: {e}")

    def obtener_resumen_breve(self, tickers_esperados=None, enviar_telegram=False):
        self.obtener_resumen(enviar_telegram=enviar_telegram)

    def identificar_top_acciones(self, pesos_optimizados, precios_actuales, num_top=5, criterio='sharpe', rendimientos=None):
        """Top-N por criterio real: sharpe (ratio por ticker) o ganancia_pct (performance reciente)."""
        if not pesos_optimizados or not precios_actuales:
            return pesos_optimizados
        tickers_comunes = [t for t in pesos_optimizados if t in (precios_actuales or {})]
        if not tickers_comunes:
            ordenados = sorted(pesos_optimizados.items(), key=lambda x: x[1], reverse=True)[:num_top]
            total = sum(p for _, p in ordenados)
            return {t: p / total for t, p in ordenados} if total > 0 else dict(ordenados)
        if rendimientos is None or not hasattr(rendimientos, 'columns'):
            ordenados = sorted(
                [(t, pesos_optimizados[t]) for t in tickers_comunes],
                key=lambda x: x[1], reverse=True
            )[:num_top]
            total = sum(p for _, p in ordenados)
            return {t: p / total for t, p in ordenados} if total > 0 else dict(ordenados)
        cols = [c for c in rendimientos.columns if c in tickers_comunes]
        if not cols:
            ordenados = sorted(
                [(t, pesos_optimizados[t]) for t in tickers_comunes],
                key=lambda x: x[1], reverse=True
            )[:num_top]
            total = sum(p for _, p in ordenados)
            return {t: p / total for t, p in ordenados} if total > 0 else dict(ordenados)
        R = rendimientos[cols].dropna(how='all')
        if R.empty or len(R) < 2:
            ordenados = sorted(
                [(t, pesos_optimizados[t]) for t in tickers_comunes],
                key=lambda x: x[1], reverse=True
            )[:num_top]
            total = sum(p for _, p in ordenados)
            return {t: p / total for t, p in ordenados} if total > 0 else dict(ordenados)
        if criterio == 'sharpe':
            rf = 0.02 / 252
            mu = R.mean()
            sigma = R.std()
            sigma = sigma.replace(0, np.nan)
            sharpe = (mu - rf) / sigma
            sharpe = sharpe.fillna(-1e9)
            # Solo mantener activos con sharpe positivo (no los que están cayendo)
            positivos = sharpe[sharpe > 0].sort_values(ascending=False)
            ordenados_tickers = positivos.head(num_top).index.tolist()
            if not ordenados_tickers:
                ordenados_tickers = sharpe.sort_values(ascending=False).head(num_top).index.tolist()
        elif criterio == 'ganancia_pct':
            ganancia = (1 + R).prod() - 1
            ganancia = ganancia.fillna(-1e9)
            # Solo mantener activos con ganancia positiva (no los que están cayendo)
            positivos = ganancia[ganancia > 0].sort_values(ascending=False)
            ordenados_tickers = positivos.head(num_top).index.tolist()
            if not ordenados_tickers:
                ordenados_tickers = ganancia.sort_values(ascending=False).head(num_top).index.tolist()
        else:
            ordenados = sorted(
                [(t, pesos_optimizados[t]) for t in tickers_comunes],
                key=lambda x: x[1], reverse=True
            )[:num_top]
            total = sum(p for _, p in ordenados)
            return {t: p / total for t, p in ordenados} if total > 0 else dict(ordenados)
        top_pesos = {t: pesos_optimizados[t] for t in ordenados_tickers if t in pesos_optimizados}
        total = sum(top_pesos.values())
        if total <= 0:
            return {t: 1.0 / len(top_pesos) for t in top_pesos} if top_pesos else pesos_optimizados
        return {t: p / total for t, p in top_pesos.items()}

    def rebalancear_cartera(self, pesos_optimizados, precios_actuales, posiciones_actuales, solo_volumen=True, enviar=False):
        """Construye lista de órdenes de rebalanceo. Si enviar=False (default), no envía; el caller aplica risk/cooldown y envía."""
        if not self.api or not precios_actuales:
            return []
        try:
            account = self.api.get_account()
            port_value = float(account.portfolio_value)
            ordenes = []
            for symbol, peso in pesos_optimizados.items():
                precio = precios_actuales.get(symbol)
                if not precio or precio <= 0:
                    continue
                objetivo_valor = port_value * peso
                qty_actual = posiciones_actuales.get(symbol, 0)
                valor_actual = qty_actual * precio
                diff = objetivo_valor - valor_actual
                if abs(diff) < 10:
                    continue
                qty = int(abs(diff) / precio)
                if qty == 0:
                    continue
                side = 'buy' if diff > 0 else 'sell'
                if side == 'sell' and qty_actual < qty:
                    qty = int(qty_actual)
                if qty <= 0:
                    continue
                ordenes.append({'symbol': symbol, 'side': side, 'qty': qty})
                if enviar:
                    try:
                        self.api.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='day')
                    except Exception as e:
                        logger.warning("Orden %s: %s", symbol, e)
            return ordenes
        except Exception as e:
            logger.warning("Rebalanceo: %s", e)
            return []

    def configurar_stop_loss(self, ticker, porcentaje_stop=0.05):
        pass  # Opcional: persistir en archivo

    def verificar_stop_loss(self):
        return []


def sistema_completo_alpaca(
    capital_inicial=10000,
    riesgo_max=0.10,
    modo_paper=True,
    ejecutar_ordenes=True,
    monitorear=True,
    tipo_cartera='alto_riesgo',
    analizar_cartera=None,
    tickers_completos=None,
    kwargs_analisis=None,
    analizar_cartera_largo_plazo=None,
):
    """Flujo completo: analizar cartera, pesos, precios, rebalancear, monitorear."""
    if not analizar_cartera:
        print("❌ analizar_cartera es obligatorio")
        return None
    trading = TradingAlpaca(modo_paper=modo_paper)
    if not trading.api:
        print("❌ No se pudo conectar con Alpaca. Revisa ALPACA_API_KEY y ALPACA_SECRET_KEY en .env")
        return None

    print("\n📊 Analizando cartera (alto riesgo)...")
    kwargs = dict(capital_inicial=capital_inicial, riesgo_max=riesgo_max, **(kwargs_analisis or {}))
    try:
        resultados = analizar_cartera(**kwargs)
    except Exception as e:
        print(f"❌ Error analizar cartera: {e}")
        import traceback
        traceback.print_exc()
        return None
    if not resultados:
        return None

    cartera_opt = resultados['cartera_optimizada']
    rm = resultados.get('rendimientos_mensuales')
    if rm is None:
        rm = resultados.get('rendimientos_anuales')
    ticker_symbols_optimizados = list(rm.index) if rm is not None else resultados.get('tickers', [])
    if not ticker_symbols_optimizados and 'tickers' in resultados:
        ticker_symbols_optimizados = resultados['tickers']
    tickers_completos = tickers_completos or ticker_symbols_optimizados
    ticker_symbols = sorted(set(tickers_completos))

    peso_min = 0.002
    n = len(ticker_symbols)
    peso_total_min = peso_min * n
    if peso_total_min > 1:
        peso_min = 1.0 / n
    ticker_to_idx = {t: i for i, t in enumerate(ticker_symbols_optimizados)}
    pesos_optimizados = {}
    for t in ticker_symbols:
        if t in ticker_to_idx:
            idx = ticker_to_idx[t]
            pesos_optimizados[t] = max(0, cartera_opt['pesos'][idx])
        else:
            pesos_optimizados[t] = peso_min
    total_p = sum(pesos_optimizados.values())
    if total_p > 0:
        pesos_optimizados = {t: p / total_p for t, p in pesos_optimizados.items()}

    rendimientos_df = resultados.get('rendimientos')
    if CONFIG.get('usar_top_5_acciones'):
        precios_actuales = {}
        for t in list(pesos_optimizados.keys())[:20]:
            p = trading.obtener_precio_actual(t, mostrar_warnings=False)
            if p:
                precios_actuales[t] = p
        if precios_actuales:
            pesos_optimizados = trading.identificar_top_acciones(
                pesos_optimizados, precios_actuales,
                num_top=CONFIG.get('top_5_numero', 4),
                criterio=CONFIG.get('top_5_criterio', 'sharpe'),
                rendimientos=rendimientos_df,
            )

    precios_actuales = {}
    for t in pesos_optimizados:
        p = trading.obtener_precio_actual(t, mostrar_warnings=False)
        if p:
            precios_actuales[t] = p
    pesos_optimizados = {t: p for t, p in pesos_optimizados.items() if t in precios_actuales}
    total_p = sum(pesos_optimizados.values())
    if total_p > 0:
        pesos_optimizados = {t: p / total_p for t, p in pesos_optimizados.items()}

    logger.info("[HIGH_RISK] Cartera: %d activos", len(pesos_optimizados))
    for t, p in sorted(pesos_optimizados.items(), key=lambda x: -x[1])[:10]:
        logger.info("  %s: %.2f%%", t, p * 100)

    # Log config de salida y reportes al arrancar
    mini_s = CONFIG.get("mini_monitor_interval_seconds", 60)
    full_min = CONFIG.get("full_report_interval_minutes", 30)
    if CONFIG.get("report_interval_seconds"):
        full_min = CONFIG["report_interval_seconds"] // 60
    logger.info(
        "[HIGH_RISK] Config: etiqueta=%s | mini_interval=%ds | full_interval=%d min | stop_loss_pct=%s | trailing_stop_pct=%s",
        CONFIG.get("etiqueta_cartera", "ALTO RIESGO"),
        mini_s,
        full_min,
        CONFIG.get("stop_loss_porcentaje"),
        CONFIG.get("trailing_stop_porcentaje"),
    )

    state_file = CONFIG.get("state_file", "state_high_risk.json")
    state = load_state(state_file) if load_state else {"symbols": {}, "meta": {"last_mini_report_ts": None, "last_full_report_ts": None, "peak_equity": None, "last_orders_since_full": []}}
    # Corregir inconsistencias: max_price_since_entry nunca debe ser < entry_price (rompe trailing/stop al iniciar)
    if normalize_state_positions:
        try:
            n_norm, n_fixed = normalize_state_positions(state)
            if n_norm or n_fixed:
                if save_state:
                    save_state(state_file, state)
                logger.info("[HIGH_RISK] [STATE] normalized %s symbols (fixed %s max_price issues)", n_norm, n_fixed)
        except Exception as e:
            logger.warning("[HIGH_RISK] normalize_state_positions: %s", e)
    price_cache = {}

    peak_equity = None
    orders_in_window = []
    last_trade_per_symbol = {}
    last_orders_snapshot = []
    if trading.api:
        try:
            peak_equity = float(trading.api.get_account().portfolio_value)
        except Exception:
            pass
    if risk_checks is not None and peak_equity is not None:
        ok, reason = risk_checks(peak_equity, peak_equity, orders_in_window, CONFIG)
        if not ok:
            logger.warning("Risk check bloqueó nuevas operaciones: %s", reason)
            ejecutar_ordenes = False

    if ejecutar_ordenes:
        posiciones_actuales = trading.obtener_todas_posiciones()
        ordenes = trading.rebalancear_cartera(pesos_optimizados, precios_actuales, posiciones_actuales, solo_volumen=True, enviar=False)
        # Vender posiciones que ya no están en el top N (solo mantener las 4 mejores)
        top_symbols = set(pesos_optimizados.keys())
        for sym, qty in (posiciones_actuales or {}).items():
            if sym not in top_symbols and qty and int(qty) > 0:
                ordenes.append({'symbol': sym, 'side': 'sell', 'qty': int(qty)})
        if ordenes and should_skip_buy_due_to_cooldown:
            cooldown_min = CONFIG.get("cooldown_minutes_after_stop", 30)
            now_ts = time.time()
            _filtered = []
            for o in ordenes:
                if o.get("side") == "buy" and should_skip_buy_due_to_cooldown(state, o["symbol"], now_ts, cooldown_min):
                    logger.info("[HIGH_RISK] SKIP_BUY_COOLDOWN %s", o["symbol"])
                    continue
                _filtered.append(o)
            ordenes = _filtered
        if ordenes and apply_cooldown_and_max_trades is not None:
            ordenes, last_trade_per_symbol, orders_in_window = apply_cooldown_and_max_trades(ordenes, last_trade_per_symbol, orders_in_window, CONFIG, state=state)
        if ordenes:
            for o in ordenes:
                try:
                    trading.api.submit_order(symbol=o['symbol'], qty=o['qty'], side=o['side'], type='market', time_in_force='day')
                    orders_in_window.append(datetime.now())
                    last_orders_snapshot.append(o)
                    logger.info("Orden enviada: %s %s %s", o['side'], o['qty'], o['symbol'])
                except Exception as e:
                    logger.warning("Orden %s: %s", o.get('symbol'), e)
            logger.info("%d órdenes enviadas", len(ordenes))
        else:
            logger.info("Sin órdenes de rebalanceo (alineado o risk/cooldown)")
    last_rebalance_ts = time.time()

    trading.obtener_resumen()

    if monitorear:
        etq = CONFIG.get('etiqueta_cartera', 'ALTO RIESGO')
        env_telegram = CONFIG.get('enviar_telegram', False)
        mini_interval = CONFIG.get('mini_monitor_interval_seconds', 60)
        full_interval_sec = CONFIG["report_interval_seconds"]
        tiene_telegram = bool(CONFIG.get('telegram_bot_token') and CONFIG.get('telegram_chat_id'))
        telegram_params = {'bot_token': CONFIG.get('telegram_bot_token'), 'chat_id': CONFIG.get('telegram_chat_id')}
        print("\n" + "─" * 50)
        print("   📌 CARTERA:", etq, "(corto plazo)")
        rebalance_interval = CONFIG.get("rebalance_interval_seconds", 60)
        print("   ⏱️ Mini reporte: cada", mini_interval, "s. — Rebalanceo cada", rebalance_interval, "s (top 4; stop en cada ciclo)")
        print("   📋 Reporte grande: cada", full_interval_sec // 60, "min")
        if env_telegram and tiene_telegram:
            print("   📋 Telegram: activado ✅")
        else:
            print("   📋 Telegram: desactivado ❌")
        print("─" * 50)
        warmup_seconds = CONFIG.get("warmup_seconds", 180)
        loop_start_ts = time.time()
        try:
            while True:
                now_ts = time.time()
                in_warmup = (now_ts - loop_start_ts) < warmup_seconds
                if in_warmup:
                    _verificar_stop_loss(trading, state, CONFIG, price_cache, state_file, warmup_only=True)
                else:
                    if _verificar_stop_loss(trading, state, CONFIG, price_cache, state_file):
                        if save_state:
                            save_state(state_file, state)
                # Rebalanceo periódico cada rebalance_interval_seconds (mismo top 4; precios actualizados)
                rebalance_interval = CONFIG.get("rebalance_interval_seconds", 60)
                if not in_warmup and (now_ts - last_rebalance_ts) >= rebalance_interval:
                    last_rebalance_ts = now_ts
                    posiciones_actuales = trading.obtener_todas_posiciones() if trading.api else {}
                    precios_rebalance = {}
                    for t in (pesos_optimizados or {}):
                        p = trading.obtener_precio_actual(t, mostrar_warnings=False)
                        if p and p > 0:
                            precios_rebalance[t] = p
                    if precios_rebalance and pesos_optimizados:
                        ordenes = trading.rebalancear_cartera(pesos_optimizados, precios_rebalance, posiciones_actuales, solo_volumen=True, enviar=False)
                        top_symbols = set(pesos_optimizados.keys())
                        for sym, qty in (posiciones_actuales or {}).items():
                            if sym not in top_symbols and qty and int(qty) > 0:
                                ordenes.append({'symbol': sym, 'side': 'sell', 'qty': int(qty)})
                        if ordenes and should_skip_buy_due_to_cooldown:
                            cooldown_min = CONFIG.get("cooldown_minutes_after_stop", 30)
                            _f = []
                            for o in ordenes:
                                if o.get("side") == "buy" and should_skip_buy_due_to_cooldown(state, o["symbol"], now_ts, cooldown_min):
                                    logger.info("[HIGH_RISK] SKIP_BUY_COOLDOWN %s", o["symbol"])
                                    continue
                                _f.append(o)
                            ordenes = _f
                        if ordenes and apply_cooldown_and_max_trades:
                            ordenes, last_trade_per_symbol, orders_in_window = apply_cooldown_and_max_trades(ordenes, last_trade_per_symbol, orders_in_window, CONFIG, state=state)
                        if ordenes:
                            for o in ordenes:
                                try:
                                    trading.api.submit_order(symbol=o['symbol'], qty=o['qty'], side=o['side'], type='market', time_in_force='day')
                                    orders_in_window.append(datetime.now())
                                    last_orders_snapshot.append(o)
                                    logger.info("[HIGH_RISK] Rebalance periódico: %s %s %s", o['side'], o['qty'], o['symbol'])
                                except Exception as e:
                                    logger.warning("Orden rebalance %s: %s", o.get('symbol'), e)
                try:
                    if trading.api:
                        acc = safe_api_call(trading.api.get_account) if safe_api_call else trading.api.get_account()
                        if acc:
                            pv = float(acc.portfolio_value)
                            peak_equity = max(peak_equity or pv, pv)
                except Exception:
                    pass
                last_mini_ts = _state_ts(state, "last_mini_report_ts")
                last_full_ts = _state_ts(state, "last_full_report_ts")
                debe_mini = mini_report_if_due(now_ts, last_mini_ts, mini_interval) if mini_report_if_due else True
                if debe_mini:
                    if generate_mini_report is not None:
                        generate_mini_report(trading, CONFIG, env_telegram and tiene_telegram, enviar_mensaje_telegram, telegram_params, peak_equity=peak_equity, data_stale=False)
                    else:
                        trading.obtener_resumen_breve(tickers_esperados=set(pesos_optimizados.keys()), enviar_telegram=env_telegram and tiene_telegram)
                    state.setdefault("meta", {})["last_mini_report_ts"] = now_ts
                    if save_state:
                        save_state(state_file, state)
                debe_full = full_report_if_due(now_ts, last_full_ts, full_interval_sec) if full_report_if_due else True
                if debe_full:
                    # Refrescar top 4 con el análisis actual para que el reporte muestre las mejores acciones de ahora
                    try:
                        kwargs = dict(capital_inicial=capital_inicial, riesgo_max=riesgo_max, **(kwargs_analisis or {}))
                        resultados_refresh = analizar_cartera(**kwargs)
                        if resultados_refresh:
                            cartera_opt = resultados_refresh['cartera_optimizada']
                            rm = resultados_refresh.get('rendimientos_mensuales') or resultados_refresh.get('rendimientos_anuales')
                            ticker_symbols_opt = list(rm.index) if rm is not None else resultados_refresh.get('tickers', [])
                            if not ticker_symbols_opt and 'tickers' in resultados_refresh:
                                ticker_symbols_opt = resultados_refresh['tickers']
                            tickers_loop = tickers_completos or ticker_symbols_opt
                            ticker_symbols = sorted(set(tickers_loop))
                            peso_min = max(0.002, 1.0 / len(ticker_symbols)) if ticker_symbols else 0.002
                            ticker_to_idx = {t: i for i, t in enumerate(ticker_symbols_opt)}
                            nuevos_pesos = {}
                            for t in ticker_symbols:
                                if t in ticker_to_idx:
                                    nuevos_pesos[t] = max(0, cartera_opt['pesos'][ticker_to_idx[t]])
                                else:
                                    nuevos_pesos[t] = peso_min
                            total_p = sum(nuevos_pesos.values())
                            if total_p > 0:
                                nuevos_pesos = {t: p / total_p for t, p in nuevos_pesos.items()}
                            rendimientos_df = resultados_refresh.get('rendimientos')
                            if CONFIG.get('usar_top_5_acciones'):
                                precios_refresh = {}
                                for t in list(nuevos_pesos.keys())[:20]:
                                    p = trading.obtener_precio_actual(t, mostrar_warnings=False)
                                    if p:
                                        precios_refresh[t] = p
                                if precios_refresh:
                                    nuevos_pesos = trading.identificar_top_acciones(
                                        nuevos_pesos, precios_refresh,
                                        num_top=CONFIG.get('top_5_numero', 4),
                                        criterio=CONFIG.get('top_5_criterio', 'sharpe'),
                                        rendimientos=rendimientos_df,
                                    )
                            precios_ok = {}
                            for t in nuevos_pesos:
                                p = trading.obtener_precio_actual(t, mostrar_warnings=False)
                                if p and p > 0:
                                    precios_ok[t] = p
                            nuevos_pesos = {t: p for t, p in nuevos_pesos.items() if t in precios_ok}
                            total_p = sum(nuevos_pesos.values())
                            if total_p > 0 and nuevos_pesos:
                                pesos_optimizados = {t: p / total_p for t, p in nuevos_pesos.items()}
                                logger.info("[HIGH_RISK] Top 4 actualizado para reporte: %s", list(pesos_optimizados.keys()))
                    except Exception as e:
                        logger.warning("[HIGH_RISK] No se pudo refrescar análisis para reporte: %s", e)
                    if generate_big_report is not None:
                        generate_big_report(trading, capital_inicial, CONFIG, env_telegram and tiene_telegram, enviar_mensaje_telegram, telegram_params, last_orders_snapshot, pesos_objetivo=pesos_optimizados, peak_equity=peak_equity)
                    else:
                        try:
                            account = trading.api.get_account()
                            positions = trading.api.list_positions()
                            cash = float(account.cash)
                            port_value = float(account.portfolio_value)
                            ganancia = port_value - capital_inicial
                            ganancia_pct = (ganancia / capital_inicial * 100) if capital_inicial else 0
                            msg = f"<b>🔴 [{etq}] REPORTE COMPLETO — ALTO RIESGO</b>\n<i>Cada {full_interval_sec//60} min</i>\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n<b>💰 RESUMEN</b>\n  Valor: ${port_value:,.2f}\n  Cash: ${cash:,.2f}\n  Ganancia: ${ganancia:,.2f} ({ganancia_pct:+.2f}%)\n\n<b>📋 POSICIONES</b>\n"
                            for p in sorted(positions, key=lambda x: -float(x.market_value or 0))[:20]:
                                msg += f"  • {p.symbol}: {p.qty} → ${float(p.market_value or 0):,.2f}\n"
                            if env_telegram and tiene_telegram:
                                enviar_mensaje_telegram(CONFIG['telegram_bot_token'], CONFIG['telegram_chat_id'], msg)
                            logger.info("[HIGH_RISK] Reporte completo enviado.")
                        except Exception as e:
                            logger.warning("Reporte completo: %s", e)
                    state.setdefault("meta", {})["last_full_report_ts"] = now_ts
                    state["meta"]["last_orders_since_full"] = []
                    if save_state:
                        save_state(state_file, state)
                time.sleep(mini_interval)
        except KeyboardInterrupt:
            logger.info("Monitoreo detenido")
            trading.obtener_resumen()

    return trading, resultados


if __name__ == "__main__":
    try:
        print("\n" + "⚠️" * 40)
        print("TRADING ALPACA - CARTERA ALTO RIESGO (CORTO PLAZO)")
        print("⚠️" * 40)
        print("\nPAPER TRADING por defecto. Revisa .env (ALPACA_API_KEY, ALPACA_SECRET_KEY).")
        print("=" * 80)

        trading_test = TradingAlpaca(modo_paper=True)
        if not trading_test.api:
            print("\n❌ No se pudo conectar con Alpaca. Revisa credenciales en .env")
            sys.exit(1)

        riesgo_max = CONFIG.get('riesgo_max_mensual', CONFIG['riesgo_max'])
        print(f"\n🚀 Iniciando cartera: alto_riesgo | riesgo_max MENSUAL: {riesgo_max*100:.1f}%")
        print("=" * 80)

        resultado = sistema_completo_alpaca(
            capital_inicial=CONFIG['capital_inicial'],
            riesgo_max=riesgo_max,
            modo_paper=CONFIG['modo_paper'],
            ejecutar_ordenes=True,
            monitorear=True,
            tipo_cartera='alto_riesgo',
            analizar_cartera=analizar_cartera_alto_riesgo,
            tickers_completos=TICKERS_ALTO_RIESGO,
            kwargs_analisis={'generar_graficos': False},
            analizar_cartera_largo_plazo=analizar_cartera_largo_plazo,
        )

        if resultado:
            print("\n✅ Sistema completado")
        else:
            print("\n⚠️ Sistema terminó sin resultado")
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
