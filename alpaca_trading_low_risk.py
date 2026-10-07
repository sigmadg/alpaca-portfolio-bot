#!/usr/bin/env python3
"""
Trading automatizado con Alpaca - CARTERA LARGO PLAZO (Bajo Riesgo).
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
    from cartera_largo_plazo import analizar_cartera_largo_plazo, get_tickers_largo_plazo
except (ImportError, ModuleNotFoundError) as e:
    print("❌ No se pudo importar cartera_largo_plazo:", e)
    sys.exit(1)
try:
    from cartera_alto_riesgo_corto_plazo import get_tickers_alto_riesgo
except ImportError:
    get_tickers_alto_riesgo = None

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
        aplicar_max_weight_pesos,
        resolve_peak_and_port,
        clamp_long_only_order,
        orders_to_flatten_shorts,
        sanitize_orders_long_only,
        sleeve_target_capital,
        cap_buys_to_cash,
        filter_orders_own_universe,
        restore_cash_sells,
        leftover_exit_orders,
        finalize_orders_shared_account,
    )
except ImportError:
    get_portfolio_config = risk_checks = position_size_from_risk = None
    generate_mini_report = generate_big_report = apply_cooldown_and_max_trades = None
    load_state = save_state = safe_api_call = compute_drawdown = None
    mini_report_if_due = full_report_if_due = check_stop_and_trailing = None
    should_skip_buy_due_to_cooldown = update_trailing_state_for_position = None
    normalize_state_positions = None
    aplicar_max_weight_pesos = resolve_peak_and_port = None
    clamp_long_only_order = orders_to_flatten_shorts = sanitize_orders_long_only = None
    sleeve_target_capital = cap_buys_to_cash = filter_orders_own_universe = restore_cash_sells = None
    leftover_exit_orders = finalize_orders_shared_account = None

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
    'riesgo_max_anual': 0.10,
    'mini_monitor_interval_seconds': 60,   # Stop cada 60 s (NO rebalanceo)
    'telegram_mini_interval_seconds': 600,  # Monitoreo Telegram cada 10 min
    'full_report_interval_minutes': 60,
    'usar_top_5_acciones': True,
    'top_5_numero': 12,   # Core / largo plazo: 8–15 posiciones
    'top_5_criterio': 'sharpe',
    'max_weight': 0.15,   # Se reaplica DESPUÉS del Top-N
    'allocation_pct': 0.50,  # 50% de la cuenta paper compartida
    'cash_reserve': 500.0,
    'cash_only': True,
    'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
    'enviar_telegram': os.getenv('ENVIAR_TELEGRAM', 'true').lower() in ('true', '1', 'yes'),
}
# Merge config desde helpers (risk, cooldown, max_drawdown, max_trades, mini_monitor, rebalance).
if get_portfolio_config is not None:
    _base = get_portfolio_config('largo_plazo', CONFIG)
    for k, v in _base.items():
        if k == "report_interval_seconds":
            continue  # siempre derivado de full_report_interval_minutes
        if k not in CONFIG or CONFIG.get(k) is None:
            CONFIG[k] = v
    # Arquitectura v2: estos valores deben venir de helpers aunque CONFIG los tuviera
    for k in (
        "rebalance_interval_seconds", "max_drawdown", "etiqueta_cartera",
        "stop_loss_porcentaje", "trailing_stop_porcentaje",
        "cooldown_minutes_after_stop", "max_trades_per_window", "warmup_seconds",
        "allocation_pct", "cash_reserve", "cash_only",
        "telegram_mini_interval_seconds",
    ):
        if k in _base:
            CONFIG[k] = _base[k]
CONFIG["report_interval_seconds"] = CONFIG["full_report_interval_minutes"] * 60
if get_portfolio_config is not None and "state_file" not in CONFIG:
    CONFIG["state_file"] = get_portfolio_config("largo_plazo", {}).get("state_file", "state_low_risk.json")
OWN_UNIVERSE = set(get_tickers_largo_plazo()) if get_tickers_largo_plazo else set()
FOREIGN_UNIVERSE = set(get_tickers_alto_riesgo().keys()) if get_tickers_alto_riesgo else set()
# Opcional: desactivar trailing (solo stop fijo): CONFIG["trailing_stop_porcentaje"] = None


def _cash_real(trading):
    try:
        if not trading or not trading.api:
            return 0.0
        return float(trading.api.get_account().cash)
    except Exception:
        return 0.0


def _precios_con_posiciones(trading, precios, posiciones):
    out = dict(precios or {})
    if not trading:
        return out
    for sym in (posiciones or {}):
        if out.get(sym):
            continue
        p = trading.obtener_precio_actual(sym, mostrar_warnings=False)
        if p:
            out[sym] = p
    return out


def _preparar_ordenes_seguras(trading, ordenes, posiciones, precios):
    precios = _precios_con_posiciones(trading, precios, posiciones)
    if leftover_exit_orders is None and finalize_orders_shared_account is None:
        return sanitize_orders_long_only(ordenes, posiciones) if sanitize_orders_long_only else (ordenes or [])
    if finalize_orders_shared_account is not None:
        return finalize_orders_shared_account(
            ordenes,
            posiciones,
            precios,
            _cash_real(trading),
            own_universe=OWN_UNIVERSE,
            foreign_universe=FOREIGN_UNIVERSE,
            cash_reserve=CONFIG.get("cash_reserve", 500.0),
        )
    return sanitize_orders_long_only(ordenes, posiciones) if sanitize_orders_long_only else (ordenes or [])

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


def enviar_foto_telegram(bot_token, chat_id, image_path, caption=None):
    """Envía una imagen a Telegram (sendPhoto). Devuelve True si se envió correctamente."""
    if not REQUESTS_DISPONIBLE or not bot_token or not chat_id or not image_path:
        return False
    if not os.path.isfile(image_path):
        print(f"   ⚠️ Telegram foto: no existe {image_path}")
        return False
    try:
        with open(image_path, 'rb') as f:
            data = {'chat_id': chat_id}
            if caption:
                data['caption'] = caption[:1024]
            r = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendPhoto",
                data=data,
                files={'photo': f},
                timeout=60,
            )
        if r.status_code != 200:
            print(f"   ⚠️ Telegram foto HTTP {r.status_code}: {r.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"   ⚠️ Telegram foto: {e}")
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


def _verificar_stop_loss(trading, state, config, price_cache, state_file, warmup_only=False, allowed_symbols=None, blocked_symbols=None):
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
        # Solo actualizar trailing state; no compras/ventas
        now_ts = time.time()
        for p in positions:
            symbol = getattr(p, "symbol", None)
            if not symbol:
                continue
            if blocked_symbols and symbol in blocked_symbols:
                continue
            if allowed_symbols and symbol not in allowed_symbols:
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
    trailing_pct = config.get("trailing_stop_porcentaje")  # None = solo stop fijo
    sold = 0
    now_ts = time.time()
    for p in positions:
        symbol = getattr(p, "symbol", None)
        if not symbol:
            continue
        if blocked_symbols and symbol in blocked_symbols:
            continue
        if allowed_symbols and symbol not in allowed_symbols:
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
                    "[LOW_RISK] %s | entry=%.2f max_since=%.2f current=%.2f trigger=%.2f pct=%.2f%% reason=%s",
                    symbol, entry_price, max_p, current_price, trigger_price, pct_used * 100, reason,
                )
                save_state(state_file, state)
            except Exception as e:
                logger.warning("[LOW_RISK] Venta stop %s: %s", symbol, e)
    return sold


def _get_api():
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
            etq = CONFIG.get('etiqueta_cartera', 'BAJO RIESGO')
            print(f"\n   [{etq}] 💰 Cash: ${cash:,.2f} | Portfolio: ${port_value:,.2f}")
            for p in positions[:10]:
                print(f"      {p.symbol}: {p.qty} @ ${float(p.market_value or 0):,.2f}")
            if enviar_telegram and CONFIG.get('enviar_telegram') and CONFIG.get('telegram_bot_token') and CONFIG.get('telegram_chat_id'):
                msg = f"<b>📊 [{etq}] Cartera Largo Plazo (bajo riesgo)</b>\n"
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
        """Construye órdenes de rebalanceo LONG-ONLY (nunca abre shorts)."""
        if not self.api or not precios_actuales:
            return []
        try:
            account = self.api.get_account()
            port_value = float(account.portfolio_value)
            if sleeve_target_capital is not None:
                port_value = sleeve_target_capital(port_value, CONFIG.get("allocation_pct", 0.50))
            ordenes = []
            # Primero: cubrir cualquier short existente
            if orders_to_flatten_shorts is not None:
                ordenes.extend(orders_to_flatten_shorts(posiciones_actuales))
            for symbol, peso in pesos_optimizados.items():
                precio = precios_actuales.get(symbol)
                if not precio or precio <= 0:
                    continue
                qty_held = float(posiciones_actuales.get(symbol, 0) or 0)
                long_qty = max(0.0, qty_held)
                objetivo_valor = port_value * peso
                valor_actual = long_qty * precio  # solo long cuenta para el target
                diff = objetivo_valor - valor_actual
                if abs(diff) < 10:
                    continue
                qty = int(abs(diff) / precio)
                if qty == 0:
                    continue
                side = 'buy' if diff > 0 else 'sell'
                if clamp_long_only_order is not None:
                    side, qty = clamp_long_only_order(side, qty, long_qty)
                elif side == 'sell':
                    qty = min(qty, int(long_qty))
                if not side or qty <= 0:
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
        pass

    def verificar_stop_loss(self):
        return []


def sistema_completo_alpaca(
    capital_inicial=10000,
    riesgo_max=0.10,
    modo_paper=True,
    ejecutar_ordenes=True,
    monitorear=True,
    tipo_cartera='largo_plazo',
    analizar_cartera=None,
    tickers_completos=None,
    kwargs_analisis=None,
    analizar_cartera_largo_plazo=None,
):
    if not analizar_cartera:
        print("❌ analizar_cartera es obligatorio")
        return None
    trading = TradingAlpaca(modo_paper=modo_paper)
    if not trading.api:
        print("❌ No se pudo conectar con Alpaca. Revisa .env")
        return None

    print("\n📊 Analizando cartera (largo plazo - bajo riesgo)...")
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
    rm = resultados.get('rendimientos_anuales')
    if rm is None:
        rm = resultados.get('rendimientos_mensuales')
    ticker_symbols_optimizados = list(rm.index) if rm is not None else resultados.get('tickers', [])
    tickers_completos = tickers_completos or ticker_symbols_optimizados
    ticker_symbols = sorted(set(tickers_completos))

    peso_min = 0.002
    n = len(ticker_symbols)
    if peso_min * n > 1:
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
        for t in list(pesos_optimizados.keys())[:40]:
            p = trading.obtener_precio_actual(t, mostrar_warnings=False)
            if p:
                precios_actuales[t] = p
        if precios_actuales:
            pesos_optimizados = trading.identificar_top_acciones(
                pesos_optimizados, precios_actuales,
                num_top=CONFIG.get('top_5_numero', 12),
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
    # Crítico: reaplicar max_weight DESPUÉS del Top-N (evita JNJ 43% etc.)
    if aplicar_max_weight_pesos is not None:
        pesos_optimizados = aplicar_max_weight_pesos(pesos_optimizados, CONFIG.get('max_weight', 0.15))

    logger.info(
        "[LOW_RISK] Cartera CORE/largo plazo: %d activos (top %s, max_weight=%.0f%%)",
        len(pesos_optimizados),
        CONFIG.get('top_5_numero', 12),
        CONFIG.get('max_weight', 0.15) * 100,
    )
    for t, p in sorted(pesos_optimizados.items(), key=lambda x: -x[1])[:15]:
        logger.info("  %s: %.2f%%", t, p * 100)

    # Log config de salida y reportes al arrancar
    mini_s = CONFIG.get("mini_monitor_interval_seconds", 60)
    full_min = CONFIG.get("full_report_interval_minutes", 60)
    if CONFIG.get("report_interval_seconds"):
        full_min = CONFIG["report_interval_seconds"] // 60
    logger.info(
        "[LOW_RISK] Config: etiqueta=%s | mini_interval=%ds | full_interval=%d min | stop_loss_pct=%s | trailing_stop_pct=%s",
        CONFIG.get("etiqueta_cartera", "BAJO RIESGO"),
        mini_s,
        full_min,
        CONFIG.get("stop_loss_porcentaje"),
        CONFIG.get("trailing_stop_porcentaje"),
    )

    state_file = CONFIG.get("state_file", "state_low_risk.json")
    state = load_state(state_file) if load_state else {"symbols": {}, "meta": {"last_mini_report_ts": None, "last_full_report_ts": None, "peak_equity": None, "last_orders_since_full": []}}
    # Corregir inconsistencias: max_price_since_entry nunca debe ser < entry_price (rompe trailing/stop al iniciar)
    if normalize_state_positions:
        try:
            n_norm, n_fixed = normalize_state_positions(state)
            if n_norm or n_fixed:
                if save_state:
                    save_state(state_file, state)
                logger.info("[LOW_RISK] [STATE] normalized %s symbols (fixed %s max_price issues)", n_norm, n_fixed)
        except Exception as e:
            logger.warning("[LOW_RISK] normalize_state_positions: %s", e)
    price_cache = {}

    # Risk checks: equity ACTUAL vs peak histórico real (persistido en state)
    peak_equity = None
    orders_in_window = []
    last_trade_per_symbol = {}
    port_value_now = None
    if trading.api:
        try:
            acc = trading.api.get_account()
            port_value_now = float(acc.portfolio_value)
            if resolve_peak_and_port is not None:
                port_value_now, peak_equity = resolve_peak_and_port(state, port_value_now)
            else:
                peak_equity = port_value_now
            if save_state:
                save_state(state_file, state)
        except Exception:
            pass
    if risk_checks is not None and peak_equity is not None and port_value_now is not None:
        ok, reason = risk_checks(port_value_now, peak_equity, orders_in_window, CONFIG)
        if not ok:
            logger.warning("Risk check bloqueó nuevas operaciones: %s", reason)
            ejecutar_ordenes = False

    last_orders_snapshot = []
    if ejecutar_ordenes:
        posiciones_actuales = trading.obtener_todas_posiciones()
        ordenes = trading.rebalancear_cartera(pesos_optimizados, precios_actuales, posiciones_actuales, solo_volumen=True, enviar=False)
        if leftover_exit_orders is not None:
            ordenes.extend(leftover_exit_orders(posiciones_actuales, pesos_optimizados.keys(), OWN_UNIVERSE, FOREIGN_UNIVERSE))
        if sanitize_orders_long_only is not None:
            ordenes = sanitize_orders_long_only(ordenes, posiciones_actuales)
        # Cooldown tras stop: no recomprar hasta cooldown_minutes_after_stop
        if ordenes and should_skip_buy_due_to_cooldown:
            cooldown_min = CONFIG.get("cooldown_minutes_after_stop", 360)
            now_ts = time.time()
            _filtered = []
            for o in ordenes:
                if o.get("side") == "buy" and o.get("reason") != "cover_short" and should_skip_buy_due_to_cooldown(state, o["symbol"], now_ts, cooldown_min):
                    logger.info("[LOW_RISK] SKIP_BUY_COOLDOWN %s", o["symbol"])
                    continue
                _filtered.append(o)
            ordenes = _filtered
        if ordenes and apply_cooldown_and_max_trades is not None:
            ordenes, last_trade_per_symbol, orders_in_window = apply_cooldown_and_max_trades(ordenes, last_trade_per_symbol, orders_in_window, CONFIG, state=state)
        ordenes = _preparar_ordenes_seguras(trading, ordenes, posiciones_actuales, precios_actuales)
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
        etq = CONFIG.get('etiqueta_cartera', 'BAJO RIESGO')
        env_telegram = CONFIG.get('enviar_telegram', False)
        mini_interval = CONFIG.get('mini_monitor_interval_seconds', 60)
        telegram_mini_interval = CONFIG.get('telegram_mini_interval_seconds', 600)
        full_interval_sec = CONFIG["report_interval_seconds"]
        tiene_telegram = bool(CONFIG.get('telegram_bot_token') and CONFIG.get('telegram_chat_id'))
        telegram_params = {'bot_token': CONFIG.get('telegram_bot_token'), 'chat_id': CONFIG.get('telegram_chat_id')}
        print("\n" + "─" * 50)
        print("   📌 CARTERA:", etq, "(core / largo plazo)")
        rebalance_interval = CONFIG.get("rebalance_interval_seconds", 7 * 24 * 3600)
        rebal_txt = f"{rebalance_interval / 86400:.0f}d" if rebalance_interval >= 86400 else f"{rebalance_interval / 3600:.1f}h"
        print("   ⏱️ Stop: cada", mini_interval, "s. — Telegram monitoreo: cada", telegram_mini_interval // 60, "min. — Rebalanceo cada", rebal_txt,
              f"(top {CONFIG.get('top_5_numero', 12)}, max_weight {CONFIG.get('max_weight', 0.15)*100:.0f}%)")
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
                # Durante warm-up: solo actualizar max_price_since_entry; no vender ni evaluar stop
                if in_warmup:
                    _verificar_stop_loss(trading, state, CONFIG, price_cache, state_file, warmup_only=True, allowed_symbols=OWN_UNIVERSE or None, blocked_symbols=FOREIGN_UNIVERSE or None)
                else:
                    if _verificar_stop_loss(trading, state, CONFIG, price_cache, state_file, allowed_symbols=OWN_UNIVERSE or None, blocked_symbols=FOREIGN_UNIVERSE or None):
                        if save_state:
                            save_state(state_file, state)
                # Rebalanceo periódico (semanal en LP). El stop se revisa cada ciclo arriba.
                rebalance_interval = CONFIG.get("rebalance_interval_seconds", 7 * 24 * 3600)
                if not in_warmup and (now_ts - last_rebalance_ts) >= rebalance_interval:
                    last_rebalance_ts = now_ts
                    # Re-chequear drawdown con equity actual vs peak real antes de comprar
                    try:
                        if trading.api:
                            acc = safe_api_call(trading.api.get_account) if safe_api_call else trading.api.get_account()
                            if acc:
                                pv = float(acc.portfolio_value)
                                if resolve_peak_and_port is not None:
                                    pv, peak_equity = resolve_peak_and_port(state, pv)
                                if risk_checks is not None and peak_equity is not None:
                                    ok, reason = risk_checks(pv, peak_equity, orders_in_window, CONFIG)
                                    if not ok:
                                        logger.warning("[LOW_RISK] Rebalance bloqueado por risk: %s", reason)
                                        raise RuntimeError(reason)
                    except RuntimeError:
                        pass
                    else:
                        posiciones_actuales = trading.obtener_todas_posiciones() if trading.api else {}
                        precios_rebalance = {}
                        for t in (pesos_optimizados or {}):
                            p = trading.obtener_precio_actual(t, mostrar_warnings=False)
                            if p and p > 0:
                                precios_rebalance[t] = p
                        if precios_rebalance and pesos_optimizados:
                            ordenes = trading.rebalancear_cartera(pesos_optimizados, precios_rebalance, posiciones_actuales, solo_volumen=True, enviar=False)
                            if leftover_exit_orders is not None:
                                ordenes.extend(leftover_exit_orders(posiciones_actuales, pesos_optimizados.keys(), OWN_UNIVERSE, FOREIGN_UNIVERSE))
                            if sanitize_orders_long_only is not None:
                                ordenes = sanitize_orders_long_only(ordenes, posiciones_actuales)
                            if ordenes and should_skip_buy_due_to_cooldown:
                                cooldown_min = CONFIG.get("cooldown_minutes_after_stop", 360)
                                _f = []
                                for o in ordenes:
                                    if o.get("side") == "buy" and o.get("reason") != "cover_short" and should_skip_buy_due_to_cooldown(state, o["symbol"], now_ts, cooldown_min):
                                        logger.info("[LOW_RISK] SKIP_BUY_COOLDOWN %s", o["symbol"])
                                        continue
                                    _f.append(o)
                                ordenes = _f
                            if ordenes and apply_cooldown_and_max_trades is not None:
                                ordenes, last_trade_per_symbol, orders_in_window = apply_cooldown_and_max_trades(ordenes, last_trade_per_symbol, orders_in_window, CONFIG, state=state)
                            ordenes = _preparar_ordenes_seguras(trading, ordenes, posiciones_actuales, precios_rebalance)
                            if ordenes:
                                for o in ordenes:
                                    try:
                                        trading.api.submit_order(symbol=o['symbol'], qty=o['qty'], side=o['side'], type='market', time_in_force='day')
                                        orders_in_window.append(datetime.now())
                                        last_orders_snapshot.append(o)
                                        logger.info("[LOW_RISK] Rebalance periódico: %s %s %s", o['side'], o['qty'], o['symbol'])
                                    except Exception as e:
                                        logger.warning("Orden rebalance %s: %s", o.get('symbol'), e)
                # Actualizar peak (con safe_api si existe) y persistir
                try:
                    if trading.api:
                        acc = safe_api_call(trading.api.get_account) if safe_api_call else trading.api.get_account()
                        if acc:
                            pv = float(acc.portfolio_value)
                            if resolve_peak_and_port is not None:
                                pv, peak_equity = resolve_peak_and_port(state, pv)
                            else:
                                peak_equity = max(peak_equity or pv, pv)
                            if save_state:
                                save_state(state_file, state)
                except Exception:
                    pass
                # Scheduler: mini cada 300s, full cada 3600s
                last_mini_ts = _state_ts(state, "last_mini_report_ts")
                last_full_ts = _state_ts(state, "last_full_report_ts")
                debe_mini = mini_report_if_due(now_ts, last_mini_ts, telegram_mini_interval) if mini_report_if_due else True
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
                    # Refrescar pesos objetivo (Top-N + max_weight) para el reporte; el trade espera al rebalanceo
                    grafico_path = None
                    grafico_paths = []
                    try:
                        kwargs = dict(capital_inicial=capital_inicial, riesgo_max=riesgo_max, **(kwargs_analisis or {}))
                        kwargs['generar_graficos'] = True
                        resultados_refresh = analizar_cartera(**kwargs)
                        if resultados_refresh:
                            grafico_path = resultados_refresh.get('grafico_path')
                            grafico_paths = resultados_refresh.get('grafico_paths') or []
                            cartera_opt = resultados_refresh['cartera_optimizada']
                            rm = resultados_refresh.get('rendimientos_anuales')
                            if rm is None:
                                rm = resultados_refresh.get('rendimientos_mensuales')
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
                                for t in list(nuevos_pesos.keys())[:40]:
                                    p = trading.obtener_precio_actual(t, mostrar_warnings=False)
                                    if p:
                                        precios_refresh[t] = p
                                if precios_refresh:
                                    nuevos_pesos = trading.identificar_top_acciones(
                                        nuevos_pesos, precios_refresh,
                                        num_top=CONFIG.get('top_5_numero', 12),
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
                                if aplicar_max_weight_pesos is not None:
                                    pesos_optimizados = aplicar_max_weight_pesos(
                                        pesos_optimizados, CONFIG.get('max_weight', 0.15)
                                    )
                                logger.info(
                                    "[LOW_RISK] Pesos actualizados para reporte (%d): %s",
                                    len(pesos_optimizados),
                                    list(pesos_optimizados.keys()),
                                )
                    except Exception as e:
                        logger.warning("[LOW_RISK] No se pudo refrescar análisis para reporte: %s", e)
                    if generate_big_report is not None:
                        generate_big_report(
                            trading, capital_inicial, CONFIG, env_telegram and tiene_telegram,
                            enviar_mensaje_telegram, telegram_params, last_orders_snapshot,
                            pesos_objetivo=pesos_optimizados, peak_equity=peak_equity,
                            image_path=grafico_path, image_paths=grafico_paths,
                            enviar_foto_fn=enviar_foto_telegram,
                        )
                    else:
                        try:
                            account = trading.api.get_account()
                            positions = trading.api.list_positions()
                            cash = float(account.cash)
                            port_value = float(account.portfolio_value)
                            ganancia = port_value - capital_inicial
                            ganancia_pct = (ganancia / capital_inicial * 100) if capital_inicial else 0
                            msg = f"<b>🟢 [{etq}] REPORTE COMPLETO — LARGO PLAZO</b>\n<i>Cada {full_interval_sec//60} min</i>\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n<b>💰 RESUMEN</b>\n  Valor: ${port_value:,.2f}\n  Cash: ${cash:,.2f}\n  Ganancia: ${ganancia:,.2f} ({ganancia_pct:+.2f}%)\n\n<b>📋 POSICIONES</b>\n"
                            for p in sorted(positions, key=lambda x: -float(x.market_value or 0))[:20]:
                                msg += f"  • {p.symbol}: {p.qty} → ${float(p.market_value or 0):,.2f}\n"
                            if env_telegram and tiene_telegram:
                                enviar_mensaje_telegram(CONFIG['telegram_bot_token'], CONFIG['telegram_chat_id'], msg)
                                for i, item in enumerate(grafico_paths or ([] if not grafico_path else [{"path": grafico_path, "caption": "Gráficas"}]), 1):
                                    path = item.get("path") if isinstance(item, dict) else item
                                    cap = item.get("caption", "Gráfica") if isinstance(item, dict) else "Gráfica"
                                    if path:
                                        enviar_foto_telegram(
                                            CONFIG['telegram_bot_token'], CONFIG['telegram_chat_id'], path,
                                            caption=f"📊 [{etq}] {cap} ({i}/{max(len(grafico_paths), 1)})",
                                        )
                            logger.info("[LOW_RISK] Reporte completo enviado.")
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
        print("TRADING ALPACA - CARTERA LARGO PLAZO (BAJO RIESGO)")
        print("⚠️" * 40)
        print("\nPAPER TRADING por defecto. Revisa .env (ALPACA_API_KEY, ALPACA_SECRET_KEY).")
        print("=" * 80)

        trading_test = TradingAlpaca(modo_paper=True)
        if not trading_test.api:
            print("\n❌ No se pudo conectar con Alpaca. Revisa credenciales en .env")
            sys.exit(1)

        CONFIG['usar_top_5_acciones'] = True
        CONFIG['top_5_numero'] = 12
        CONFIG['top_5_criterio'] = 'sharpe'
        CONFIG['max_weight'] = 0.15

        riesgo_max = CONFIG.get('riesgo_max_anual', CONFIG['riesgo_max'])
        print(f"\n🚀 Iniciando cartera: CORE / largo_plazo | riesgo_max ANUAL: {riesgo_max*100:.1f}%")
        print(f"   📌 Top {CONFIG['top_5_numero']} + max_weight {CONFIG['max_weight']*100:.0f}% | rebalanceo semanal | stop cada 60s")
        print("=" * 80)

        resultado = sistema_completo_alpaca(
            capital_inicial=CONFIG['capital_inicial'],
            riesgo_max=riesgo_max,
            modo_paper=CONFIG['modo_paper'],
            ejecutar_ordenes=True,
            monitorear=True,
            tipo_cartera='largo_plazo',
            analizar_cartera=analizar_cartera_largo_plazo,
            tickers_completos=None,
            kwargs_analisis={'estrategia': 'sharpe', 'generar_graficos': True},
            analizar_cartera_largo_plazo=None,
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
