"""
Helpers reutilizables para trading Alpaca (alto y bajo riesgo).
- Estado persistente (load_state, save_state).
- Llamadas API con retry/backoff (safe_api_call).
- Configuración centralizada, risk checks, position sizing.
- Stop-loss y trailing stop (lógica en scripts; helpers dan utilidades).
- Generación de mini reporte y reporte completo.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Configurar logging una vez (nivel INFO por defecto)
_logger = logging.getLogger("alpaca_trading")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)


def _ts_from_value(v: Any) -> Optional[float]:
    """Convierte valor (float, int, ISO str) a unix timestamp float."""
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


def load_state(state_file: str) -> Dict[str, Any]:
    """
    Carga estado persistente desde JSON.
    Estructura: {"symbols": {SYM: {entry_price, max_price_since_entry, last_stop_ts, last_action_ts}}, "meta": {last_mini_report_ts, last_full_report_ts}}.
    Compatible con formato antiguo (positions, cooldown_until) que se migra a symbols+meta.
    """
    default: Dict[str, Any] = {
        "symbols": {},
        "meta": {
            "last_mini_report_ts": None,
            "last_full_report_ts": None,
            "peak_equity": None,
            "last_orders_since_full": [],
        },
    }
    if not state_file:
        return default
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, TypeError) as e:
        _logger.warning("load_state %s: %s", state_file, e)
        return default

    # Ya tiene formato nuevo
    if "symbols" in data and "meta" in data:
        symbols = data.get("symbols") or {}
        meta = data.get("meta") or {}
        for k in default["meta"]:
            if k not in meta:
                meta[k] = default["meta"][k]
        return {"symbols": symbols, "meta": meta}

    # Migrar formato antiguo -> symbols + meta
    symbols: Dict[str, Dict[str, Any]] = {}
    positions = data.get("positions") or {}
    cooldown_until = data.get("cooldown_until") or {}
    for sym, info in positions.items():
        if not isinstance(info, dict):
            continue
        entry = info.get("entry_price")
        if entry is not None:
            try:
                entry = float(entry)
            except (TypeError, ValueError):
                entry = None
        max_p = info.get("max_price_since_entry")
        if max_p is not None:
            try:
                max_p = float(max_p)
            except (TypeError, ValueError):
                max_p = None
        last_stop = cooldown_until.get(sym)
        last_stop_ts = _ts_from_value(last_stop) if last_stop else 0
        symbols[sym] = {
            "entry_price": entry,
            "max_price_since_entry": max_p,
            "last_stop_ts": last_stop_ts,
            "last_action_ts": _ts_from_value(info.get("last_action_ts")) or 0,
        }
    for sym, until in cooldown_until.items():
        if sym not in symbols:
            symbols[sym] = {
                "entry_price": None,
                "max_price_since_entry": None,
                "last_stop_ts": _ts_from_value(until) or 0,
                "last_action_ts": 0,
            }
        else:
            ts = _ts_from_value(until)
            if ts is not None:
                symbols[sym]["last_stop_ts"] = ts

    meta = {
        "last_mini_report_ts": data.get("last_mini_report_ts"),
        "last_full_report_ts": data.get("last_full_report_ts"),
        "peak_equity": data.get("peak_equity"),
        "last_orders_since_full": data.get("last_orders_since_full") or [],
    }
    return {"symbols": symbols, "meta": meta}


def save_state(state_file: str, state: Dict[str, Any]) -> None:
    """Guarda estado en JSON. Estructura: symbols + meta. Serializa timestamps a número o ISO."""
    if not state_file or not state:
        return
    try:
        symbols = state.get("symbols") or {}
        meta = state.get("meta") or {}
        out_symbols: Dict[str, Dict[str, Any]] = {}
        for sym, info in symbols.items():
            if not isinstance(info, dict):
                continue
            out_symbols[sym] = {
                "entry_price": info.get("entry_price"),
                "max_price_since_entry": info.get("max_price_since_entry"),
                "last_stop_ts": info.get("last_stop_ts"),
                "last_action_ts": info.get("last_action_ts"),
            }
        out_meta: Dict[str, Any] = {}
        for k, v in meta.items():
            if k in ("last_mini_report_ts", "last_full_report_ts"):
                out_meta[k] = v
            elif k == "last_orders_since_full":
                out_meta[k] = v[-50:] if isinstance(v, list) else []
            else:
                out_meta[k] = v
        out = {"symbols": out_symbols, "meta": out_meta}
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    except OSError as e:
        _logger.warning("save_state %s: %s", state_file, e)


def safe_api_call(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    backoff_base: float = 1.5,
    **kwargs: Any,
) -> Optional[Any]:
    """Ejecuta fn con reintentos y backoff exponencial. Devuelve resultado o None si falla."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff_base ** attempt)
    _logger.warning("safe_api_call failed after %d retries: %s", max_retries, last_err)
    return None


def compute_drawdown(peak_equity: float, current_equity: float) -> float:
    """Drawdown como fracción 0..1. Si peak<=0 devuelve 0."""
    if peak_equity is None or peak_equity <= 0:
        return 0.0
    if current_equity is None or current_equity < 0:
        return 0.0
    if current_equity >= peak_equity:
        return 0.0
    return (peak_equity - current_equity) / peak_equity


def mini_report_if_due(now_ts: float, last_ts: Optional[float], interval_seconds: float) -> bool:
    """True si debe enviarse mini reporte (nunca enviado o pasaron interval_seconds)."""
    if last_ts is None:
        return True
    return (now_ts - last_ts) >= interval_seconds


def full_report_if_due(now_ts: float, last_ts: Optional[float], interval_seconds: float) -> bool:
    """True si debe enviarse reporte completo."""
    if last_ts is None:
        return True
    return (now_ts - last_ts) >= interval_seconds


def clamp_long_only_order(side: str, qty: int, qty_held: float) -> tuple:
    """
    Long-only: nunca abre ni profundiza un short.
    - sell: solo hasta la cantidad long (>=0) poseída.
    - buy: se permite (cubre shorts o abre long).
    Devuelve (side, qty) o (None, 0) si no hay orden válida.
    """
    try:
        q = int(qty)
    except (TypeError, ValueError):
        return None, 0
    if q <= 0:
        return None, 0
    held = float(qty_held or 0)
    if side == "sell":
        long_qty = int(max(0.0, held))
        q = min(q, long_qty)
        if q <= 0:
            return None, 0
        return "sell", q
    if side == "buy":
        return "buy", q
    return None, 0


def orders_to_flatten_shorts(posiciones_actuales: Dict[str, float]) -> list:
    """Genera buys para cerrar cualquier qty negativa (short)."""
    out = []
    for sym, qty in (posiciones_actuales or {}).items():
        try:
            q = float(qty)
        except (TypeError, ValueError):
            continue
        if q < -1e-9:
            cover = int(abs(q))
            if cover > 0:
                out.append({"symbol": sym, "side": "buy", "qty": cover, "reason": "cover_short"})
    return out


def sanitize_orders_long_only(ordenes, posiciones_actuales: Optional[Dict[str, float]] = None) -> list:
    """Filtra/ajusta órdenes para que ningún sell abra o agrande un short."""
    held: Dict[str, float] = {k: float(v) for k, v in (posiciones_actuales or {}).items()}
    out = []
    for o in ordenes or []:
        if not isinstance(o, dict):
            continue
        sym = o.get("symbol")
        side = o.get("side")
        qty = o.get("qty")
        if not sym or not side:
            continue
        side2, qty2 = clamp_long_only_order(side, qty, held.get(sym, 0.0))
        if not side2 or qty2 <= 0:
            _logger.info("SKIP_SHORT_GUARD %s %s qty=%s held=%s", side, sym, qty, held.get(sym, 0.0))
            continue
        neo = dict(o)
        neo["side"] = side2
        neo["qty"] = qty2
        out.append(neo)
        if side2 == "buy":
            held[sym] = held.get(sym, 0.0) + qty2
        else:
            held[sym] = held.get(sym, 0.0) - qty2
    return out


def sleeve_target_capital(equity: float, allocation_pct: float) -> float:
    """Capital máximo de ESTA cartera dentro de una cuenta compartida."""
    try:
        eq = float(equity or 0)
        pct = float(allocation_pct or 0)
    except (TypeError, ValueError):
        return 0.0
    if eq <= 0 or pct <= 0:
        return 0.0
    return eq * min(pct, 1.0)


def cash_available_for_buys(cash: float, cash_reserve: float = 500.0) -> float:
    """Solo cash real menos reserva. Nunca buying_power/margen."""
    try:
        c = float(cash or 0)
        r = float(cash_reserve or 0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, c - max(0.0, r))


def filter_orders_own_universe(
    ordenes,
    own_universe=None,
    foreign_universe=None,
) -> list:
    """
    Un bot no vende ni compra el universo de la otra cartera.
    cover_short sí se permite (cerrar riesgo).
    """
    own = {str(s).upper() for s in (own_universe or [])}
    foreign = {str(s).upper() for s in (foreign_universe or [])}
    out = []
    for o in ordenes or []:
        if not isinstance(o, dict):
            continue
        sym = str(o.get("symbol") or "").upper()
        side = (o.get("side") or "").lower()
        if o.get("reason") == "cover_short":
            out.append(o)
            continue
        if foreign and sym in foreign:
            _logger.info("SKIP_OTHER_SLEEVE %s %s", side, sym)
            continue
        if own and sym not in own:
            if side == "sell":
                _logger.info("SKIP_SELL_OUTSIDE_UNIVERSE %s", sym)
                continue
        out.append(o)
    return out


def cap_buys_to_cash(ordenes, precios: Optional[Dict[str, float]], cash: float, cash_reserve: float = 500.0) -> list:
    """
    Aplica ventas primero (liberan cash) y luego compras solo si hay cash real.
    Evita margen y cash negativo.
    """
    available = float(cash or 0)
    reserve = float(cash_reserve or 0)
    precios = precios or {}
    sells = [o for o in (ordenes or []) if (o.get("side") or "").lower() == "sell"]
    covers = [o for o in (ordenes or []) if o.get("reason") == "cover_short"]
    buys = [
        o for o in (ordenes or [])
        if (o.get("side") or "").lower() == "buy" and o.get("reason") != "cover_short"
    ]
    out = list(sells)
    for o in sells:
        px = float(precios.get(o.get("symbol"), 0) or 0)
        qty = float(o.get("qty") or 0)
        if px > 0 and qty > 0:
            available += px * qty
    # Cubrir shorts solo con cash libre (si no hay, no agrandar el hueco)
    spendable = available - reserve
    for o in covers:
        px = float(precios.get(o.get("symbol"), 0) or 0)
        qty = int(o.get("qty") or 0)
        if px <= 0 or qty <= 0:
            continue
        max_qty = int(spendable // px) if spendable > 0 else 0
        qty = min(qty, max_qty)
        if qty <= 0:
            _logger.info("SKIP_COVER_NO_CASH %s", o.get("symbol"))
            continue
        neo = dict(o)
        neo["qty"] = qty
        out.append(neo)
        spendable -= px * qty
        available -= px * qty
    spendable = available - reserve
    if spendable <= 0:
        for o in buys:
            _logger.info("SKIP_BUY_NO_CASH %s qty=%s", o.get("symbol"), o.get("qty"))
        return out
    for o in buys:
        px = float(precios.get(o.get("symbol"), 0) or 0)
        qty = int(o.get("qty") or 0)
        if px <= 0 or qty <= 0:
            continue
        max_qty = int(spendable // px)
        if max_qty <= 0:
            _logger.info("SKIP_BUY_NO_CASH %s qty=%s", o.get("symbol"), qty)
            continue
        if qty > max_qty:
            _logger.info("TRIM_BUY %s %s→%s (cash)", o.get("symbol"), qty, max_qty)
            qty = max_qty
        neo = dict(o)
        neo["qty"] = qty
        out.append(neo)
        spendable -= px * qty
    return out


def restore_cash_sells(
    posiciones_actuales: Dict[str, float],
    precios: Dict[str, float],
    cash: float,
    cash_reserve: float,
    own_universe=None,
    foreign_universe=None,
) -> list:
    """Si el cash está bajo/negativo, vende de ESTA manga lo mínimo para volver a reserva."""
    try:
        need = float(cash_reserve or 0) - float(cash or 0)
    except (TypeError, ValueError):
        return []
    if need <= 0:
        return []
    own = {str(s).upper() for s in (own_universe or [])}
    foreign = {str(s).upper() for s in (foreign_universe or [])}
    candidates = []
    for sym, qty in (posiciones_actuales or {}).items():
        s = str(sym).upper()
        q = float(qty or 0)
        px = float((precios or {}).get(sym) or (precios or {}).get(s) or 0)
        if q < 1 or px <= 0:
            continue
        if foreign and s in foreign:
            continue
        if own and s not in own:
            continue
        candidates.append((s, int(q), px, q * px))
    candidates.sort(key=lambda x: -x[3])
    out = []
    remaining = need
    for s, q, px, _mv in candidates:
        qty_sell = min(q, max(1, int((remaining / px) + 0.999)))
        qty_sell = min(qty_sell, q)
        if qty_sell <= 0:
            continue
        out.append({"symbol": s, "side": "sell", "qty": qty_sell, "reason": "restore_cash"})
        remaining -= qty_sell * px
        if remaining <= 0:
            break
    return out


def leftover_exit_orders(
    posiciones_actuales,
    top_symbols,
    own_universe=None,
    foreign_universe=None,
) -> list:
    """Vende solo leftovers de ESTA manga; cubre shorts de cualquiera."""
    top = {str(s).upper() for s in (top_symbols or [])}
    own = {str(s).upper() for s in (own_universe or [])}
    foreign = {str(s).upper() for s in (foreign_universe or [])}
    out = []
    for sym, qty in (posiciones_actuales or {}).items():
        s = str(sym).upper()
        try:
            q = float(qty or 0)
        except (TypeError, ValueError):
            continue
        if s in top:
            continue
        if q < -1e-9:
            cover = int(abs(q))
            if cover > 0:
                out.append({"symbol": s, "side": "buy", "qty": cover, "reason": "cover_short"})
            continue
        if q < 1:
            continue
        if foreign and s in foreign:
            continue
        if own and s not in own:
            continue
        out.append({"symbol": s, "side": "sell", "qty": int(q), "reason": "exit_not_in_top"})
    return out


def finalize_orders_shared_account(
    ordenes,
    posiciones_actuales,
    precios,
    cash,
    own_universe=None,
    foreign_universe=None,
    cash_reserve=500.0,
    restore_cash=True,
) -> list:
    """Filtro final: no tocar la otra manga, recuperar cash y no comprar con margen."""
    ordenes = list(ordenes or [])
    if restore_cash:
        ordenes.extend(
            restore_cash_sells(
                posiciones_actuales or {},
                precios or {},
                cash,
                cash_reserve,
                own_universe=own_universe,
                foreign_universe=foreign_universe,
            )
        )
    ordenes = filter_orders_own_universe(ordenes, own_universe, foreign_universe)
    ordenes = sanitize_orders_long_only(ordenes, posiciones_actuales)
    ordenes = cap_buys_to_cash(ordenes, precios, cash, cash_reserve)
    return sanitize_orders_long_only(ordenes, posiciones_actuales)


def get_portfolio_config(tipo_cartera, env_config=None):
    """
    Configuración centralizada por cartera. env_config puede sobreescribir (ej. desde CONFIG del script).

    Arquitectura de riesgo (v2):
    - CORE / LARGO PLAZO: stop cada 60s, rebalanceo semanal, 8–15 posiciones, max_weight 15%.
    - ALTO RIESGO: stop cada 60s, rebalanceo cada 2h, 4–8 posiciones, max_weight 35%.
    """
    if tipo_cartera == "largo_plazo":
        base = {
            "report_interval_seconds": 3600,
            "mini_monitor_interval_seconds": 60,  # stop / loop
            "telegram_mini_interval_seconds": 600,  # monitoreo Telegram cada 10 min
            "rebalance_interval_seconds": 7 * 24 * 3600,  # semanal
            "max_risk_per_trade": 0.005,
            "rr_ratio": 2.0,
            "cooldown_seconds": 3600,
            "cooldown_after_stop_seconds": 21600,
            "cooldown_minutes_after_stop": 360,
            "max_drawdown": 0.12,
            "max_trades_per_window": 5,
            "window_seconds": 86400,
            "stop_loss_porcentaje": 0.08,
            "trailing_stop_porcentaje": 0.10,  # trailing amplio / opcional en LP
            "etiqueta_cartera": "CORE / LARGO PLAZO",
            "state_file": "state_low_risk.json",
            "warmup_seconds": 180,
            "max_weight": 0.15,
            "top_n_posiciones": 12,
            "usar_filtro_top": True,
            "top_criterio": "sharpe",
            "allocation_pct": 0.50,
            "cash_reserve": 500.0,
            "cash_only": True,
        }
    else:
        base = {
            "report_interval_seconds": 1800,
            "mini_monitor_interval_seconds": 60,
            "telegram_mini_interval_seconds": 600,
            "rebalance_interval_seconds": 2 * 3600,  # 2 horas
            "max_risk_per_trade": 0.01,
            "rr_ratio": 1.5,
            "cooldown_seconds": 600,
            "cooldown_after_stop_seconds": 1800,
            "cooldown_minutes_after_stop": 30,
            "max_drawdown": 0.20,
            "max_trades_per_window": 8,
            "window_seconds": 3600,
            "stop_loss_porcentaje": 0.10,
            "trailing_stop_porcentaje": 0.08,
            "etiqueta_cartera": "ALTO RIESGO",
            "state_file": "state_high_risk.json",
            "warmup_seconds": 180,
            "max_weight": 0.35,
            "top_n_posiciones": 6,
            "usar_filtro_top": True,
            "top_criterio": "sharpe",
            "allocation_pct": 0.50,
            "cash_reserve": 500.0,
            "cash_only": True,
        }
    if env_config:
        for k, v in env_config.items():
            if k in base and v is not None:
                base[k] = v
    return base


def aplicar_max_weight_pesos(
    pesos: Dict[str, float],
    max_weight: float = 0.15,
    max_iter: int = 25,
) -> Dict[str, float]:
    """
    Reaplica límite de concentración DESPUÉS de Top-N / renormalización.
    Evita que un Top-4 convierta un 15% teórico en 40%+.
    """
    if not pesos:
        return pesos
    if max_weight is None or max_weight <= 0:
        total = sum(max(0.0, float(v)) for v in pesos.values())
        if total <= 0:
            return pesos
        return {k: max(0.0, float(v)) / total for k, v in pesos.items()}

    w = {k: max(0.0, float(v)) for k, v in pesos.items()}
    n = len(w)
    # Si es imposible (n * max_weight < 1), relajar al mínimo factible
    if n > 0 and n * max_weight < 1.0 - 1e-12:
        max_weight = 1.0 / n

    for _ in range(max_iter):
        total = sum(w.values())
        if total <= 0:
            return {k: 1.0 / n for k in w}
        w = {k: v / total for k, v in w.items()}
        if all(v <= max_weight + 1e-12 for v in w.values()):
            return w
        excess = 0.0
        free = []
        for k, v in list(w.items()):
            if v > max_weight:
                excess += v - max_weight
                w[k] = max_weight
            else:
                free.append(k)
        if excess <= 0 or not free:
            break
        free_total = sum(w[k] for k in free)
        if free_total <= 0:
            add = excess / len(free)
            for k in free:
                w[k] += add
        else:
            for k in free:
                w[k] += excess * (w[k] / free_total)

    total = sum(w.values())
    return {k: v / total for k, v in w.items()} if total > 0 else w


def resolve_peak_and_port(state: Dict[str, Any], port_value: float) -> Tuple[float, float]:
    """
    Devuelve (port_value, peak_equity) actualizando el pico en state['meta'].
    El peak NUNCA se reinicia al equity actual: se carga del estado y solo sube.
    """
    meta = state.setdefault("meta", {}) if state is not None else {}
    peak = meta.get("peak_equity")
    try:
        peak_f = float(peak) if peak is not None else None
    except (TypeError, ValueError):
        peak_f = None
    pv = float(port_value)
    if peak_f is None or peak_f <= 0:
        peak_f = pv
    else:
        peak_f = max(peak_f, pv)
    meta["peak_equity"] = peak_f
    return pv, peak_f


def risk_checks(port_value, peak_equity, orders_in_window, config):
    """
    Verifica si está permitido abrir nuevas operaciones.
    - max_drawdown: bloquea si (peak - current) / peak > max_drawdown.
    - max_trades_per_window: bloquea si len(orders_in_window) >= max_trades_per_window.
    IMPORTANTE: port_value debe ser el equity ACTUAL y peak_equity el pico histórico real.
    Returns (ok_to_trade: bool, reason: str).
    """
    max_dd = config.get("max_drawdown", 0.10)
    max_trades = config.get("max_trades_per_window", 3)
    window_sec = config.get("window_seconds", 86400)
    now = datetime.now()
    cutoff = now - timedelta(seconds=window_sec)
    recent = [t for t in (orders_in_window or []) if t > cutoff]
    if len(recent) >= max_trades:
        return False, f"max_trades_per_window alcanzado ({len(recent)} >= {max_trades})"
    if peak_equity and peak_equity > 0 and port_value is not None:
        dd = compute_drawdown(float(peak_equity), float(port_value))
        if dd >= max_dd:
            return False, f"max_drawdown alcanzado ({dd*100:.1f}% >= {max_dd*100:.0f}%)"
    return True, "ok"


def position_size_from_risk(equity, precio, stop_loss_pct, risk_pct):
    """
    Tamaño de posición (en unidades) basado en riesgo por trade.
    risk_amount = equity * risk_pct; distance_per_share = precio * stop_loss_pct;
    qty = risk_amount / distance_per_share. Redondea a int.
    """
    if not precio or precio <= 0 or not stop_loss_pct or not risk_pct:
        return None
    risk_amount = equity * risk_pct
    distance_per_share = precio * stop_loss_pct
    if distance_per_share <= 0:
        return None
    qty = risk_amount / distance_per_share
    return max(0, int(qty))


def generate_mini_report(
    trading,
    config,
    send_telegram,
    enviar_telegram_fn=None,
    telegram_params=None,
    peak_equity: Optional[float] = None,
    data_stale: bool = False,
):
    """
    Solo monitoreo: timestamp, equity, cash, num posiciones, top 3 por valor, drawdown.
    NO ejecuta órdenes. Si data_stale=True indica DATA_STALE (precios cacheados).
    """
    if not trading or not trading.api:
        return
    try:
        account = trading.api.get_account()
        positions = trading.api.list_positions()
        cash = float(account.cash)
        port_value = float(account.portfolio_value)
        n_pos = len(positions)
        etq = config.get("etiqueta_cartera", "CARTERA")
        dd = compute_drawdown(peak_equity or port_value, port_value) if (peak_equity is not None and peak_equity > 0) else 0.0
        top3 = sorted(positions, key=lambda x: -float(x.market_value or 0))[:3]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stale_tag = " [DATA_STALE]" if data_stale else ""
        _logger.info(
            "[%s] Mini reporte%s | Equity: $%.2f | Cash: $%.2f | Pos: %d | DD: %.1f%%",
            etq, stale_tag, port_value, cash, n_pos, dd * 100,
        )
        if send_telegram and enviar_telegram_fn and telegram_params:
            msg = "<b>📊 [%s] Monitoreo%s</b>\n🕐 %s\n💰 Equity: $%s\n💵 Cash: $%s\n📋 Posiciones: %d\n📉 Drawdown: %.1f%%\n<b>Top 3</b>\n" % (
                etq,
                " [DATA_STALE]" if data_stale else "",
                ts,
                f"{port_value:,.2f}",
                f"{cash:,.2f}",
                n_pos,
                dd * 100,
            )
            for p in top3:
                msg += "  • %s: %s → $%s\n" % (p.symbol, p.qty, f"{float(p.market_value or 0):,.2f}")
            enviar_telegram_fn(
                telegram_params.get("bot_token"),
                telegram_params.get("chat_id"),
                msg,
            )
    except Exception as e:
        _logger.warning("generate_mini_report: %s", e)


def generate_big_report(
    trading,
    capital_inicial,
    config,
    send_telegram,
    enviar_telegram_fn=None,
    telegram_params=None,
    last_orders=None,
    pesos_objetivo=None,
    peak_equity: Optional[float] = None,
    image_path: Optional[str] = None,
    image_paths=None,
    enviar_foto_fn=None,
):
    """
    Reporte completo: resumen, PnL, posiciones, pesos actual vs objetivo, señales resumidas,
    exposición/riesgo, acciones desde último reporte. Envía cada gráfica por separado a Telegram.
    NO ejecuta órdenes.
    """
    if not trading or not trading.api:
        return
    try:
        account = trading.api.get_account()
        positions = trading.api.list_positions()
        cash = float(account.cash)
        port_value = float(account.portfolio_value)
        ganancia = port_value - capital_inicial
        ganancia_pct = (ganancia / capital_inicial * 100) if capital_inicial and capital_inicial != 0 else 0
        etq = config.get("etiqueta_cartera", "CARTERA")
        full_min = config.get("report_interval_seconds", 3600) // 60
        dd = compute_drawdown(peak_equity or port_value, port_value) if (peak_equity and peak_equity > 0) else 0.0
        ahora = datetime.now()
        msg = "<b>📋 [%s] REPORTE COMPLETO</b>\n<i>Cada %d min</i>\n🕐 %s\n\n<b>💰 RESUMEN</b>\n  Valor: $%s\n  Cash: $%s\n  Ganancia: $%s (%+.2f%%)\n  Drawdown: %.1f%%\n\n<b>📋 POSICIONES</b>\n" % (
            etq,
            full_min,
            ahora.strftime("%Y-%m-%d %H:%M:%S"),
            f"{port_value:,.2f}",
            f"{cash:,.2f}",
            f"{ganancia:,.2f}",
            ganancia_pct,
            dd * 100,
        )
        for p in sorted(positions, key=lambda x: -float(x.market_value or 0))[:20]:
            msg += "  • %s: %s → $%s\n" % (p.symbol, p.qty, f"{float(p.market_value or 0):,.2f}")
        if len(positions) > 20:
            msg += "  ... +%d más\n" % (len(positions) - 20)
        if pesos_objetivo:
            msg += "\n<b>🎯 Pesos objetivo (top 5)</b>\n"
            for sym, w in sorted(pesos_objetivo.items(), key=lambda x: -x[1])[:5]:
                msg += "  %s: %.1f%%\n" % (sym, w * 100)
        msg += "\n<b>📤 Acciones desde último reporte</b>\n"
        if last_orders:
            for o in last_orders[-10:]:
                msg += "  %s %s %s\n" % (o.get("side", ""), o.get("qty", ""), o.get("symbol", ""))
        else:
            msg += "  (ninguna)\n"
        _logger.info("[%s] Reporte completo generado.", etq)
        if send_telegram and enviar_telegram_fn and telegram_params:
            bot_token = telegram_params.get("bot_token")
            chat_id = telegram_params.get("chat_id")
            enviar_telegram_fn(bot_token, chat_id, msg)
            _logger.info("[%s] Reporte completo enviado a Telegram.", etq)
            fotos = []
            if image_paths:
                for item in image_paths:
                    if isinstance(item, dict):
                        fotos.append(item)
                    elif isinstance(item, (list, tuple)) and item:
                        fotos.append({"path": item[0], "caption": item[1] if len(item) > 1 else ""})
                    elif isinstance(item, str):
                        fotos.append({"path": item, "caption": ""})
            elif image_path:
                if isinstance(image_path, (list, tuple)):
                    for item in image_path:
                        if isinstance(item, dict):
                            fotos.append(item)
                        elif isinstance(item, str):
                            fotos.append({"path": item, "caption": ""})
                else:
                    fotos.append({"path": image_path, "caption": "Gráficas de cartera"})
            enviadas = 0
            for i, item in enumerate(fotos, 1):
                path = item.get("path")
                caption_txt = (item.get("caption") or "Gráfica").strip()
                if not path or not enviar_foto_fn:
                    continue
                if not os.path.isfile(path):
                    _logger.warning("[%s] Gráfica no encontrada: %s", etq, path)
                    continue
                caption = "📊 [%s] %s (%d/%d)" % (etq, caption_txt, i, len(fotos))
                ok = enviar_foto_fn(bot_token, chat_id, path, caption=caption)
                if ok:
                    enviadas += 1
                    _logger.info("[%s] Gráfica enviada a Telegram: %s", etq, path)
                else:
                    _logger.warning("[%s] No se pudo enviar gráfica a Telegram: %s", etq, path)
                time.sleep(0.4)
            if fotos and enviadas == 0 and enviar_foto_fn is None:
                _logger.warning("[%s] Hay gráficas pero no hay enviar_foto_fn", etq)
    except Exception as e:
        _logger.warning("generate_big_report: %s", e)


def should_skip_buy_due_to_cooldown(
    state: Dict[str, Any],
    symbol: str,
    now_ts: float,
    cooldown_minutes: int,
) -> bool:
    """
    True si no debe comprarse por cooldown tras un stop (last_stop_ts + cooldown_minutes).
    """
    symbols = state.get("symbols") or {}
    sym_info = symbols.get(symbol)
    if not sym_info:
        return False
    last_stop = sym_info.get("last_stop_ts")
    if last_stop is None or last_stop == 0:
        return False
    last_ts = _ts_from_value(last_stop) if not isinstance(last_stop, (int, float)) else float(last_stop)
    return (now_ts - last_ts) < (cooldown_minutes * 60)


def normalize_state_positions(state: Dict[str, Any]) -> Tuple[int, int]:
    """
    Corrige inconsistencias en state["symbols"] para que stop/trailing no disparen mal al iniciar.
    Por qué se corrige max_price_since_entry: el "máximo desde entrada" debe ser >= entry_price por definición.
    Si max_price_since_entry < entry_price (p. ej. por migración, corrupción o bug), el trigger_trail quedaría
    por debajo del entry y podría disparar ventas incorrectas o comportamientos inconsistentes al arrancar.
    - Si falta max_price_since_entry -> se setea a entry_price.
    - Si max_price_since_entry < entry_price -> se corrige a entry_price.
    - Si faltan last_stop_ts o last_action_ts -> se setean a 0.
    No lanza excepciones; tolerante a datos faltantes o inválidos.
    Returns (num_symbols_normalized, num_max_price_fixed).

    # Prueba manual (ejemplo comentado):
    # state = {"symbols": {"AAPL": {"entry_price": 150.0, "max_price_since_entry": 140.0, "last_stop_ts": 0, "last_action_ts": 0}}}
    # n_norm, n_fixed = normalize_state_positions(state)
    # assert state["symbols"]["AAPL"]["max_price_since_entry"] == 150.0  # corregido a entry_price
    # assert n_fixed == 1
    """
    num_normalized = 0
    num_max_fixed = 0
    try:
        symbols = state.get("symbols")
        if not isinstance(symbols, dict):
            return 0, 0
        for sym, info in list(symbols.items()):
            if not isinstance(info, dict):
                continue
            try:
                entry = info.get("entry_price")
                if entry is not None:
                    try:
                        entry = float(entry)
                    except (TypeError, ValueError):
                        entry = None
                if entry is None or not isinstance(entry, (int, float)) or entry <= 0:
                    continue  # Sin entry_price válido: dejar intacto
                max_p = info.get("max_price_since_entry")
                if max_p is not None:
                    try:
                        max_p = float(max_p)
                    except (TypeError, ValueError):
                        max_p = None
                if max_p is None:
                    info["max_price_since_entry"] = entry
                    num_normalized += 1
                    num_max_fixed += 1
                else:
                    if max_p < entry:
                        info["max_price_since_entry"] = entry
                        num_max_fixed += 1
                    num_normalized += 1
                if info.get("last_stop_ts") is None:
                    info["last_stop_ts"] = 0
                if info.get("last_action_ts") is None:
                    info["last_action_ts"] = 0
            except Exception:
                continue
    except Exception:
        pass
    return num_normalized, num_max_fixed


def update_trailing_state_for_position(
    state: Dict[str, Any],
    symbol: str,
    entry_price: Optional[float],
    current_price: float,
    now_ts: float,
) -> Dict[str, Any]:
    """
    Actualiza state.symbols[symbol]: entry_price (si no existe), max_price_since_entry, last_action_ts.
    No sobreescribe entry_price si ya existe (posición ya registrada).
    Devuelve el dict actualizado del símbolo.
    """
    state.setdefault("symbols", {})
    state["symbols"].setdefault(symbol, {
        "entry_price": None,
        "max_price_since_entry": None,
        "last_stop_ts": 0,
        "last_action_ts": 0,
    })
    info = state["symbols"][symbol]
    if entry_price is not None and (info.get("entry_price") is None or info.get("entry_price") == 0):
        info["entry_price"] = entry_price
    prev_max = info.get("max_price_since_entry")
    if prev_max is not None:
        try:
            prev_max = float(prev_max)
        except (TypeError, ValueError):
            prev_max = current_price
    else:
        prev_max = current_price
    info["max_price_since_entry"] = max(prev_max, current_price)
    info["last_action_ts"] = now_ts
    return info


def is_symbol_in_cooldown(state: Dict[str, Any], symbol: str, config: Dict[str, Any]) -> bool:
    """True si el símbolo está en cooldown (ej. vendido por stop) y no debe recomprarse aún."""
    # Nuevo formato: last_stop_ts + cooldown_minutes_after_stop
    if "symbols" in state:
        cooldown_min = config.get("cooldown_minutes_after_stop", 360)
        return should_skip_buy_due_to_cooldown(state, symbol, time.time(), cooldown_min)
    # Formato antiguo: cooldown_until
    cooldown_until = state.get("cooldown_until") or {}
    until = cooldown_until.get(symbol)
    if until is None:
        return False
    until_ts = _ts_from_value(until)
    if until_ts is None:
        return False
    return time.time() < until_ts


def check_stop_and_trailing(
    entry_price: float,
    max_price_since_entry: Optional[float],
    current_price: float,
    stop_loss_pct: float,
    trailing_stop_pct: Optional[float] = None,
) -> Optional[Tuple[str, float, float]]:
    """
    TRAILING + FALLBACK A STOP FIJO.
    - Regla 1 (siempre): STOP FIJO. trigger_fijo = entry_price * (1 - stop_loss_pct).
      Si current_price <= trigger_fijo => ("STOP_LOSS_FIJO", trigger_fijo, stop_loss_pct).
    - Regla 2 (solo si trailing_stop_pct está definido): TRAILING STOP.
      trigger_trail = max_price_since_entry * (1 - trailing_stop_pct).
      Si current_price <= trigger_trail => ("TRAILING_STOP", trigger_trail, trailing_stop_pct).
    - Si trailing_stop_pct is None: no se evalúa trailing; solo existe stop fijo.
    """
    if entry_price is None or entry_price <= 0 or current_price is None or current_price <= 0:
        return None
    # Regla 1: stop fijo (protección dura, siempre)
    trigger_fijo = entry_price * (1.0 - stop_loss_pct)
    if current_price <= trigger_fijo:
        return ("STOP_LOSS_FIJO", trigger_fijo, stop_loss_pct)
    # Sin trailing configurado: solo stop fijo (ya evaluado)
    if trailing_stop_pct is None:
        return None
    # Regla 2: trailing stop (solo si trailing_stop_pct está definido)
    max_p = max_price_since_entry if (max_price_since_entry is not None and max_price_since_entry > 0) else entry_price
    trigger_trail = max_p * (1.0 - trailing_stop_pct)
    if current_price <= trigger_trail:
        return ("TRAILING_STOP", trigger_trail, trailing_stop_pct)
    return None


def apply_cooldown_and_max_trades(ordenes_candidatas, last_trade_per_symbol, orders_in_window, config, state=None):
    """
    Filtra órdenes: respeta cooldown por símbolo y max_trades_per_window.
    ordenes_candidatas: list of dict with 'symbol', 'side', 'qty'.
    last_trade_per_symbol: dict symbol -> datetime.
    orders_in_window: list of datetime (cuando se enviaron órdenes).
    Returns (filtered_orders, updated_last_trade, updated_orders_in_window).
    """
    cooldown = config.get("cooldown_seconds", 600)
    max_trades = config.get("max_trades_per_window", 6)
    window_sec = config.get("window_seconds", 3600)
    now = datetime.now()
    cutoff = now - timedelta(seconds=window_sec)
    recent = [t for t in orders_in_window if t > cutoff]
    out = []
    last_new = dict(last_trade_per_symbol) if last_trade_per_symbol else {}
    new_times = list(recent)
    for o in ordenes_candidatas:
        if len(new_times) >= max_trades:
            break
        sym = o.get("symbol")
        if not sym:
            continue
        if state and is_symbol_in_cooldown(state, sym, config):
            continue
        last = last_new.get(sym)
        if last and (now - last).total_seconds() < cooldown:
            continue
        out.append(o)
        last_new[sym] = now
        new_times.append(now)
    return out, last_new, new_times


# ---------------------------------------------------------------------------
# Prueba manual (ejemplo comentado): state con max_price_since_entry < entry_price
# ---------------------------------------------------------------------------
# state_sample = {
#     "symbols": {
#         "AAPL": {"entry_price": 150.0, "max_price_since_entry": 140.0, "last_stop_ts": 0, "last_action_ts": 0},
#         "MSFT": {"entry_price": 380.0, "max_price_since_entry": None, "last_stop_ts": None, "last_action_ts": None},
#     },
#     "meta": {},
# }
# n_norm, n_fixed = normalize_state_positions(state_sample)
# # Después: AAPL["max_price_since_entry"] == 150.0 (corregido); MSFT["max_price_since_entry"] == 380.0; last_stop_ts/last_action_ts == 0
# assert state_sample["symbols"]["AAPL"]["max_price_since_entry"] == 150.0
# assert state_sample["symbols"]["MSFT"]["max_price_since_entry"] == 380.0
# assert n_fixed >= 1
