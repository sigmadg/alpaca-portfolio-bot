#!/usr/bin/env python3
"""
Validación manual rápida: Trailing + fallback a Stop fijo y scheduler de reportes.
Ejecutar: python3 validar_trailing_y_reportes.py
"""
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from alpaca_trading_helpers import (
    check_stop_and_trailing,
    get_portfolio_config,
    should_skip_buy_due_to_cooldown,
    update_trailing_state_for_position,
    sleeve_target_capital,
    cap_buys_to_cash,
    filter_orders_own_universe,
    leftover_exit_orders,
    restore_cash_sells,
    finalize_orders_shared_account,
)


def test_caso_a():
    """Caso A: precio cae 5% desde entry => vende por STOP_LOSS_FIJO."""
    entry, stop_pct = 100.0, 0.05
    current = 94.0  # -6% desde entry
    r = check_stop_and_trailing(entry, 100.0, current, stop_pct, trailing_stop_pct=0.05)
    assert r is not None, "Debe activar stop"
    reason, trigger, pct = r
    assert reason == "STOP_LOSS_FIJO", f"Esperado STOP_LOSS_FIJO, obtuvo {reason}"
    assert abs(trigger - 95.0) < 1e-6, f"trigger_fijo esperado 95.0, obtuvo {trigger}"
    print("  Caso A OK: precio cae 5% desde entry => STOP_LOSS_FIJO")


def test_caso_b():
    """Caso B: precio sube, marca max; luego cae 5% desde max => vende por TRAILING_STOP (si trailing activo)."""
    entry, stop_pct, trail_pct = 100.0, 0.05, 0.05
    max_since = 110.0
    current = 104.0  # 110 * 0.95 = 104.5, 104 <= 104.5
    r = check_stop_and_trailing(entry, max_since, current, stop_pct, trailing_stop_pct=trail_pct)
    assert r is not None, "Debe activar trailing"
    reason, trigger, pct = r
    assert reason == "TRAILING_STOP", f"Esperado TRAILING_STOP, obtuvo {reason}"
    assert abs(trigger - 104.5) < 1e-6, f"trigger_trail esperado 104.5, obtuvo {trigger}"
    print("  Caso B OK: sube luego cae 5% desde max => TRAILING_STOP")


def test_caso_c():
    """Caso C: trailing es None => nunca vende por trailing, solo por fijo."""
    entry, stop_pct = 100.0, 0.05
    max_since = 110.0
    current = 104.0  # por debajo de trigger_trail pero por encima de trigger_fijo (95)
    r = check_stop_and_trailing(entry, max_since, current, stop_pct, trailing_stop_pct=None)
    assert r is None, "Con trailing=None no debe activar trailing; 104 > 95 (fijo)"
    current_fijo = 94.0
    r2 = check_stop_and_trailing(entry, max_since, current_fijo, stop_pct, trailing_stop_pct=None)
    assert r2 is not None and r2[0] == "STOP_LOSS_FIJO", "Debe activar solo stop fijo"
    print("  Caso C OK: trailing=None => solo STOP_LOSS_FIJO cuando corresponde")


def test_caso_d_cooldown():
    """Caso D: después de vender por stop, buy del mismo símbolo bloqueado hasta cooldown."""
    state = {"symbols": {"AAPL": {"last_stop_ts": 1000.0, "last_action_ts": 1000.0}}}
    now_ts = 1000.0 + 60 * 10  # 10 min después
    # Cooldown 30 min => 10 min < 30 min => skip
    assert should_skip_buy_due_to_cooldown(state, "AAPL", now_ts, 30) is True
    now_ts_after = 1000.0 + 60 * 40  # 40 min después
    assert should_skip_buy_due_to_cooldown(state, "AAPL", now_ts_after, 30) is False
    # Cooldown 360 min (largo plazo)
    assert should_skip_buy_due_to_cooldown(state, "AAPL", 1000.0 + 60 * 60, 360) is True
    print("  Caso D OK: cooldown bloquea buy hasta pasar cooldown_minutes_after_stop")


def test_intervalos():
    """Intervalos v2: stop/mini 60s; LP rebalanceo semanal; HR rebalanceo 2h."""
    low = get_portfolio_config("largo_plazo", None)
    high = get_portfolio_config("alto_riesgo", None)
    assert low["report_interval_seconds"] == 3600, "Largo plazo: report_interval_seconds debe ser 3600"
    assert low["mini_monitor_interval_seconds"] == 60, "Largo plazo: mini debe ser 60"
    assert low["telegram_mini_interval_seconds"] == 600, "Largo plazo: Telegram monitoreo cada 10 min"
    assert high["report_interval_seconds"] == 1800, "Alto riesgo: report_interval_seconds debe ser 1800"
    assert high["mini_monitor_interval_seconds"] == 60, "Alto riesgo: mini debe ser 60"
    assert high["telegram_mini_interval_seconds"] == 600, "Alto riesgo: Telegram monitoreo cada 10 min"
    assert low["cooldown_minutes_after_stop"] == 360, "Largo plazo: cooldown 360 min"
    assert high["cooldown_minutes_after_stop"] == 30, "Alto riesgo: cooldown 30 min"
    assert low["rebalance_interval_seconds"] == 7 * 24 * 3600, "LP: rebalanceo semanal"
    assert high["rebalance_interval_seconds"] == 2 * 3600, "HR: rebalanceo cada 2h"
    assert low["max_weight"] == 0.15
    assert high["max_weight"] == 0.35
    assert low["etiqueta_cartera"] == "CORE / LARGO PLAZO"
    print("  Intervalos OK: LP stop 60s / rebalance 7d; HR stop 60s / rebalance 2h; max_weight 15%/35%")


def test_update_trailing_state():
    """update_trailing_state_for_position actualiza max_price_since_entry y entry_price si no existe."""
    state = {"symbols": {}}
    update_trailing_state_for_position(state, "AAPL", 100.0, 105.0, 1000.0)
    assert state["symbols"]["AAPL"]["entry_price"] == 100.0
    assert state["symbols"]["AAPL"]["max_price_since_entry"] == 105.0
    update_trailing_state_for_position(state, "AAPL", 100.0, 108.0, 1001.0)
    assert state["symbols"]["AAPL"]["max_price_since_entry"] == 108.0
    assert state["symbols"]["AAPL"]["entry_price"] == 100.0  # no sobreescribe
    print("  update_trailing_state_for_position OK")


def test_max_weight_post_top():
    """max_weight se reaplica tras Top-N (evita concentraciones tipo JNJ 43%)."""
    from alpaca_trading_helpers import aplicar_max_weight_pesos
    # Escenario realista CORE: ~12 nombres tras filtro, uno hiperconcentrado por renormalización
    pesos = {
        "JNJ": 0.434, "ASML": 0.12, "XOM": 0.10, "TSM": 0.08,
        "KO": 0.05, "PG": 0.04, "V": 0.04, "JPM": 0.03,
        "MSFT": 0.03, "AAPL": 0.03, "GLD": 0.02, "BND": 0.022,
    }
    capped = aplicar_max_weight_pesos(pesos, max_weight=0.15)
    assert abs(sum(capped.values()) - 1.0) < 1e-9
    assert all(v <= 0.15 + 1e-9 for v in capped.values()), capped
    assert capped["JNJ"] <= 0.15 + 1e-9
    print("  max_weight post Top-N OK: ningún peso > 15%")


def test_risk_checks_drawdown_real():
    """risk_checks usa equity actual vs peak real (no peak==port)."""
    from alpaca_trading_helpers import risk_checks
    cfg = {"max_drawdown": 0.10, "max_trades_per_window": 99, "window_seconds": 86400}
    ok, reason = risk_checks(90.0, 100.0, [], cfg)
    assert ok is False and "max_drawdown" in reason
    ok2, _ = risk_checks(100.0, 100.0, [], cfg)
    assert ok2 is True
    ok3, _ = risk_checks(95.0, 100.0, [], cfg)
    assert ok3 is True  # 5% < 10%
    print("  risk_checks drawdown real OK")


def test_alto_riesgo_sin_100pct():
    """optimizar_cartera_maximo_rendimiento ya no asigna 100% a un solo activo."""
    import numpy as np
    from cartera_alto_riesgo_corto_plazo import optimizar_cartera_maximo_rendimiento
    n = 5
    C = np.eye(n) * 0.04
    m = np.array([0.01, 0.05, 0.02, 0.03, 0.015])
    r = optimizar_cartera_maximo_rendimiento(C, m, max_weight=0.35)
    assert abs(r["pesos"].sum() - 1.0) < 1e-9
    assert np.all(r["pesos"] <= 0.35 + 1e-9), r["pesos"]
    assert np.count_nonzero(r["pesos"] > 1e-6) >= 2
    print("  alto riesgo sin winner-takes-all OK")


def test_cuenta_compartida_cash_y_mangas():
    """Dos bots en una cuenta: 50/50, no margen, no vender la otra manga."""
    assert abs(sleeve_target_capital(50000, 0.50) - 25000) < 1e-9
    assert sleeve_target_capital(0, 0.50) == 0.0
    core = {"JNJ", "KO", "XOM"}
    hr = {"AMD", "CRWD", "SNOW"}
    leftover = leftover_exit_orders(
        {"AMD": 25, "JNJ": 10, "CRWD": 5},
        top_symbols={"CRWD"},
        own_universe=core,
        foreign_universe=hr,
    )
    assert leftover == [{"symbol": "JNJ", "side": "sell", "qty": 10, "reason": "exit_not_in_top"}]
    leftover_hr = leftover_exit_orders(
        {"AMD": 25, "JNJ": 10, "CRWD": 5},
        top_symbols={"CRWD"},
        own_universe=hr,
        foreign_universe=core,
    )
    assert leftover_hr == [{"symbol": "AMD", "side": "sell", "qty": 25, "reason": "exit_not_in_top"}]
    filtered = filter_orders_own_universe(
        [{"symbol": "JNJ", "side": "sell", "qty": 10}, {"symbol": "AMD", "side": "buy", "qty": 2}],
        own_universe=hr,
        foreign_universe=core,
    )
    assert all(o["symbol"] != "JNJ" for o in filtered)
    capped = cap_buys_to_cash(
        [{"symbol": "NVDA", "side": "buy", "qty": 100}],
        {"NVDA": 100.0},
        cash=-300.0,
        cash_reserve=500.0,
    )
    assert capped == []
    restore = restore_cash_sells(
        {"AMD": 25, "JNJ": 10},
        {"AMD": 160.0, "JNJ": 180.0},
        cash=-360.0,
        cash_reserve=500.0,
        own_universe=hr,
        foreign_universe=core,
    )
    assert restore and restore[0]["symbol"] == "AMD"
    assert restore[0]["side"] == "sell"
    final = finalize_orders_shared_account(
        [{"symbol": "NVDA", "side": "buy", "qty": 50}, {"symbol": "JNJ", "side": "sell", "qty": 10}],
        {"AMD": 25, "JNJ": 10},
        {"NVDA": 100.0, "AMD": 160.0, "JNJ": 180.0},
        cash=-360.0,
        own_universe=hr,
        foreign_universe=core,
        cash_reserve=500.0,
    )
    assert all(o["symbol"] != "JNJ" for o in final)
    assert any(o.get("symbol") == "AMD" and o.get("side") == "sell" for o in final)
    nvda_buys = [o for o in final if o.get("symbol") == "NVDA" and o.get("side") == "buy"]
    assert sum(int(o.get("qty") or 0) for o in nvda_buys) <= 1
    print("  cuenta compartida: 50/50 + cash-only + no vender otra manga OK")


def main():
    print("Validación: Trailing + fallback a Stop fijo y scheduler")
    print("-" * 50)
    test_caso_a()
    test_caso_b()
    test_caso_c()
    test_caso_d_cooldown()
    test_intervalos()
    test_update_trailing_state()
    test_max_weight_post_top()
    test_risk_checks_drawdown_real()
    test_alto_riesgo_sin_100pct()
    test_cuenta_compartida_cash_y_mangas()
    print("-" * 50)
    print("Todas las validaciones pasaron.")


if __name__ == "__main__":
    main()
