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
    """Intervalos obligatorios: largo plazo 3600/60, alto riesgo 1800/60 (mini reporte cada 60 s en ambas)."""
    low = get_portfolio_config("largo_plazo", None)
    high = get_portfolio_config("alto_riesgo", None)
    assert low["report_interval_seconds"] == 3600, "Largo plazo: report_interval_seconds debe ser 3600"
    assert low["mini_monitor_interval_seconds"] == 60, "Largo plazo: mini debe ser 60"
    assert high["report_interval_seconds"] == 1800, "Alto riesgo: report_interval_seconds debe ser 1800"
    assert high["mini_monitor_interval_seconds"] == 60, "Alto riesgo: mini debe ser 60"
    assert low["cooldown_minutes_after_stop"] == 360, "Largo plazo: cooldown 360 min"
    assert high["cooldown_minutes_after_stop"] == 30, "Alto riesgo: cooldown 30 min"
    print("  Intervalos OK: largo 3600s/60s, alto 1800s/60s; cooldown 360/30 min")


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


def main():
    print("Validación: Trailing + fallback a Stop fijo y scheduler")
    print("-" * 50)
    test_caso_a()
    test_caso_b()
    test_caso_c()
    test_caso_d_cooldown()
    test_intervalos()
    test_update_trailing_state()
    print("-" * 50)
    print("Todas las validaciones pasaron.")


if __name__ == "__main__":
    main()
