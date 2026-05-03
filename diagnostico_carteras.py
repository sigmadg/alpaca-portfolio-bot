#!/usr/bin/env python3
"""
Diagnóstico rápido de carteras: estado persistente y posibles causas de caídas.
Lee state_low_risk.json y state_high_risk.json y muestra:
- Símbolos con last_stop_ts reciente (ventas por stop/trailing)
- Meta (últimos reportes)
- Sugerencias según teoría (ANALISIS_CARTERAS_TEORIA.md)
"""
import json
import os
from datetime import datetime, timezone

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()

    for label, state_file in [("Largo plazo (bajo riesgo)", "state_low_risk.json"),
                               ("Alto riesgo", "state_high_risk.json")]:
        path = os.path.join(script_dir, state_file)
        print("=" * 60)
        print(f"  {label}: {state_file}")
        print("=" * 60)
        if not os.path.exists(path):
            print("  (archivo no existe)\n")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            print(f"  Error leyendo: {e}\n")
            continue

        symbols = state.get("symbols") or {}
        meta = state.get("meta") or {}

        # Símbolos con last_stop_ts (venta por stop en algún momento)
        con_stop = [(s, info.get("last_stop_ts")) for s, info in symbols.items() if info.get("last_stop_ts")]
        if con_stop:
            print("\n  Símbolos con venta por stop registrada (last_stop_ts):")
            cooldown_min = 360 if "low" in state_file else 30
            for sym, ts in con_stop:
                if ts is None:
                    continue
                try:
                    t = float(ts)
                except (TypeError, ValueError):
                    t = 0
                if t <= 0:
                    continue
                dt = now_ts - t
                mins = dt / 60
                horas = dt / 3600
                en_cooldown = dt < (cooldown_min * 60)
                status = "EN COOLDOWN (no compra)" if en_cooldown else "cooldown pasado"
                print(f"    {sym}: hace {horas:.1f}h ({mins:.0f} min) — {status}")
        else:
            print("\n  Ningún símbolo con last_stop_ts (no hay ventas por stop recientes en estado).")

        # Meta
        print("\n  Meta (reportes):")
        for k in ("last_mini_report_ts", "last_full_report_ts"):
            v = meta.get(k)
            if v is None:
                print(f"    {k}: (nunca)")
            else:
                try:
                    t = float(v)
                    dt = now_ts - t
                    print(f"    {k}: hace {dt/60:.0f} min")
                except (TypeError, ValueError):
                    print(f"    {k}: {v}")

        print()

    print("=" * 60)
    print("  Resumen (teoría)")
    print("=" * 60)
    print("""
  Si hay muchos símbolos con last_stop_ts reciente:
  - Los stops 5% están vendiendo en caídas; en mercados volátiles eso
    realiza pérdidas y evita participar en rebotes.
  - Revisar ANALISIS_CARTERAS_TEORIA.md para:
    • Subir stop_loss_porcentaje (ej. 7-8%) o desactivar trailing.
    • Revisar cooldown y max_drawdown.
  Si la cartera 'no deja de caer': suele ser combinación de
  mercado bajista + ventas por stop + bloqueo por drawdown/trades.
""")


if __name__ == "__main__":
    main()
