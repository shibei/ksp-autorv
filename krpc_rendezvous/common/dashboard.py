"""Real-time terminal HUD for flight telemetry using ANSI escape sequences."""

import math
import sys


def format_value(val, unit, decimals=1):
    """Format a numeric value with unit, choosing appropriate notation."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return '-'
    abs_val = abs(val)
    if abs_val >= 1e5 or 0 < abs_val < 0.001:
        return f'{val:.{decimals}e}'
    else:
        return f'{val:.{decimals}f}'


def format_time(seconds):
    """Format seconds as MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f'{m:02d}:{s:02d}'


class Dashboard:
    """ANSI-based real-time terminal dashboard for flight telemetry.

    Usage:
        db = Dashboard(columns=[("ALT", "m", 1), ("VEL", "m/s", 1)])
        db.start()
        while flying:
            db.update([altitude, velocity])
        db.stop()
    """

    def __init__(self, columns, update_interval=0.5):
        """Initialize dashboard with column definitions.

        columns: list of (label, unit, decimals) tuples
        """
        self.columns = columns
        self.update_interval = update_interval
        self._last_render = 0.0
        self._started = False

    def start(self):
        """Hide cursor and prepare terminal."""
        self._started = True
        sys.stdout.write('\033[?25l')
        sys.stdout.flush()

    def stop(self):
        """Show cursor and clean up."""
        sys.stdout.write('\033[?25h')
        sys.stdout.write('\n')
        sys.stdout.flush()
        self._started = False

    def update(self, values, ut=None, extra_lines=None):
        """Render one frame of the dashboard."""
        if not self._started:
            return

        line = self.format_row(values, ut)
        sys.stdout.write('\r\033[K' + line)

        if extra_lines:
            for el in extra_lines:
                sys.stdout.write('\n\033[K' + el)
            sys.stdout.write(f'\033[{len(extra_lines)}A')

        sys.stdout.flush()

    def format_row(self, values, ut=None):
        """Format one row of the dashboard."""
        parts = []
        for (label, unit, decimals), val in zip(self.columns, values):
            formatted = format_value(val, unit, decimals)
            parts.append(f'{label}:{formatted}')
        row = ' | '.join(parts)
        if ut is not None:
            row = f'T+{format_time(ut)} | ' + row
        return row
