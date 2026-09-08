import type { ReactNode } from 'react';
import { fmt, verdictOf, type Verdict } from '../api';
import { Delta } from './Delta';

/**
 * The primary display primitive: one measured number, said properly.
 *
 * This component is where "uncertainty must be visible" is enforced for the
 * whole app, so the intent is worth stating plainly:
 *
 * 1. THE NUMBER IS THE CONTENT. It is set large, monospaced and tabular, and
 *    nothing else in the block competes with it.
 *
 * 2. AN ERROR BAR IS PART OF THE NUMBER, NOT A FOOTNOTE. `stderr` renders
 *    inline, immediately after the value, one size step down and one contrast
 *    step down -- close enough that the eye takes them as one quantity. It is
 *    never smaller than the label above it, because an error bar the reader has
 *    to hunt for is an error bar that gets ignored.
 *
 * 3. INSIGNIFICANT RECEDES; IT DOES NOT DISAPPEAR. `significant={false}` drops
 *    the value to `--fg-noise`, thins its weight and marks it `~`. It stays
 *    perfectly readable. `report.Verdict` is explicit about why: a measurement
 *    that does not clear its own error "must be shown, because a user who is
 *    told nothing was found will go looking on his own".
 *
 * 4. `null` IS NOT ZERO. A verdict of `null` means nothing was measured -- an
 *    already-optimal lineup, a plan identical to holding -- and its zero
 *    standard error is structural. It renders grey with a `–` marker, never as
 *    a precise "0.00".
 *
 * 5. SIGNIFICANCE IS NOT COMPUTED HERE. `significant` and `verdict` are passed
 *    through from the payload; the right test differs by surface and lives in
 *    `report.verdict_for`.
 */
export interface StatProps {
  /** Short, uppercase in the render. Say the unit here, not in the value. */
  label: string;
  /**
   * The figure. Probabilities arrive as FRACTIONS (0.051 with `format="pct"`
   * renders "5.1%"). A string passes through untouched for the rare
   * non-numeric stat ("Tuesday night").
   */
  value: number | string | null | undefined;
  format?: 'pct' | 'pp' | 'points' | 'number' | 'int' | 'raw';
  digits?: number;
  /** Monte Carlo error on `value`, in the same unit as `value`. */
  stderr?: number | null;
  /** A trailing unit, e.g. "pts" or "σ". Set small and dim beside the value. */
  unit?: string;
  /** A signed change to show under/beside the value. Rendered by `<Delta>`. */
  delta?: number | null;
  deltaStderr?: number | null;
  deltaUnit?: 'pp' | 'points' | 'number';
  /**
   * Server-side significance of the thing this stat is about (the delta if
   * there is one, otherwise the value). `false` triggers the receding variant.
   */
  significant?: boolean | null;
  /** Server-side verdict; preferred over `significant` where the payload has it. */
  verdict?: Verdict | null;
  /** `verdict_note` or similar. Becomes the hover title on the figure. */
  note?: string | null;
  /** One line of context under the figure. Keep it short. */
  sub?: ReactNode;
  size?: 'lg' | 'md' | 'sm' | 'xs';
  align?: 'left' | 'right';
  className?: string;
}

const MARKERS: Partial<Record<Verdict, string>> = {
  noise: '~',
  null: '–',
  harm: '!',
};

function isNum(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function formatValue(
  value: number | string | null | undefined,
  format: NonNullable<StatProps['format']>,
  digits: number | undefined,
): string {
  if (typeof value === 'string') return value;
  if (!isNum(value)) return fmt.dash;
  switch (format) {
    case 'pct':
      return fmt.pct(value, digits ?? 1);
    case 'pp':
      return fmt.pp(value, digits ?? 2);
    case 'points':
      return fmt.num(value, digits ?? 1);
    case 'int':
      return fmt.int(value);
    case 'number':
    case 'raw':
    default:
      return fmt.num(value, digits ?? 2);
  }
}

function formatError(
  stderr: number | null | undefined,
  format: NonNullable<StatProps['format']>,
  digits: number | undefined,
): string | null {
  if (!isNum(stderr) || stderr <= 0) return null;
  if (format === 'pct' || format === 'pp') return fmt.ppError(stderr, digits ?? 2);
  if (format === 'int') return fmt.int(stderr);
  return fmt.num(Math.abs(stderr), digits ?? 2);
}

export function Stat({
  label,
  value,
  format = 'number',
  digits,
  stderr,
  unit,
  delta,
  deltaStderr,
  deltaUnit = 'pp',
  significant,
  verdict,
  note,
  sub,
  size = 'md',
  align = 'left',
  className,
}: StatProps) {
  // A stat is only "insignificant" when the server said so. `undefined` means
  // this figure is a level, not a measured effect (a championship probability,
  // a roster size) and there is nothing to de-emphasise.
  const judged = significant !== undefined && significant !== null;
  const kind: Verdict | null = judged || verdict ? verdictOf({ verdict, significant }) : null;

  // `harm` stays loud: a resolved loss is a finding, not a shrug. Only noise and
  // "nothing measured" recede.
  const recedes = kind === 'noise' || kind === 'null';
  const marker = kind ? MARKERS[kind] : undefined;

  const classes = [
    'stat',
    `stat--${size}`,
    align === 'right' ? 'stat--right' : '',
    recedes ? 'stat--insignificant' : '',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ');

  const errorText = formatError(stderr, format, digits);

  return (
    <div className={classes}>
      <div className="stat__label" title={label}>
        {label}
      </div>
      <div className="stat__figure" title={note ?? undefined}>
        <span className="stat__value">{formatValue(value, format, digits)}</span>
        {unit ? <span className="stat__unit">{unit}</span> : null}
        {errorText ? <span className="stat__err">±{errorText}</span> : null}
        {marker ? (
          <span
            className={`stat__marker${kind === 'null' ? ' stat__marker--null' : ''}${
              kind === 'harm' ? ' stat__marker--harm' : ''
            }`}
            title={note ?? undefined}
          >
            {marker}
          </span>
        ) : null}
        {isNum(delta) ? (
          <Delta
            value={delta}
            stderr={deltaStderr ?? null}
            significant={significant ?? null}
            verdict={verdict ?? null}
            unit={deltaUnit}
            size={size === 'lg' ? 'md' : 'sm'}
            note={note ?? null}
          />
        ) : null}
      </div>
      {sub ? <div className="stat__sub">{sub}</div> : null}
    </div>
  );
}

/**
 * A row of stats with hairline separators. The standard header for a view: three
 * to six figures, biggest first, no chart. Wraps on a narrow screen.
 */
export function StatRow({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={`statrow${className ? ` ${className}` : ''}`}>{children}</div>;
}

/** A stat that has not loaded yet. Same footprint, so nothing jumps on arrival. */
export function StatSkeleton({
  label,
  size = 'md',
}: {
  label?: string;
  size?: StatProps['size'];
}) {
  return (
    <div className={`stat stat--${size}`}>
      <div className="stat__label">{label ?? ' '}</div>
      <div className="stat__figure">
        <span className="stat__value skeleton">0.00%</span>
      </div>
    </div>
  );
}
