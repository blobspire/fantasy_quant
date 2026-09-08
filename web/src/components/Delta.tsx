import { fmt, verdictOf, type Verdict } from '../api';

/**
 * A signed change, rendered so that a real effect and a number inside its own
 * Monte Carlo error cannot be mistaken for each other.
 *
 * THE RULE THIS COMPONENT ENFORCES: never print a bare figure the data says is
 * noise. If the server did not mark a delta `act`, this renders the `+/-` beside
 * it and marks the row, every time, whether or not the caller asked for it. A
 * `+1.23pp` with the error hidden is the single most misleading thing this app
 * could show -- on the live trade board every row is below the noise ceiling of
 * its own search, and the biggest number in the portfolio is one of them.
 *
 * The four states come straight off `report.Verdict` and are visually distinct
 * because they mean different things:
 *
 *   act   green (or red for a measured loss)  cleared its own error
 *   noise amber, lighter weight, `~`          measured, did not clear its error
 *   harm  red, `!`                            resolved, in the wrong direction
 *   null  grey, `—`                           nothing was measured at all
 *
 * `null` is deliberately not "0.00pp". A zero delta with a zero standard error
 * means nothing was measured -- an already-optimal lineup produces two
 * bit-identical simulated arms -- and printing it as a precise zero says the
 * opposite of what happened.
 */
export interface DeltaProps {
  /**
   * The change. A probability difference as a FRACTION when `unit` is "pp"
   * (0.0277 renders "+2.77pp"); a plain number for "points" and "number".
   */
  value: number | null | undefined;
  /** Monte Carlo error on `value`, in the same unit. */
  stderr?: number | null;
  /** Server-side significance. Passed straight through from the payload. */
  significant?: boolean | null;
  /** Server-side verdict; preferred over `significant` when the payload has it. */
  verdict?: Verdict | null;
  unit?: 'pp' | 'points' | 'number';
  digits?: number;
  size?: 'sm' | 'md' | 'lg';
  /**
   * Whether to show the `+/-`. "auto" (the default) shows it whenever there is
   * one to show. Setting `false` suppresses it ONLY for `act` rows -- an
   * insignificant number always carries its error.
   */
  showError?: boolean | 'auto';
  /** `verdict_note` from the payload. Becomes the hover title. */
  note?: string | null;
  className?: string;
}

const MARKERS: Record<Verdict, string> = {
  act: '',
  noise: '~',
  null: '',
  harm: '!',
};

const FALLBACK_NOTES: Record<Verdict, string> = {
  act: 'clears its own Monte Carlo error',
  noise: 'inside its own Monte Carlo error — treat as unresolved',
  null: 'nothing was measured: this move changes no roster',
  harm: 'measured as a LOSS of title probability — do not do this',
};

function isNum(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

export function Delta({
  value,
  stderr,
  significant,
  verdict,
  unit = 'pp',
  digits,
  size = 'md',
  showError = 'auto',
  note,
  className,
}: DeltaProps) {
  const kind = verdictOf({
    verdict: verdict ?? null,
    significant: significant ?? null,
    delta_title: isNum(value) ? value : null,
  });
  const places = digits ?? (unit === 'pp' ? 2 : 2);
  const hasError = isNum(stderr) && stderr > 0;

  // The forcing rule: anything that is not `act` shows its error if it has one.
  const withError = kind === 'act' ? showError !== false && hasError : hasError;

  const negative = isNum(value) && value < 0;
  const classes = [
    'delta',
    `delta--${kind}`,
    `delta--${size}`,
    negative ? 'delta--negative' : '',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ');

  const title = note || FALLBACK_NOTES[kind];

  if (!isNum(value)) {
    return (
      <span className="delta delta--null" title="no value">
        <span className="delta__value">{fmt.dash}</span>
      </span>
    );
  }

  const text =
    unit === 'pp' ? fmt.pp(value, places) : fmt.signed(value, places) + (unit === 'points' ? '' : '');
  const errorText =
    unit === 'pp' ? fmt.ppError(stderr, places) : fmt.num(stderr ? Math.abs(stderr) : null, places);

  return (
    <span className={classes} title={title}>
      {MARKERS[kind] ? <span className="delta__mark">{MARKERS[kind]}</span> : null}
      <span className="delta__value">{text}</span>
      {withError ? <span className="delta__err">±{errorText}</span> : null}
    </span>
  );
}

/**
 * The same thing for a row in a table: no error bar inline (the table gives the
 * `+/-` its own column so it stays comparable down the page), but the colour,
 * the marker and the hover note are identical. Views that use this MUST render
 * a stderr column beside it -- see `Table.tsx`'s `deltaColumns` helper.
 */
export function DeltaCell(props: Omit<DeltaProps, 'showError'>) {
  return <Delta {...props} showError={false} size={props.size ?? 'sm'} />;
}

/**
 * A bare `+/-`, for the column beside a `DeltaCell`. Renders the em dash when
 * there is no error to show, which for a `null` verdict is the honest output:
 * a zero error there means nothing was measured, not measured perfectly.
 */
export function ErrorBar({
  stderr,
  unit = 'pp',
  digits = 2,
}: {
  stderr: number | null | undefined;
  unit?: 'pp' | 'points' | 'number';
  digits?: number;
}) {
  if (!isNum(stderr) || stderr <= 0) return <span className="faint">{fmt.dash}</span>;
  return (
    <span className="delta__err">
      ±{unit === 'pp' ? fmt.ppError(stderr, digits) : fmt.num(Math.abs(stderr), digits)}
    </span>
  );
}
