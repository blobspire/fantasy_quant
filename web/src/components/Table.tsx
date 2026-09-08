import { useMemo, useState, type ReactNode } from 'react';

/**
 * A dense, sortable table with tabular numerals and a sticky header.
 *
 * Two behaviours are deliberate and views should not work around them:
 *
 * THE DEFAULT ORDER IS THE SERVER'S ORDER. Nothing is sorted until the reader
 * clicks a header, and a third click on the same header restores the payload's
 * own order. That matters here: the action queue's ranking already puts
 * everything inside its own error below everything that is not
 * (`report.rank_actions`), and silently re-sorting it by point estimate would
 * put the noisiest row on top -- exactly what that ranking exists to prevent.
 *
 * AN INSIGNIFICANT ROW RECEDES, IT IS NOT DROPPED. Pass `dim` and every cell in
 * the row -- numbers included -- moves to `--fg-noise`. It stays readable, and
 * hovering restores full contrast.
 */
export type Align = 'left' | 'right' | 'center';

export interface Column<T> {
  /** Stable id. Also the sort key. */
  key: string;
  header: ReactNode;
  /** Hover text on the header. Use it for the definition of the column. */
  help?: string;
  align?: Align;
  /** Monospaced, tabular figures. Set it for anything a reader compares down the column. */
  num?: boolean;
  width?: number | string;
  /** Cell contents. Falls back to `value`. */
  render?: (row: T, index: number) => ReactNode;
  /** Sort key and default cell contents. Return null for "no value" (sorts last). */
  value?: (row: T) => number | string | null | undefined;
  /** Full control over ordering; overrides `value`. */
  sort?: (a: T, b: T) => number;
  /** Defaults to true when the column has `value` or `sort`. */
  sortable?: boolean;
  /** First click direction. Defaults to descending for numeric columns. */
  defaultDesc?: boolean;
}

export interface TableProps<T> {
  columns: Array<Column<T>>;
  rows: T[];
  rowKey: (row: T, index: number) => string | number;
  /** Start sorted on this column instead of the payload's own order. */
  initialSort?: { key: string; dir?: 'asc' | 'desc' };
  /** True for a row whose effect is inside its own error, or otherwise not load-bearing. */
  dim?: (row: T) => boolean;
  /** True for the row to read first -- the user's own team, the recommended claim. */
  highlight?: (row: T) => boolean;
  onRowClick?: (row: T, index: number) => void;
  /** Shown in place of the body when `rows` is empty. Say what "empty" means. */
  empty?: ReactNode;
  /** Scroll the body past this height, keeping the header pinned. */
  maxHeight?: number | string;
  compact?: boolean;
  className?: string;
  /** A totals or note row, rendered under the body inside the same scroll box. */
  footer?: ReactNode;
}

type SortState = { key: string; dir: 'asc' | 'desc' } | null;

function compareValues(a: unknown, b: unknown): number {
  const aMissing = a === null || a === undefined || (typeof a === 'number' && Number.isNaN(a));
  const bMissing = b === null || b === undefined || (typeof b === 'number' && Number.isNaN(b));
  // Missing values sort last in both directions: an unknown is not a small number.
  if (aMissing && bMissing) return 0;
  if (aMissing) return 1;
  if (bMissing) return -1;
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  return String(a).localeCompare(String(b), 'en', { numeric: true, sensitivity: 'base' });
}

export function Table<T>({
  columns,
  rows,
  rowKey,
  initialSort,
  dim,
  highlight,
  onRowClick,
  empty = 'nothing to show',
  maxHeight,
  compact,
  className,
  footer,
}: TableProps<T>) {
  const [sort, setSort] = useState<SortState>(
    initialSort ? { key: initialSort.key, dir: initialSort.dir ?? 'desc' } : null,
  );

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const column = columns.find((c) => c.key === sort.key);
    if (!column) return rows;
    const sign = sort.dir === 'asc' ? 1 : -1;
    // decorate/sort/undecorate keeps the original index available, so ties keep
    // the server's order rather than whatever the engine's sort happens to do.
    return rows
      .map((row, index) => ({ row, index }))
      .sort((a, b) => {
        const delta = column.sort
          ? column.sort(a.row, b.row)
          : compareValues(column.value?.(a.row), column.value?.(b.row));
        // Missing values stay last regardless of direction: compareValues has
        // already put them there, so do not flip that with the sign.
        const aMissing = !column.sort && column.value?.(a.row) == null;
        const bMissing = !column.sort && column.value?.(b.row) == null;
        if (aMissing !== bMissing) return aMissing ? 1 : -1;
        return delta === 0 ? a.index - b.index : delta * sign;
      })
      .map((entry) => entry.row);
  }, [rows, columns, sort]);

  function toggle(column: Column<T>) {
    const sortable = column.sortable ?? Boolean(column.value || column.sort);
    if (!sortable) return;
    const first: 'asc' | 'desc' = (column.defaultDesc ?? column.num) ? 'desc' : 'asc';
    setSort((current) => {
      if (!current || current.key !== column.key) return { key: column.key, dir: first };
      if (current.dir === first) return { key: column.key, dir: first === 'desc' ? 'asc' : 'desc' };
      return null; // third click: back to the payload's own order
    });
  }

  const wrapStyle =
    maxHeight === undefined
      ? undefined
      : { maxHeight: typeof maxHeight === 'number' ? `${maxHeight}px` : maxHeight };

  return (
    <div className="tbl-wrap" style={wrapStyle}>
      <table className={`tbl${compact ? ' tbl--compact' : ''}${className ? ` ${className}` : ''}`}>
        <thead>
          <tr>
            {columns.map((column) => {
              const sortable = column.sortable ?? Boolean(column.value || column.sort);
              const active = sort && sort.key === column.key ? sort : null;
              const classes = [
                column.num ? 'num' : '',
                column.align === 'right' || (column.num && !column.align) ? 'ta-right' : '',
                column.align === 'center' ? 'ta-center' : '',
                sortable ? 'is-sortable' : '',
                active ? 'is-sorted' : '',
              ]
                .filter(Boolean)
                .join(' ');
              return (
                <th
                  key={column.key}
                  className={classes || undefined}
                  style={column.width ? { width: column.width } : undefined}
                  title={column.help}
                  onClick={() => toggle(column)}
                  aria-sort={active ? (active.dir === 'asc' ? 'ascending' : 'descending') : undefined}
                  scope="col"
                >
                  {column.header}
                  {sortable ? (
                    <span className={`tbl__sort${active ? '' : ' tbl__sort--idle'}`}>
                      {active && active.dir === 'asc' ? '▲' : '▼'}
                    </span>
                  ) : null}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 ? (
            <tr>
              <td className="tbl__empty" colSpan={columns.length}>
                {empty}
              </td>
            </tr>
          ) : (
            sorted.map((row, index) => {
              const classes = [
                dim?.(row) ? 'is-dim' : '',
                highlight?.(row) ? 'is-highlight' : '',
                onRowClick ? 'is-clickable' : '',
              ]
                .filter(Boolean)
                .join(' ');
              return (
                <tr
                  key={rowKey(row, index)}
                  className={classes || undefined}
                  onClick={onRowClick ? () => onRowClick(row, index) : undefined}
                >
                  {columns.map((column) => {
                    const cellClasses = [
                      column.num ? 'num' : '',
                      column.align === 'right' || (column.num && !column.align) ? 'ta-right' : '',
                      column.align === 'center' ? 'ta-center' : '',
                    ]
                      .filter(Boolean)
                      .join(' ');
                    const content = column.render
                      ? column.render(row, index)
                      : renderValue(column.value?.(row));
                    return (
                      <td key={column.key} className={cellClasses || undefined}>
                        {content}
                      </td>
                    );
                  })}
                </tr>
              );
            })
          )}
        </tbody>
        {footer ? (
          <tfoot>
            <tr>
              <td colSpan={columns.length}>{footer}</td>
            </tr>
          </tfoot>
        ) : null}
      </table>
    </div>
  );
}

function renderValue(value: number | string | null | undefined): ReactNode {
  if (value === null || value === undefined) return <span className="faint">—</span>;
  return String(value);
}

/** A full-width rule inside a table body, e.g. between claims and the rest of the board. */
export function TableRule({ span }: { span: number }) {
  return (
    <tr className="tbl__rule">
      <td colSpan={span} />
    </tr>
  );
}
