import { useEffect, useState, type CSSProperties } from 'react'
import type { Comercio, Fluxo, Produto } from '../types/atlas'
import { SectionFrame } from './SectionFrame'
import { sectionById } from './manifest'

const ANO = 2025
const TOPO = 8
const WAFFLE = 100

type Load =
  | { state: 'loading' }
  | { state: 'error' }
  | { state: 'ready'; comercio: Comercio[] }

const styles = `
.pc-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 16px 28px;
  margin: 22px 0 32px;
}
.pc-pill-label {
  display: block;
  margin-bottom: 6px;
  color: var(--ink-soft);
  font-size: 13px;
}
.pc-pill {
  display: inline-flex;
  padding: 3px;
  border: 1px solid var(--ink);
  border-radius: 999px;
}
.pc-pill button {
  appearance: none;
  margin: 0;
  border: 0;
  border-radius: 999px;
  background: transparent;
  color: var(--ink);
  font-family: var(--font-sans);
  font-size: 15px;
  font-weight: 500;
  line-height: 1;
  min-height: 36px;
  padding: 0 14px;
  cursor: pointer;
}
.pc-pill button[aria-pressed='true'] {
  background: var(--accent);
  color: #fff;
}
.pc-pill button[aria-pressed='true']:hover {
  background: var(--accent-deep);
}
.pc-pill button:hover:not([aria-pressed='true']) {
  color: var(--accent-deep);
}
.pc-split {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 328px;
  gap: 36px 32px;
  align-items: start;
}
.pc-chart-title {
  margin: 0 0 8px;
  font-family: var(--font-serif);
  font-size: clamp(26px, 3vw, 32px);
  font-weight: 400;
  letter-spacing: -0.02em;
  line-height: 1.15;
}
.pc-rank-note {
  margin-bottom: 14px;
  color: var(--ink-soft);
  font-size: 14px;
}
.pc-bars {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.pc-row {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 28px;
}
.pc-name {
  flex: 0 1 var(--bar-name, 9.5rem);
  min-width: 5.5rem;
  overflow: hidden;
  font-size: 15px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.pc-fill {
  flex-basis: 0;
  flex-shrink: 1;
  height: 22px;
  min-width: 2px;
}
.pc-rest {
  flex-basis: 0;
  flex-shrink: 1;
  height: 1px;
}
.pc-value {
  flex: 0 0 auto;
  font-size: 14px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.pc-tip {
  position: absolute;
  z-index: 2;
  left: calc(var(--bar-name, 9.5rem) + 10px);
  bottom: calc(100% + 2px);
  padding: 3px 8px;
  background: var(--ink);
  color: #fff;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  line-height: 1.35;
  white-space: nowrap;
  opacity: 0;
  pointer-events: none;
}
.pc-row:hover > .pc-tip,
.pc-row:focus-visible > .pc-tip {
  opacity: 1;
}
.pc-waffle {
  margin: 0;
}
.pc-waffle-body {
  display: flex;
  flex-wrap: nowrap;
  align-items: center;
  gap: 16px;
}
.pc-grid {
  display: grid;
  grid-template-columns: repeat(10, 12px);
  gap: 3px;
  width: max-content;
}
.pc-cell {
  position: relative;
  display: block;
  width: 12px;
  height: 12px;
  background: #4a453f;
}
.pc-cell.is-on {
  background: var(--accent);
}
.pc-cell:hover::after {
  content: attr(data-tip);
  position: absolute;
  z-index: 3;
  right: 0;
  bottom: calc(100% + 6px);
  padding: 3px 8px;
  background: var(--ink);
  color: #fff;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  line-height: 1.35;
  white-space: nowrap;
  pointer-events: none;
}
.pc-pct-wrap {
  width: 9.5rem;
}
.pc-pct {
  font-size: 64px;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.045em;
  line-height: 0.9;
}
.pc-pct-note {
  display: block;
  margin-top: 8px;
  color: var(--ink-soft);
  font-size: 14px;
  line-height: 1.35;
}
.pc-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 16px;
  margin-top: 16px;
  color: var(--ink);
  font-size: 14px;
}
.pc-swatch {
  display: inline-block;
  width: 12px;
  height: 12px;
  margin-right: 6px;
  vertical-align: -1px;
}
.pc-swatch.is-on {
  background: var(--accent);
}
.pc-swatch.is-rest {
  background: #4a453f;
}
.pc-fixed {
  margin-top: 14px;
  color: var(--ink-soft);
  font-size: 14px;
  line-height: 1.4;
}
@media (max-width: 720px) {
  .pc-split {
    grid-template-columns: 1fr;
  }
  .pc-value {
    font-size: 13px;
  }
}
@media (max-width: 380px) {
  .pc-waffle-body {
    flex-wrap: wrap;
  }
}
`

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function readComercio(data: unknown): Comercio[] | null {
  if (!isRecord(data) || !Array.isArray(data.comercio) || data.comercio.length === 0) {
    return null
  }

  const list: Comercio[] = []
  for (const item of data.comercio) {
    if (!isRecord(item)) return null
    const { ano, fluxo, produto, pais, kg, usd } = item
    if (typeof ano !== 'number' || !Number.isFinite(ano)) return null
    if (fluxo !== 'importacao' && fluxo !== 'exportacao') return null
    if (produto !== 'mesa' && produto !== 'espumante') return null
    if (typeof pais !== 'string' || pais.length === 0) return null
    if (typeof kg !== 'number' || !Number.isFinite(kg)) return null
    if (typeof usd !== 'number' || !Number.isFinite(usd)) return null
    list.push({ ano, fluxo, produto, pais, kg, usd })
  }
  return list
}

function useComercio(): Load {
  const [status, setStatus] = useState<Load>({ state: 'loading' })

  useEffect(() => {
    let active = true
    const url = `${import.meta.env.BASE_URL}atlas.json`

    fetch(url)
      .then((response) => {
        if (!response.ok) throw new Error(String(response.status))
        return response.json() as Promise<unknown>
      })
      .then((data) => {
        if (!active) return
        const comercio = readComercio(data)
        if (!comercio) {
          setStatus({ state: 'error' })
          return
        }
        setStatus({ state: 'ready', comercio })
      })
      .catch(() => {
        if (!active) return
        setStatus({ state: 'error' })
      })

    return () => {
      active = false
    }
  }, [])

  return status
}

function formatKg(kg: number): string {
  return `${new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 0 }).format(Math.round(kg))} kg`
}

function nomeFluxo(fluxo: Fluxo): string {
  return fluxo === 'importacao' ? 'Importação' : 'Exportação'
}

function nomeProduto(produto: Produto): string {
  return produto === 'mesa' ? 'vinho de mesa' : 'espumante'
}

type RankRow = {
  pais: string
  kg: number
  lider: boolean
}

function ranking(comercio: Comercio[], fluxo: Fluxo, produto: Produto): { linhas: RankRow[]; total: number } {
  const serie = comercio.filter((row) => row.ano === ANO && row.fluxo === fluxo && row.produto === produto && row.kg > 0)
  serie.sort((a, b) => b.kg - a.kg || a.pais.localeCompare(b.pais, 'pt'))
  const max = serie[0]?.kg ?? 0
  const linhas = serie.slice(0, TOPO).map((row) => ({
    pais: row.pais,
    kg: row.kg,
    lider: max > 0 && row.kg === max,
  }))
  return { linhas, total: serie.length }
}

function fatiaParaguai(comercio: Comercio[]): { paraguai: number; total: number } | null {
  const serie = comercio.filter((row) => row.ano === ANO && row.fluxo === 'exportacao' && row.produto === 'mesa' && row.kg > 0)
  const total = serie.reduce((sum, row) => sum + row.kg, 0)
  const paraguai = serie.reduce((sum, row) => (row.pais === 'Paraguai' ? sum + row.kg : sum), 0)
  if (total <= 0 || paraguai <= 0) return null
  return { paraguai, total }
}

function Pills<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: { id: T; nome: string }[]
  onChange: (next: T) => void
}) {
  return (
    <div>
      <span className="pc-pill-label">{label}</span>
      <div className="pc-pill" role="group" aria-label={label}>
        {options.map((option) => (
          <button key={option.id} type="button" aria-pressed={value === option.id} onClick={() => onChange(option.id)}>
            {option.nome}
          </button>
        ))}
      </div>
    </div>
  )
}

function Barras({ linhas }: { linhas: RankRow[] }) {
  const max = Math.max(...linhas.map((linha) => linha.kg), 0)

  return (
    <ul className="pc-bars" style={{ '--bar-name': '9.5rem' } as CSSProperties} aria-label={`Ranking de parceiros em ${ANO}`}>
      {linhas.map((linha) => {
        const ratio = max > 0 ? linha.kg / max : 0
        return (
          <li key={linha.pais} className="pc-row" tabIndex={0} aria-label={`${linha.pais}, ${formatKg(linha.kg)}, ${ANO}`}>
            <span className="pc-name">{linha.pais}</span>
            <span className="pc-fill" style={{ flexGrow: ratio, background: linha.lider ? 'var(--win)' : 'var(--bar)' }} />
            <span className="pc-value">{formatKg(linha.kg)}</span>
            {ratio < 1 ? <span className="pc-rest" style={{ flexGrow: 1 - ratio }} /> : null}
            <span className="pc-tip" aria-hidden="true">
              {formatKg(linha.kg)} · {ANO}
            </span>
          </li>
        )
      })}
    </ul>
  )
}

function Waffle({ paraguai, total }: { paraguai: number; total: number }) {
  const percent = Math.round((paraguai / total) * 100)
  const preenchidas = Math.min(WAFFLE, Math.max(0, percent))
  const resto = total - paraguai
  const dicaParaguai = `Paraguai · ${formatKg(paraguai)} · ${ANO}`
  const dicaResto = `Demais países · ${formatKg(resto)} · ${ANO}`

  return (
    <figure
      className="pc-waffle"
      aria-label={`Paraguai, ${percent}% das exportações de vinho de mesa em ${ANO}, ${formatKg(paraguai)} de ${formatKg(total)}.`}
    >
      <h3 className="pc-chart-title">Fatia do Paraguai</h3>
      <div className="pc-waffle-body">
        <div className="pc-grid" aria-hidden="true">
          {Array.from({ length: WAFFLE }, (_, index) => {
            const ligada = index < preenchidas
            return <span key={index} className={ligada ? 'pc-cell is-on' : 'pc-cell'} data-tip={ligada ? dicaParaguai : dicaResto} />
          })}
        </div>
        <div className="pc-pct-wrap">
          <p className="pc-pct">{percent}%</p>
          <span className="pc-pct-note">do vinho de mesa exportado em {ANO}.</span>
        </div>
      </div>
      <ul className="pc-legend">
        <li>
          <span className="pc-swatch is-on" aria-hidden="true" />
          Paraguai
        </li>
        <li>
          <span className="pc-swatch is-rest" aria-hidden="true" />
          Demais países
        </li>
      </ul>
      <p className="pc-fixed">
        {formatKg(paraguai)} de {formatKg(total)}. Esta fatia não muda com o controle.
      </p>
    </figure>
  )
}

export function Parceiros() {
  const meta = sectionById('parceiros')
  const status = useComercio()
  const [fluxo, setFluxo] = useState<Fluxo>('importacao')
  const [produto, setProduto] = useState<Produto>('mesa')

  return (
    <SectionFrame meta={meta}>
      <style>{styles}</style>
      <p>O país líder da série escolhida fica em verde. Ao lado, a fatia do Paraguai nas exportações de vinho de mesa em {ANO}.</p>
      {status.state === 'loading' ? <p role="status">Carregando os parceiros de {ANO}.</p> : null}
      {status.state === 'error' ? <p role="alert">Não foi possível ler o comércio de {ANO}.</p> : null}
      {status.state === 'ready' ? (
        <Conteudo comercio={status.comercio} fluxo={fluxo} produto={produto} onFluxo={setFluxo} onProduto={setProduto} />
      ) : null}
    </SectionFrame>
  )
}

function Conteudo({
  comercio,
  fluxo,
  produto,
  onFluxo,
  onProduto,
}: {
  comercio: Comercio[]
  fluxo: Fluxo
  produto: Produto
  onFluxo: (fluxo: Fluxo) => void
  onProduto: (produto: Produto) => void
}) {
  const serie = ranking(comercio, fluxo, produto)
  const fatia = fatiaParaguai(comercio)

  return (
    <>
      <div className="pc-pills">
        <Pills
          label="Fluxo"
          value={fluxo}
          onChange={onFluxo}
          options={[
            { id: 'importacao', nome: 'Importação' },
            { id: 'exportacao', nome: 'Exportação' },
          ]}
        />
        <Pills
          label="Produto"
          value={produto}
          onChange={onProduto}
          options={[
            { id: 'mesa', nome: 'Mesa' },
            { id: 'espumante', nome: 'Espumante' },
          ]}
        />
      </div>
      <div className="pc-split">
        <div>
          <h3 className="pc-chart-title">
            {nomeFluxo(fluxo)} de {nomeProduto(produto)}, {ANO}
          </h3>
          {serie.total > serie.linhas.length ? <p className="pc-rank-note">Oito maiores parceiros, em kg.</p> : null}
          {serie.linhas.length === 0 ? <p>Sem quantidade registrada em {ANO} para esta série.</p> : <Barras linhas={serie.linhas} />}
        </div>
        {fatia ? <Waffle paraguai={fatia.paraguai} total={fatia.total} /> : <p role="alert">Não foi possível ler a fatia do Paraguai.</p>}
      </div>
    </>
  )
}
