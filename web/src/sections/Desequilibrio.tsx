import { useEffect, useState, type CSSProperties } from 'react'
import type { Comercio } from '../types/atlas'
import { SectionFrame } from './SectionFrame'
import { sectionById } from './manifest'

const ANO = 2025
const PRODUTO = 'mesa'

type Load =
  | { state: 'loading' }
  | { state: 'error' }
  | { state: 'ready'; comercio: Comercio[] }

const styles = `
.bal-figures {
  display: grid;
  grid-template-columns: repeat(2, max-content);
  gap: 20px 48px;
  margin: 28px 0 40px;
}
.bal-figure {
  font-size: clamp(28px, 3vw, 44px);
  font-weight: 500;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.04em;
  line-height: 1;
  white-space: nowrap;
}
.bal-unit {
  font-size: 0.38em;
  font-weight: 500;
  letter-spacing: 0;
}
.bal-caption {
  display: block;
  margin-top: 10px;
  color: var(--ink-soft);
  font-size: 15px;
}
.bal-chart-title {
  margin: 0 0 18px;
  font-family: var(--font-serif);
  font-size: clamp(26px, 3vw, 32px);
  font-weight: 400;
  letter-spacing: -0.02em;
  line-height: 1.15;
}
.bal-bars {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.bal-row {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 28px;
}
.bal-name {
  flex: 0 1 var(--bar-name, 7.5rem);
  min-width: 5.5rem;
  overflow: hidden;
  font-size: 15px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.bal-fill {
  flex-basis: 0;
  flex-shrink: 1;
  height: 22px;
  min-width: 2px;
}
.bal-rest {
  flex-basis: 0;
  flex-shrink: 1;
  height: 1px;
}
.bal-value {
  flex: 0 0 auto;
  font-size: 14px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.bal-tip {
  position: absolute;
  z-index: 2;
  left: calc(var(--bar-name, 7.5rem) + 10px);
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
.bal-row:hover > .bal-tip,
.bal-row:focus-visible > .bal-tip {
  opacity: 1;
}
@media (max-width: 640px) {
  .bal-figures {
    grid-template-columns: 1fr;
    gap: 18px;
  }
  .bal-value {
    font-size: 13px;
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

function formatMilhoes(kg: number): string {
  const milhoes = kg / 1_000_000
  const digits = milhoes >= 10 ? 0 : 1
  const text = new Intl.NumberFormat('pt-BR', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(milhoes)
  return `${text} milhões`
}

function formatInteiro(valor: number): string {
  return new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 0 }).format(valor)
}

function soma(comercio: Comercio[], fluxo: 'importacao' | 'exportacao'): number {
  return comercio.reduce((total, row) => {
    if (row.ano !== ANO || row.produto !== PRODUTO || row.fluxo !== fluxo) return total
    return total + row.kg
  }, 0)
}

type BarItem = {
  id: string
  nome: string
  kg: number
  cor: string
}

function Barras({ items }: { items: BarItem[] }) {
  const max = Math.max(...items.map((item) => item.kg), 0)

  return (
    <ul className="bal-bars" style={{ '--bar-name': '7.5rem' } as CSSProperties} aria-label={`Quilos de vinho de mesa em ${ANO}`}>
      {items.map((item) => {
        const ratio = max > 0 ? item.kg / max : 0
        return (
          <li key={item.id} className="bal-row" tabIndex={0} aria-label={`${item.nome}, ${formatKg(item.kg)}, ${ANO}`}>
            <span className="bal-name">{item.nome}</span>
            <span className="bal-fill" style={{ flexGrow: ratio, background: item.cor }} />
            <span className="bal-value">{formatKg(item.kg)}</span>
            {ratio < 1 ? <span className="bal-rest" style={{ flexGrow: 1 - ratio }} /> : null}
            <span className="bal-tip" aria-hidden="true">
              {formatKg(item.kg)} · {ANO}
            </span>
          </li>
        )
      })}
    </ul>
  )
}

export function Desequilibrio() {
  const meta = sectionById('desequilibrio')
  const status = useComercio()

  if (status.state === 'loading') {
    return (
      <SectionFrame meta={meta}>
        <p role="status">Carregando os números de {ANO}.</p>
      </SectionFrame>
    )
  }

  if (status.state === 'error') {
    return (
      <SectionFrame meta={meta}>
        <p role="alert">Não foi possível ler o comércio de {ANO}.</p>
      </SectionFrame>
    )
  }

  const importados = soma(status.comercio, 'importacao')
  const exportados = soma(status.comercio, 'exportacao')
  if (importados <= 0 || exportados <= 0) {
    return (
      <SectionFrame meta={meta}>
        <p role="alert">Não foi possível ler o comércio de {ANO}.</p>
      </SectionFrame>
    )
  }

  const fator = Math.round(importados / exportados)
  const texto = `Em ${ANO} o Brasil importou cerca de ${formatInteiro(fator)} vezes mais vinho de mesa do que exportou.`

  return (
    <SectionFrame meta={meta}>
      <style>{styles}</style>
      <p>{texto}</p>
      <div className="bal-figures">
        <div>
          <p className="bal-figure">
            {formatMilhoes(importados)} <span className="bal-unit">de kg</span>
          </p>
          <span className="bal-caption">importados em {ANO}</span>
        </div>
        <div>
          <p className="bal-figure">
            {formatMilhoes(exportados)} <span className="bal-unit">de kg</span>
          </p>
          <span className="bal-caption">exportados em {ANO}</span>
        </div>
      </div>
      <h3 className="bal-chart-title">Vinho de mesa em {ANO}</h3>
      <Barras
        items={[
          { id: 'importacao', nome: 'Importação', kg: importados, cor: 'var(--accent)' },
          { id: 'exportacao', nome: 'Exportação', kg: exportados, cor: 'var(--bar)' },
        ]}
      />
    </SectionFrame>
  )
}
