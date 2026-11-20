import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { PointerEvent } from 'react'
import type { Comercio, Fluxo } from '../types/atlas'
import { SectionFrame } from './SectionFrame'
import { sectionById } from './manifest'

const FLUXOS: { fluxo: Fluxo; rotulo: string }[] = [
  { fluxo: 'importacao', rotulo: 'Importação' },
  { fluxo: 'exportacao', rotulo: 'Exportação' },
]

type Caixa = {
  largura: number
  altura: number
  esquerda: number
  direita: number
  topo: number
  base: number
}
const FRANCA = 'França'

type Estado =
  | { status: 'loading' }
  | { status: 'error' }
  | { status: 'ready'; comercio: Comercio[] }

type Volume = { pais: string; kg: number; usd: number }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function lerComercio(data: unknown): Comercio[] | null {
  if (!isRecord(data) || !Array.isArray(data.comercio)) return null
  const comercio: Comercio[] = []
  for (const item of data.comercio) {
    if (!isRecord(item)) return null
    if (
      typeof item.ano !== 'number' ||
      (item.fluxo !== 'importacao' && item.fluxo !== 'exportacao') ||
      (item.produto !== 'mesa' && item.produto !== 'espumante') ||
      typeof item.pais !== 'string' ||
      typeof item.kg !== 'number' ||
      typeof item.usd !== 'number' ||
      !Number.isFinite(item.kg) ||
      !Number.isFinite(item.usd)
    ) {
      return null
    }
    comercio.push({
      ano: item.ano,
      fluxo: item.fluxo,
      produto: item.produto,
      pais: item.pais,
      kg: item.kg,
      usd: item.usd,
    })
  }
  const temMesa = FLUXOS.every((item) =>
    comercio.some((registro) => registro.fluxo === item.fluxo && registro.produto === 'mesa' && registro.kg > 0),
  )
  return temMesa ? comercio : null
}

function useComercio(): Estado {
  const [estado, setEstado] = useState<Estado>({ status: 'loading' })

  useEffect(() => {
    let ativo = true
    fetch(`${import.meta.env.BASE_URL}atlas.json`)
      .then((resposta) => {
        if (!resposta.ok) throw new Error(String(resposta.status))
        return resposta.json() as Promise<unknown>
      })
      .then((data) => {
        if (!ativo) return
        const comercio = lerComercio(data)
        setEstado(comercio ? { status: 'ready', comercio } : { status: 'error' })
      })
      .catch(() => {
        if (ativo) setEstado({ status: 'error' })
      })
    return () => {
      ativo = false
    }
  }, [])

  return estado
}

function mediaPonderada(comercio: Comercio[], fluxo: Fluxo): Map<number, number> {
  const totais = new Map<number, { kg: number; usd: number }>()
  for (const registro of comercio) {
    if (registro.fluxo !== fluxo || registro.produto !== 'mesa') continue
    const atual = totais.get(registro.ano) ?? { kg: 0, usd: 0 }
    atual.kg += registro.kg
    atual.usd += registro.usd
    totais.set(registro.ano, atual)
  }
  const medias = new Map<number, number>()
  for (const [ano, total] of totais) {
    if (total.kg > 0) medias.set(ano, total.usd / total.kg)
  }
  return medias
}

function volumesEspumante(comercio: Comercio[], ano: number): Volume[] {
  const mapa = new Map<string, Volume>()
  for (const registro of comercio) {
    if (registro.ano !== ano || registro.fluxo !== 'importacao' || registro.produto !== 'espumante') {
      continue
    }
    if (registro.kg <= 0) continue
    const atual = mapa.get(registro.pais) ?? { pais: registro.pais, kg: 0, usd: 0 }
    atual.kg += registro.kg
    atual.usd += registro.usd
    mapa.set(registro.pais, atual)
  }
  return [...mapa.values()].sort((a, b) => b.kg - a.kg || a.pais.localeCompare(b.pais, 'pt'))
}

function usdPorKg(valor: number): string {
  const numero = valor.toLocaleString('pt-BR', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
  return `US$ ${numero}/kg`
}

function kgCurtos(kg: number): string {
  const milhoes = kg / 1_000_000
  if (milhoes >= 0.95) {
    const texto = milhoes.toLocaleString('pt-BR', {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    })
    return `${texto} ${milhoes < 1.95 ? 'milhão' : 'milhões'} de kg`
  }
  const mil = kg / 1_000
  if (mil >= 10) return `${Math.round(mil).toLocaleString('pt-BR')} mil kg`
  return `${Math.round(kg).toLocaleString('pt-BR')} kg`
}

function kgExatos(kg: number): string {
  return `${Math.round(kg).toLocaleString('pt-BR')} kg`
}

function caixaDe(largura: number): Caixa {
  if (largura < 520) {
    return { largura, altura: 240, esquerda: 28, direita: 8, topo: 12, base: 24 }
  }
  return { largura, altura: 320, esquerda: 44, direita: 16, topo: 16, base: 32 }
}

function xDo(ano: number, inicio: number, fim: number, caixa: Caixa): number {
  const plot = caixa.largura - caixa.esquerda - caixa.direita
  if (fim === inicio || plot <= 0) return caixa.esquerda
  return caixa.esquerda + ((ano - inicio) / (fim - inicio)) * plot
}

function yDo(valor: number, teto: number, caixa: Caixa): number {
  const plot = caixa.altura - caixa.topo - caixa.base
  const limitado = teto <= 0 ? 0 : Math.min(Math.max(valor, 0), teto)
  return caixa.topo + (1 - limitado / teto) * plot
}

function useLargura(ref: { current: SVGSVGElement | null }) {
  const [largura, setLargura] = useState(640)

  useLayoutEffect(() => {
    const elemento = ref.current
    if (!elemento) return
    const aplicar = (valor: number) => {
      if (valor <= 0) return
      const proxima = Math.round(valor)
      setLargura((atual) => (atual === proxima ? atual : proxima))
    }
    aplicar(elemento.getBoundingClientRect().width)
    const observer = new ResizeObserver((entries) => {
      aplicar(entries[0]?.contentRect.width ?? 0)
    })
    observer.observe(elemento)
    return () => observer.disconnect()
  }, [ref])

  return largura
}

function marcasDoEixo(inicio: number, fim: number, caixa: Caixa): number[] {
  const plot = Math.max(0, caixa.largura - caixa.esquerda - caixa.direita)
  if (plot < 460) {
    const meio = Math.round((inicio + fim) / 20) * 10
    const marcas = [inicio]
    if (meio > inicio + 5 && meio < fim - 5) marcas.push(meio)
    if (fim !== inicio) marcas.push(fim)
    return marcas
  }
  const marcas = [inicio]
  for (let ano = Math.ceil((inicio + 1) / 10) * 10; ano < fim; ano += 10) marcas.push(ano)
  if (marcas[marcas.length - 1] !== fim) marcas.push(fim)
  return marcas
}

function caminho(pontos: { x: number; y: number }[]): string {
  return pontos
    .map((ponto, indice) => `${indice === 0 ? 'M' : 'L'}${ponto.x.toFixed(2)} ${ponto.y.toFixed(2)}`)
    .join(' ')
}

function SerieAnual({
  series,
  foco,
}: {
  series: { fluxo: Fluxo; rotulo: string; valores: Map<number, number> }[]
  foco: Fluxo
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const largura = useLargura(svgRef)
  const caixa = caixaDe(largura)
  const tituloId = useId()
  const anos = useMemo(() => {
    const conjunto = new Set<number>()
    for (const serie of series) {
      for (const ano of serie.valores.keys()) conjunto.add(ano)
    }
    return [...conjunto].sort((a, b) => a - b)
  }, [series])
  const inicio = anos[0] ?? 0
  const fim = anos[anos.length - 1] ?? inicio
  const [anoAtivo, setAnoAtivo] = useState(fim)

  const maximo = Math.max(
    0,
    ...series.flatMap((serie) => [...serie.valores.values()]),
  )
  const inteiro = Math.ceil(maximo)
  const teto = Math.max(1, inteiro - maximo < 0.12 ? inteiro + 1 : inteiro)
  const marcasY: number[] = []
  for (let marca = 0; marca <= teto; marca += 1) marcasY.push(marca)

  const marcasX = marcasDoEixo(inicio, fim, caixa)

  const base = yDo(0, teto, caixa)
  const plot = caixa.largura - caixa.esquerda - caixa.direita
  const espaco = anos.length > 1 ? plot / (anos.length - 1) : plot
  const ordem = [...series].sort((a, b) => Number(a.fluxo === foco) - Number(b.fluxo === foco))

  function apontar(event: PointerEvent<SVGSVGElement>) {
    const svg = svgRef.current
    if (!svg || anos.length === 0) return
    const rect = svg.getBoundingClientRect()
    if (rect.width === 0) return
    const x = ((event.clientX - rect.left) / rect.width) * caixa.largura
    if (x < caixa.esquerda - 12 || x > caixa.largura - caixa.direita + 12) return
    let melhor = anos[0]
    let menor = Infinity
    for (const ano of anos) {
      const distancia = Math.abs(xDo(ano, inicio, fim, caixa) - x)
      if (distancia < menor) {
        menor = distancia
        melhor = ano
      }
    }
    setAnoAtivo(melhor)
  }

  function corDe(fluxo: Fluxo): string {
    return fluxo === foco ? 'var(--accent)' : 'var(--ink-soft)'
  }

  return (
    <figure className="preco-figura">
      <figcaption className="preco-titulo" id={tituloId}>
        US$/kg do vinho de mesa, {inicio}–{fim}
      </figcaption>
      <ul className="preco-legenda">
        {series.map((serie) => (
          <li key={serie.fluxo}>
            <span className="preco-amostra" style={{ background: corDe(serie.fluxo) }} aria-hidden="true" />
            {serie.rotulo}
          </li>
        ))}
      </ul>
      <svg
        ref={svgRef}
        className="preco-svg"
        viewBox={`0 0 ${caixa.largura} ${caixa.altura}`}
        role="img"
        aria-labelledby={tituloId}
        onPointerDown={apontar}
        onPointerMove={apontar}
        onPointerLeave={() => setAnoAtivo(fim)}
      >
        <line
          x1={caixa.esquerda}
          x2={caixa.largura - caixa.direita}
          y1={base}
          y2={base}
          stroke="var(--line)"
          strokeWidth="1"
        />
        <line
          x1={xDo(anoAtivo, inicio, fim, caixa)}
          x2={xDo(anoAtivo, inicio, fim, caixa)}
          y1={caixa.topo}
          y2={base}
          stroke="var(--ink-soft)"
          strokeOpacity="0.35"
          strokeWidth="1"
        />
        {marcasY.map((marca) => (
          <text
            key={marca}
            x={caixa.esquerda - 8}
            y={yDo(marca, teto, caixa)}
            textAnchor="end"
            dominantBaseline="middle"
            fill="var(--ink-soft)"
            fontFamily="var(--font-sans)"
            fontSize="12"
          >
            {marca}
          </text>
        ))}
        {marcasX.map((ano) => (
          <text
            key={ano}
            x={xDo(ano, inicio, fim, caixa)}
            y={caixa.altura - 6}
            textAnchor={ano === inicio ? 'start' : ano === fim ? 'end' : 'middle'}
            fill="var(--ink-soft)"
            fontFamily="var(--font-sans)"
            fontSize="12"
          >
            {ano}
          </text>
        ))}
        {ordem.map((serie) => {
          const pontos = anos.flatMap((ano) => {
            const valor = serie.valores.get(ano)
            if (valor === undefined) return []
            return [{ ano, valor, x: xDo(ano, inicio, fim, caixa), y: yDo(valor, teto, caixa) }]
          })
          return (
            <g key={serie.fluxo}>
              <path
                d={caminho(pontos)}
                fill="none"
                stroke={corDe(serie.fluxo)}
                strokeWidth="2"
                strokeLinejoin="miter"
                strokeLinecap="butt"
              />
              {pontos.map((ponto) =>
                ponto.ano === anoAtivo || espaco >= 8 || (ponto.ano - inicio) % 5 === 0 ? (
                  <circle
                    key={ponto.ano}
                    cx={ponto.x}
                    cy={ponto.y}
                    r={ponto.ano === anoAtivo ? 4 : 2}
                    fill={corDe(serie.fluxo)}
                  />
                ) : null,
              )}
            </g>
          )
        })}
      </svg>
      <p className="preco-leitura">
        <span className="preco-ano">{anoAtivo}</span>
        {series.map((serie) => {
          const valor = serie.valores.get(anoAtivo)
          return (
            <span key={serie.fluxo}>
              {serie.rotulo} {valor === undefined ? 'sem kg' : usdPorKg(valor)}
            </span>
          )
        })}
      </p>
      <div className="preco-sr">
      <table>
        <caption>US$ por quilograma do vinho de mesa, média ponderada por ano.</caption>
        <thead>
          <tr>
            <th scope="col">Ano</th>
            {series.map((serie) => (
              <th key={serie.fluxo} scope="col">
                {serie.rotulo}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {anos.map((ano) => (
            <tr key={ano}>
              <th scope="row">{ano}</th>
              {series.map((serie) => {
                const valor = serie.valores.get(ano)
                return <td key={serie.fluxo}>{valor === undefined ? '—' : usdPorKg(valor)}</td>
              })}
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </figure>
  )
}

function comArtigo(pais: string): string {
  if (pais === 'Espanha' || pais === 'França' || pais === 'Argentina' || pais === 'Itália') {
    return `A ${pais}`
  }
  if (pais === 'Chile' || pais === 'Paraguai' || pais === 'Uruguai') {
    return `O ${pais}`
  }
  return pais
}

function corDaBarra(pais: string, lider: string): string {
  if (pais === FRANCA) return 'var(--accent)'
  if (pais === lider) return 'var(--win)'
  return 'var(--bar)'
}

function Recorte({ volumes, ano }: { volumes: Volume[]; ano: number }) {
  const tituloId = useId()
  const lider = volumes[0]
  const franca = volumes.find((volume) => volume.pais === FRANCA)
  const nomes = new Set(volumes.slice(0, 5).map((volume) => volume.pais))
  if (franca) nomes.add(franca.pais)
  const lista = volumes.filter((volume) => nomes.has(volume.pais))
  const maximo = lista[0]?.kg ?? 1
  const total = volumes.reduce((soma, volume) => soma + volume.kg, 0)
  const [paisAtivo, setPaisAtivo] = useState<string | null>(null)
  const ativo = lista.find((volume) => volume.pais === paisAtivo) ?? lider

  if (!lider || !ativo) return null

  const precoDe = (volume: Volume) => (volume.kg > 0 ? volume.usd / volume.kg : 0)
  const fatia = total > 0 ? Math.round((lider.kg / total) * 100) : 0
  const demais = lista.some((volume) => volume.pais !== FRANCA && volume.pais !== lider.pais)
  const fraseFranca =
    franca && franca.pais !== lider.pais
      ? `A França chegou a ${usdPorKg(precoDe(franca))}, com ${kgCurtos(franca.kg)}. `
      : ''

  return (
    <figure className="preco-figura">
      <figcaption className="preco-titulo" id={tituloId}>
        Espumante importado em {ano}, em kg
      </figcaption>
      <p>
        {fraseFranca}
        {comArtigo(lider.pais)} tem o maior volume: {kgCurtos(lider.kg)}, cerca de {fatia}% do
        espumante importado, a {usdPorKg(precoDe(lider))}.
      </p>
      <ul className="preco-legenda">
        {franca ? (
          <li>
            <span className="preco-amostra" style={{ background: 'var(--accent)' }} aria-hidden="true" />
            França
          </li>
        ) : null}
        {lider.pais !== FRANCA ? (
          <li>
            <span className="preco-amostra" style={{ background: 'var(--win)' }} aria-hidden="true" />
            {lider.pais}
          </li>
        ) : null}
        {demais ? (
          <li>
            <span className="preco-amostra" style={{ background: 'var(--bar)' }} aria-hidden="true" />
            Demais
          </li>
        ) : null}
      </ul>
      <div className="preco-barras" role="group" aria-labelledby={tituloId} onMouseLeave={() => setPaisAtivo(null)}>
        {lista.map((volume) => (
          <button
            key={volume.pais}
            type="button"
            className="preco-linha"
            onMouseEnter={() => setPaisAtivo(volume.pais)}
            onFocus={() => setPaisAtivo(volume.pais)}
            onBlur={() => setPaisAtivo(null)}
          >
            <span className="preco-pais">{volume.pais}</span>
            <span className="preco-trilho">
              <span
                className="preco-barra"
                style={{
                  ['--fatia' as string]: String(volume.kg / maximo),
                  background: corDaBarra(volume.pais, lider.pais),
                }}
              />
              <span className="preco-valor">
                <span>{kgCurtos(volume.kg)}</span>
                <span style={volume.pais === FRANCA ? { color: 'var(--accent)' } : undefined}>
                  {usdPorKg(precoDe(volume))}
                </span>
              </span>
            </span>
          </button>
        ))}
      </div>
      <p className="preco-leitura">
        <span className="preco-ano">{ano}</span>
        <span>{ativo.pais}</span>
        <span>{kgExatos(ativo.kg)}</span>
        <span>{usdPorKg(precoDe(ativo))}</span>
      </p>
    </figure>
  )
}

function Conteudo({ comercio }: { comercio: Comercio[] }) {
  const [foco, setFoco] = useState<Fluxo>('importacao')
  const controlesId = useId()
  const series = useMemo(
    () =>
      FLUXOS.map((item) => ({
        ...item,
        valores: mediaPonderada(comercio, item.fluxo),
      })),
    [comercio],
  )
  const anos = [...series[0].valores.keys(), ...series[1].valores.keys()]
  const fim = anos.length > 0 ? Math.max(...anos) : 0
  const importado = series[0].valores.get(fim)
  const exportado = series[1].valores.get(fim)
  const volumes = useMemo(() => volumesEspumante(comercio, fim), [comercio, fim])

  return (
    <>
      <p>
        O vinho de mesa ao longo dos anos, em US$ por quilograma: o valor do ano dividido pelos
        quilogramas. A série em foco usa a cor de destaque.
        {importado !== undefined && exportado !== undefined
          ? ` Em ${fim} a importação ficou em ${usdPorKg(importado)} e a exportação em ${usdPorKg(exportado)}.`
          : ''}
      </p>
      <div className="preco-controles">
        <p className="preco-rotulo" id={controlesId}>
          Fluxo em foco
        </p>
        <div className="preco-pills" role="group" aria-labelledby={controlesId}>
          {FLUXOS.map((item) => (
            <button
              key={item.fluxo}
              type="button"
              className={foco === item.fluxo ? 'btn btn-solid' : 'btn btn-outline'}
              aria-pressed={foco === item.fluxo}
              onClick={() => setFoco(item.fluxo)}
            >
              {item.rotulo}
            </button>
          ))}
        </div>
      </div>
      <SerieAnual series={series} foco={foco} />
      <Recorte volumes={volumes} ano={fim} />
    </>
  )
}

export function Preco() {
  const estado = useComercio()

  return (
    <SectionFrame meta={sectionById('preco')}>
      <style>{css}</style>
      {estado.status === 'ready' ? (
        <Conteudo comercio={estado.comercio} />
      ) : (
        <p role={estado.status === 'error' ? 'alert' : 'status'}>
          {estado.status === 'error' ? 'Não foi possível ler os preços.' : 'Carregando os números.'}
        </p>
      )}
    </SectionFrame>
  )
}

const css = `
.preco-controles {
  margin-top: 22px;
}

.preco-rotulo {
  color: var(--ink-soft);
  font-size: 14px;
}

.preco-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.preco-figura {
  position: relative;
  margin: 32px 0 0;
}

.preco-titulo {
  font-family: var(--font-serif);
  font-size: 28px;
  font-weight: 400;
  letter-spacing: -0.02em;
  line-height: 1.15;
}

.preco-figura > p {
  margin-top: 12px;
}

.preco-legenda {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  margin-top: 14px;
}

.preco-legenda li {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--ink-soft);
  font-size: 14px;
}

.preco-amostra {
  display: inline-block;
  width: 18px;
  height: 8px;
  flex: none;
}

.preco-svg {
  display: block;
  width: 100%;
  height: auto;
  margin-top: 16px;
  touch-action: pan-y;
  user-select: none;
}

.preco-leitura {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 16px;
  max-width: none;
  margin-top: 12px;
  color: var(--ink);
  font-size: 14px;
  font-variant-numeric: tabular-nums;
}

.preco-ano {
  color: var(--ink-soft);
}

.preco-barras {
  margin-top: 18px;
}

.preco-linha {
  display: grid;
  grid-template-columns: 7.5rem minmax(0, 1fr);
  gap: 12px;
  align-items: center;
  width: 100%;
  padding: 7px 0;
  border: 0;
  background: transparent;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}

.preco-pais {
  font-size: 15px;
}

.preco-trilho {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.preco-barra {
  height: 12px;
  flex: none;
  width: calc(var(--fatia) * (100% - 16.5rem));
}

.preco-valor {
  display: inline-flex;
  gap: 8px;
  color: var(--ink);
  font-size: 14px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.preco-sr {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

@media (max-width: 700px) {
  .preco-linha {
    grid-template-columns: 1fr;
    gap: 4px;
  }

  .preco-trilho {
    flex-wrap: wrap;
  }

  .preco-barra {
    width: calc(var(--fatia) * 100%);
  }

  .preco-valor {
    flex-basis: 100%;
    white-space: normal;
  }
}
`
