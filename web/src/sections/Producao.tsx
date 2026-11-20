import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { PointerEvent } from 'react'
import type { Producao as RegistroProducao } from '../types/atlas'
import { SectionFrame } from './SectionFrame'
import { sectionById } from './manifest'

const SERIES = [
  { familia: 'VINHO DE MESA', rotulo: 'Mesa', cor: 'var(--ink)', traco: '' },
  { familia: 'SUCO', rotulo: 'Suco', cor: 'var(--accent)', traco: '', area: true },
  { familia: 'VINHO FINO DE MESA (VINIFERA)', rotulo: 'Fino', cor: 'var(--ink-soft)', traco: '' },
  { familia: 'ESPUMANTE', rotulo: 'Espumante', cor: 'var(--ink-soft)', traco: '5 4' },
] as const

const ORDEM_TRACO = [
  'VINHO FINO DE MESA (VINIFERA)',
  'ESPUMANTE',
  'VINHO DE MESA',
  'SUCO',
] as const

type Caixa = {
  largura: number
  altura: number
  esquerda: number
  direita: number
  topo: number
  base: number
}

type Estado =
  | { status: 'loading' }
  | { status: 'error' }
  | { status: 'ready'; registros: RegistroProducao[] }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function lerProducao(data: unknown): RegistroProducao[] | null {
  if (!isRecord(data) || !Array.isArray(data.producao)) return null
  const registros: RegistroProducao[] = []
  for (const item of data.producao) {
    if (!isRecord(item)) return null
    if (
      typeof item.ano !== 'number' ||
      typeof item.familia !== 'string' ||
      typeof item.litros !== 'number' ||
      !Number.isFinite(item.litros)
    ) {
      return null
    }
    registros.push({ ano: item.ano, familia: item.familia, litros: item.litros })
  }
  const familias = new Set(registros.map((registro) => registro.familia))
  const completa = SERIES.every((serie) => familias.has(serie.familia))
  return completa && registros.length > 0 ? registros : null
}

function useProducao(): Estado {
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
        const registros = lerProducao(data)
        setEstado(registros ? { status: 'ready', registros } : { status: 'error' })
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

function litrosPorExtenso(litros: number): string {
  if (litros <= 0) return '0 litros'
  const milhoes = litros / 1_000_000
  if (milhoes >= 0.95) {
    const texto = milhoes.toLocaleString('pt-BR', {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    })
    const unidade = milhoes < 1.95 ? 'milhão' : 'milhões'
    return `${texto} ${unidade} de litros`
  }
  const mil = litros / 1_000
  if (mil >= 10) return `${Math.round(mil).toLocaleString('pt-BR')} mil litros`
  return `${Math.round(litros).toLocaleString('pt-BR')} litros`
}

function litrosExatos(litros: number): string {
  return `${Math.round(litros).toLocaleString('pt-BR')} L`
}

function litrosDoEixo(litros: number): string {
  if (litros === 0) return '0'
  const milhoes = litros / 1_000_000
  return `${milhoes.toLocaleString('pt-BR', { maximumFractionDigits: 0 })} milhões`
}

function tetoLitros(maximo: number): number {
  const passo = 100_000_000
  if (maximo <= 0) return passo
  return Math.ceil((maximo * 1.02) / passo) * passo
}

function caixaDe(largura: number): Caixa {
  if (largura < 520) {
    return { largura, altura: 260, esquerda: 40, direita: 8, topo: 12, base: 24 }
  }
  return { largura, altura: 340, esquerda: 108, direita: 16, topo: 16, base: 32 }
}

function xDo(ano: number, inicio: number, fim: number, caixa: Caixa): number {
  const plot = caixa.largura - caixa.esquerda - caixa.direita
  if (fim === inicio || plot <= 0) return caixa.esquerda
  return caixa.esquerda + ((ano - inicio) / (fim - inicio)) * plot
}

function yDo(litros: number, teto: number, caixa: Caixa): number {
  const plot = caixa.altura - caixa.topo - caixa.base
  const valor = teto <= 0 ? 0 : Math.min(Math.max(litros, 0), teto)
  return caixa.topo + (1 - valor / teto) * plot
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

function areaDo(pontos: { x: number; y: number }[], base: number): string {
  if (pontos.length < 2) return ''
  const primeiro = pontos[0]
  const ultimo = pontos[pontos.length - 1]
  return `${caminho(pontos)} L${ultimo.x.toFixed(2)} ${base.toFixed(2)} L${primeiro.x.toFixed(2)} ${base.toFixed(2)} Z`
}

function Grafico({ registros }: { registros: RegistroProducao[] }) {
  const svgRef = useRef<SVGSVGElement>(null)
  const largura = useLargura(svgRef)
  const caixa = caixaDe(largura)
  const curto = largura < 520
  const tituloId = useId()
  const resumoId = useId()
  const porFamilia = useMemo(() => {
    const mapa = new Map<string, Map<number, number>>()
    for (const serie of SERIES) mapa.set(serie.familia, new Map())
    for (const registro of registros) {
      const serie = mapa.get(registro.familia)
      if (serie) serie.set(registro.ano, registro.litros)
    }
    return mapa
  }, [registros])

  const anos = useMemo(() => {
    const conjunto = new Set<number>()
    for (const registro of registros) conjunto.add(registro.ano)
    return [...conjunto].sort((a, b) => a - b)
  }, [registros])

  const inicio = anos[0] ?? 0
  const fim = anos[anos.length - 1] ?? inicio
  const [anoAtivo, setAnoAtivo] = useState(fim)

  const teto = useMemo(
    () => tetoLitros(Math.max(...registros.map((registro) => registro.litros), 0)),
    [registros],
  )

  const ler = (familia: string, ano: number) => porFamilia.get(familia)?.get(ano) ?? 0

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

  const marcasY: number[] = []
  for (let valor = 0; valor <= teto; valor += 100_000_000) marcasY.push(valor)

  const marcasX = marcasDoEixo(inicio, fim, caixa)

  const base = yDo(0, teto, caixa)
  const plot = caixa.largura - caixa.esquerda - caixa.direita
  const espaco = anos.length > 1 ? plot / (anos.length - 1) : plot
  const mesa0 = ler('VINHO DE MESA', inicio)
  const mesa1 = ler('VINHO DE MESA', fim)
  const suco0 = ler('SUCO', inicio)
  const suco1 = ler('SUCO', fim)
  const resumo = `De ${inicio} a ${fim}, o vinho de mesa passa de ${litrosPorExtenso(mesa0)} para ${litrosPorExtenso(mesa1)}. O suco passa de ${litrosPorExtenso(suco0)} para ${litrosPorExtenso(suco1)}.`

  return (
    <figure className="prod-figura">
      <figcaption className="prod-titulo" id={tituloId}>
        Litros, {inicio}–{fim}
        {curto ? <span className="prod-unidade">Milhões de litros</span> : null}
      </figcaption>
      <ul className="prod-legenda">
        {SERIES.map((serie) => (
          <li key={serie.familia}>
            {serie.traco ? (
              <svg className="prod-amostra" width="18" height="8" aria-hidden="true">
                <line
                  x1="0"
                  y1="4"
                  x2="18"
                  y2="4"
                  stroke={serie.cor}
                  strokeWidth="2"
                  strokeDasharray={serie.traco}
                />
              </svg>
            ) : (
              <span className="prod-amostra" style={{ background: serie.cor }} aria-hidden="true" />
            )}
            {serie.rotulo}
          </li>
        ))}
      </ul>
      <svg
        ref={svgRef}
        className="prod-svg"
        viewBox={`0 0 ${caixa.largura} ${caixa.altura}`}
        role="img"
        aria-labelledby={tituloId}
        aria-describedby={resumoId}
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
            {curto ? (marca === 0 ? '0' : Math.round(marca / 1_000_000).toLocaleString('pt-BR')) : litrosDoEixo(marca)}
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
        {ORDEM_TRACO.map((familia) => {
          const serie = SERIES.find((item) => item.familia === familia)
          if (!serie) return null
          const pontos = anos.map((ano) => {
            const litros = ler(familia, ano)
            return { ano, litros, x: xDo(ano, inicio, fim, caixa), y: yDo(litros, teto, caixa) }
          })
          const linha = pontos.filter((_, indice) => {
            if (familia !== 'ESPUMANTE') return true
            const primeiro = pontos.findIndex((item) => item.litros > 0)
            return primeiro !== -1 && indice >= primeiro
          })
          return (
            <g key={familia}>
              {'area' in serie && serie.area ? (
                <path d={areaDo(linha, base)} fill={serie.cor} opacity="0.14" />
              ) : null}
              <path
                d={caminho(linha)}
                fill="none"
                stroke={serie.cor}
                strokeWidth="2"
                strokeLinejoin="miter"
                strokeLinecap="butt"
                strokeDasharray={serie.traco || undefined}
              />
              {linha.map((ponto) =>
                (ponto.litros > 0 || ponto.ano === anoAtivo) &&
                (ponto.ano === anoAtivo || espaco >= 8 || (ponto.ano - inicio) % 5 === 0) ? (
                  <circle
                    key={ponto.ano}
                    cx={ponto.x}
                    cy={ponto.y}
                    r={ponto.ano === anoAtivo ? 4 : 2}
                    fill={serie.cor}
                  />
                ) : null,
              )}
            </g>
          )
        })}
      </svg>
      <p className="prod-leitura" id={resumoId}>
        <span className="prod-ano">{anoAtivo}</span>
        {SERIES.map((serie) => (
          <span key={serie.familia}>
            {serie.rotulo} {litrosExatos(ler(serie.familia, anoAtivo))}
          </span>
        ))}
      </p>
      <p className="prod-sr">{resumo}</p>
      <div className="prod-sr">
      <table>
        <caption>Litros por família e ano. Só os totais.</caption>
        <thead>
          <tr>
            <th scope="col">Ano</th>
            {SERIES.map((serie) => (
              <th key={serie.familia} scope="col">
                {serie.rotulo}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {anos.map((ano) => (
            <tr key={ano}>
              <th scope="row">{ano}</th>
              {SERIES.map((serie) => (
                <td key={serie.familia}>{litrosExatos(ler(serie.familia, ano))}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </figure>
  )
}

function Texto({ registros }: { registros: RegistroProducao[] }) {
  const anos = [...new Set(registros.map((registro) => registro.ano))].sort((a, b) => a - b)
  const inicio = anos[0]
  const fim = anos[anos.length - 1]
  const ler = (familia: string, ano: number) =>
    registros.find((registro) => registro.familia === familia && registro.ano === ano)?.litros ?? 0

  return (
    <>
      <p>
        Só entram os totais de cada família, em litros: mesa, fino, suco e espumante. O total da
        família não é somado outra vez com as subcategorias.
      </p>
      <p>
        O vinho de mesa passou de {litrosPorExtenso(ler('VINHO DE MESA', inicio))} em {inicio} para{' '}
        {litrosPorExtenso(ler('VINHO DE MESA', fim))} em {fim}. O suco de uva subiu de{' '}
        {litrosPorExtenso(ler('SUCO', inicio))} para {litrosPorExtenso(ler('SUCO', fim))}.
      </p>
      <p>
        O fino passou de {litrosPorExtenso(ler('VINHO FINO DE MESA (VINIFERA)', inicio))} para{' '}
        {litrosPorExtenso(ler('VINHO FINO DE MESA (VINIFERA)', fim))}. O espumante somou{' '}
        {litrosPorExtenso(ler('ESPUMANTE', inicio))} em {inicio} e{' '}
        {litrosPorExtenso(ler('ESPUMANTE', fim))} em {fim}.
      </p>
      <Grafico registros={registros} />
    </>
  )
}

export function Producao() {
  const estado = useProducao()

  return (
    <SectionFrame meta={sectionById('producao')}>
      <style>{css}</style>
      {estado.status === 'ready' ? (
        <Texto registros={estado.registros} />
      ) : (
        <p role={estado.status === 'error' ? 'alert' : 'status'}>
          {estado.status === 'error'
            ? 'Não foi possível ler a produção.'
            : 'Carregando os números.'}
        </p>
      )}
    </SectionFrame>
  )
}

const css = `
.prod-figura {
  position: relative;
  margin: 32px 0 0;
}

.prod-titulo {
  font-family: var(--font-serif);
  font-size: 28px;
  font-weight: 400;
  letter-spacing: -0.02em;
  line-height: 1.15;
}

.prod-unidade {
  display: block;
  margin-top: 4px;
  color: var(--ink-soft);
  font-family: var(--font-sans);
  font-size: 13px;
  letter-spacing: 0;
}

.prod-legenda {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  margin-top: 14px;
}

.prod-legenda li {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--ink-soft);
  font-size: 14px;
}

.prod-amostra {
  display: inline-block;
  width: 18px;
  height: 8px;
  flex: none;
}

.prod-svg {
  display: block;
  width: 100%;
  height: auto;
  margin-top: 16px;
  touch-action: pan-y;
  user-select: none;
}

.prod-leitura {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 16px;
  max-width: none;
  margin-top: 12px;
  color: var(--ink);
  font-size: 14px;
  font-variant-numeric: tabular-nums;
}

.prod-ano {
  color: var(--ink-soft);
}

.prod-sr {
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
`
