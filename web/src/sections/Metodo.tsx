import { useEffect, useState } from 'react'
import { SectionFrame } from './SectionFrame'
import { sectionById } from './manifest'

type Meta = { fonte: string; inicio: number; fim: number }

type Estado = { status: 'loading' } | { status: 'error' } | { status: 'ready'; meta: Meta }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function lerMeta(data: unknown): Meta | null {
  if (!isRecord(data) || !isRecord(data.meta)) return null
  const { fonte, anos } = data.meta
  if (typeof fonte !== 'string' || fonte.trim() === '') return null
  if (!Array.isArray(anos) || anos.length === 0) return null
  const numeros = anos.filter((ano): ano is number => typeof ano === 'number' && Number.isFinite(ano))
  if (numeros.length !== anos.length) return null
  const ordenados = [...numeros].sort((a, b) => a - b)
  return { fonte: fonte.trim(), inicio: ordenados[0], fim: ordenados[ordenados.length - 1] }
}

function useMeta(): Estado {
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
        const meta = lerMeta(data)
        setEstado(meta ? { status: 'ready', meta } : { status: 'error' })
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

function Aviso() {
  return (
    <p className="metodo-aviso">
      <span className="metodo-olho">Fora desta leitura</span>
      O laboratório de qualidade usa o WineQT, química de vinho tinto de outro conjunto. Não é
      prova de vinho brasileiro.
    </p>
  )
}

function Texto({ meta }: { meta: Meta }) {
  return (
    <>
      <p>
        A fonte é {meta.fonte}. A série vai de {meta.inicio} a {meta.fim}.
      </p>
      <p>
        No comércio, cada ano traz um par: quilogramas e US$. O preço por quilograma é o valor
        dividido pela quantidade, e só entra quando há quilogramas.
      </p>
      <p>
        Na produção entram apenas os totais de família: vinho de mesa, vinho fino, suco e a soma
        dos espumantes. Esse total não se mistura com as subcategorias. Em {meta.fim}, o ponto de
        milhar foi lido como separador.
      </p>
      <p>
        Os nomes de país foram alinhados. Eslováquia fica como “Eslovaca, Republica”. As grafias da
        Coreia do Sul viram um só nome. A Geórgia do Sul permanece separada da Geórgia.
      </p>
      <Aviso />
    </>
  )
}

export function Metodo() {
  const estado = useMeta()

  return (
    <SectionFrame meta={sectionById('metodo')}>
      <style>{css}</style>
      {estado.status === 'ready' ? (
        <Texto meta={estado.meta} />
      ) : estado.status === 'error' ? (
        <>
          <p role="alert">Não foi possível ler a origem dos números.</p>
          <Aviso />
        </>
      ) : (
        <p role="status">Carregando os números.</p>
      )}
    </SectionFrame>
  )
}

const css = `
.section-slot .metodo-aviso {
  margin-top: 28px;
  padding-top: 20px;
  border-top: 1px solid var(--line);
}

.metodo-olho {
  display: block;
  margin-bottom: 6px;
  color: var(--ink-soft);
  font-size: 14px;
}
`
