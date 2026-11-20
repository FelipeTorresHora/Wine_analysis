import type { Destaque } from '../types/atlas'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

export function readDestaques(data: unknown): Destaque[] | null {
  if (!isRecord(data) || !Array.isArray(data.destaques) || data.destaques.length === 0) {
    return null
  }

  const list: Destaque[] = []
  for (const item of data.destaques) {
    if (!isRecord(item)) return null
    if (typeof item.valor !== 'string' || typeof item.frase !== 'string') return null
    list.push({
      id: typeof item.id === 'string' ? item.id : '',
      valor: item.valor,
      frase: item.frase,
    })
  }
  return list
}
