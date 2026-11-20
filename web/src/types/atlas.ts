export type Fluxo = 'importacao' | 'exportacao'
export type Produto = 'mesa' | 'espumante'

export type Comercio = {
  ano: number
  fluxo: Fluxo
  produto: Produto
  pais: string
  kg: number
  usd: number
}

export type Producao = {
  ano: number
  familia: string
  litros: number
}

export type Preco = {
  ano: number
  fluxo: Fluxo
  produto: Produto
  pais: string
  usd_por_kg: number
}

export type Destaque = {
  id: string
  valor: string
  frase: string
}

export type Atlas = {
  meta: {
    fonte: string
    anos: number[]
    unidades: {
      kg: string
      usd: string
      litros: string
      preco: string
    }
  }
  destaques: Destaque[]
  comercio: Comercio[]
  producao: Producao[]
  preco: Preco[]
}
