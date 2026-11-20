export const sections = [
  {
    id: 'desequilibrio',
    number: '01',
    nav: 'Panorama',
    title: 'O tamanho do desequilíbrio',
    answer: 'Quanto o Brasil importa e quanto exporta.',
  },
  {
    id: 'parceiros',
    number: '02',
    nav: 'Parceiros',
    title: 'Com quem o Brasil negocia',
    answer: 'Quais países lideram cada fluxo.',
  },
  {
    id: 'producao',
    number: '03',
    nav: 'Produção',
    title: 'O que o país passou a produzir',
    answer: 'Como mudou o volume, em litros, por família.',
  },
  {
    id: 'preco',
    number: '04',
    nav: 'Preço',
    title: 'O preço por quilo',
    answer: 'O valor em US$ por quilograma, ao longo dos anos.',
  },
  {
    id: 'metodo',
    number: '05',
    nav: 'Método',
    title: 'Como estes números foram lidos',
    answer: 'A origem dos dados e o que ficou de fora.',
  },
] as const

export type SectionMeta = (typeof sections)[number]

export function sectionTitleId(id: string) {
  return `${id}-titulo`
}

export function sectionById(id: SectionMeta['id']) {
  const section = sections.find((item) => item.id === id)
  if (!section) throw new Error(`Seção ausente: ${id}`)
  return section
}
