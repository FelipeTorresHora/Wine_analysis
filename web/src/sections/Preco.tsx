import { SectionFrame } from './SectionFrame'
import { sectionById } from './manifest'

export function Preco() {
  return <SectionFrame meta={sectionById('preco')} />
}
