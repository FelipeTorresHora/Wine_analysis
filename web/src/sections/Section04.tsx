import { SectionFrame } from './SectionFrame'
import { sectionById } from './manifest'

export function Section04() {
  return <SectionFrame meta={sectionById('preco')} />
}
