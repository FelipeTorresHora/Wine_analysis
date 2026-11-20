import { SectionFrame } from './SectionFrame'
import { sectionById } from './manifest'

export function Producao() {
  return <SectionFrame meta={sectionById('producao')} />
}
