import { SectionFrame } from './SectionFrame'
import { sectionById } from './manifest'

export function Section03() {
  return <SectionFrame meta={sectionById('producao')} />
}
