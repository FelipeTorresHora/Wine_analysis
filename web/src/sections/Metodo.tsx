import { SectionFrame } from './SectionFrame'
import { sectionById } from './manifest'

export function Metodo() {
  return <SectionFrame meta={sectionById('metodo')} />
}
