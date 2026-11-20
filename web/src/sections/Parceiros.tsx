import { SectionFrame } from './SectionFrame'
import { sectionById } from './manifest'

export function Parceiros() {
  return <SectionFrame meta={sectionById('parceiros')} />
}
