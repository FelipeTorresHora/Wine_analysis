import { SectionFrame } from './SectionFrame'
import { sectionById } from './manifest'

export function Section02() {
  return <SectionFrame meta={sectionById('parceiros')} />
}
