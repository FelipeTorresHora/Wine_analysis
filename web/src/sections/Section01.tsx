import { SectionFrame } from './SectionFrame'
import { sectionById } from './manifest'

export function Section01() {
  return <SectionFrame meta={sectionById('desequilibrio')} />
}
