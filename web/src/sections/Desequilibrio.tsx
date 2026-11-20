import { SectionFrame } from './SectionFrame'
import { sectionById } from './manifest'

export function Desequilibrio() {
  return <SectionFrame meta={sectionById('desequilibrio')} />
}
