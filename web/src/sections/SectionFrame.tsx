import type { ReactNode } from 'react'
import { sectionTitleId, type SectionMeta } from './manifest'

type Props = {
  meta: SectionMeta
  children?: ReactNode
}

/** Casca da seção. O conteúdo entra em `children`. */
export function SectionFrame({ meta, children }: Props) {
  const titleId = sectionTitleId(meta.id)

  return (
    <section id={meta.id} className="section" aria-labelledby={titleId}>
      <p className="section-number">{meta.number}</p>
      <h2 id={titleId} className="section-title">
        {meta.title}
      </h2>
      <div className="section-slot">{children}</div>
    </section>
  )
}
