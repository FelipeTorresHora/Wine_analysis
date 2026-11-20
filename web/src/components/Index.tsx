import { sections } from '../sections/manifest'

type Props = {
  activeId: string | null
  visible: boolean
}

export function Index({ activeId, visible }: Props) {
  return (
    <nav
      id="indice"
      className={visible ? 'index is-visible' : 'index'}
      aria-label="Nesta página"
      aria-hidden={visible ? undefined : true}
      inert={!visible}
    >
      <ol>
        {sections.map((section) => {
          const current = activeId === section.id
          return (
            <li key={section.id}>
              <a
                href={`#${section.id}`}
                aria-current={current ? 'location' : undefined}
                className={current ? 'is-active' : undefined}
              >
                <span className="index-number">{section.number}</span>
                <span className="index-label">
                  <span className="index-title">{section.title}</span>
                  <span className="index-short">{section.nav}</span>
                </span>
              </a>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
