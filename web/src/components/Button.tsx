import type { ReactNode } from 'react'

type Variant = 'solid' | 'outline' | 'outline-light'

type Props = {
  href: string
  variant: Variant
  children: ReactNode
}

export function Button({ href, variant, children }: Props) {
  return (
    <a className={`btn btn-${variant}`} href={href}>
      {children}
    </a>
  )
}
