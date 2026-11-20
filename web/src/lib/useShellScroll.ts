import { useEffect, useState } from 'react'
import { sectionTitleId, sections } from '../sections/manifest'

export function useShellScroll() {
  const [activeId, setActiveId] = useState<string | null>(null)
  const [indexVisible, setIndexVisible] = useState(false)
  const [navOnPaper, setNavOnPaper] = useState(false)

  useEffect(() => {
    let frame = 0
    let mounted = true

    const update = () => {
      frame = 0
      if (!mounted) return

      const marker = window.innerHeight / 3
      let current: string | null = null
      for (const section of sections) {
        const heading = document.getElementById(sectionTitleId(section.id))
        if (!heading) continue
        if (heading.getBoundingClientRect().top <= marker) current = section.id
      }

      const nav = document.getElementById('navegacao')
      const dark = document.getElementById('faixa-escura')
      const navBottom = nav?.getBoundingClientRect().bottom ?? 0
      const onPaper = dark ? dark.getBoundingClientRect().bottom <= navBottom + 1 : false

      const article = document.getElementById('artigo')
      const index = document.getElementById('indice')
      const indexTop = index?.getBoundingClientRect().top ?? 0
      const visible = article ? article.getBoundingClientRect().top <= indexTop + 1 : false

      setActiveId((prev) => (prev === current ? prev : current))
      setNavOnPaper((prev) => (prev === onPaper ? prev : onPaper))
      setIndexVisible((prev) => (prev === visible ? prev : visible))
    }

    const request = () => {
      if (!mounted || frame) return
      frame = window.requestAnimationFrame(update)
    }

    update()
    window.addEventListener('scroll', request, { passive: true })
    window.addEventListener('resize', request)
    window.addEventListener('hashchange', request)
    document.fonts?.ready.then(() => request()).catch(() => undefined)

    return () => {
      mounted = false
      window.removeEventListener('scroll', request)
      window.removeEventListener('resize', request)
      window.removeEventListener('hashchange', request)
      if (frame) cancelAnimationFrame(frame)
    }
  }, [])

  return { activeId, indexVisible, navOnPaper }
}
