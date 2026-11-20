import { useEffect, useState } from 'react'
import { sectionTitleId, sections } from '../sections/manifest'

export type NavTone = 'hero' | 'paper' | 'cta'

export function useShellScroll() {
  const [activeId, setActiveId] = useState<string | null>(null)
  const [indexVisible, setIndexVisible] = useState(false)
  const [navTone, setNavTone] = useState<NavTone>('hero')

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
      const closing = document.getElementById('continuar')
      const footer = document.getElementById('rodape')
      const navBottom = nav?.getBoundingClientRect().bottom ?? 0
      const pastHero = dark ? dark.getBoundingClientRect().bottom <= navBottom + 1 : false
      const closingTop = closing?.getBoundingClientRect().top ?? Number.POSITIVE_INFINITY
      const footerTop = footer?.getBoundingClientRect().top ?? Number.POSITIVE_INFINITY
      const overClosing = closingTop <= navBottom + 1 && footerTop > navBottom + 1
      const tone: NavTone = overClosing ? 'cta' : pastHero ? 'paper' : 'hero'

      const article = document.getElementById('artigo')
      const index = document.getElementById('indice')
      const indexBox = index?.getBoundingClientRect()
      const indexTop = indexBox?.top ?? 0
      const indexBottom = indexBox ? indexBox.bottom : indexTop
      const articleTop = article?.getBoundingClientRect().top ?? Number.POSITIVE_INFINITY
      const visible = article ? articleTop <= indexTop + 1 && closingTop > indexBottom : false

      setActiveId((prev) => (prev === current ? prev : current))
      setNavTone((prev) => (prev === tone ? prev : tone))
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

  return { activeId, indexVisible, navTone }
}
